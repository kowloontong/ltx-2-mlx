"""Two-stage pipeline — dev model + CFG at half res, upscale, distilled LoRA refine.

Matches the reference architecture:
  Stage 1: Dev (non-distilled) model + CFG guidance at half resolution.
  Stage 2: Dev + distilled LoRA fused, simple denoising at full resolution.

Requires the dev model + distilled LoRA weights (e.g. dgrauet/ltx-2.3-mlx-q8).

Ported from ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import (
    LatentState,
    VideoConditionByLatentIndex,
    apply_conditioning,
    create_initial_state,
    noise_latent_state,
)
from ltx_core_mlx.loader.fuse_loras import apply_loras
from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core_mlx.model.transformer.model import LTXModel, X0Model
from ltx_core_mlx.model.upsampler import LatentUpsampler
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
from ltx_core_mlx.utils.image import prepare_image_for_encoding
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors
from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS, ltx2_schedule
from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
from ltx_pipelines_mlx.utils.samplers import denoise_loop, guided_denoise_loop

# Reference defaults
DEFAULT_CFG_SCALE = 3.0


def _remap_lora_keys(lora_sd: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap LoRA keys from ComfyUI/diffusion_model format to MLX model format."""
    remapped: dict[str, mx.array] = {}
    for key, value in lora_sd.items():
        new_key = LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key(key)
        new_key = new_key.replace(".linear_1.", ".linear1.").replace(".linear_2.", ".linear2.")
        new_key = new_key.replace("audio_ff.net.0.proj.", "audio_ff.proj_in.")
        new_key = new_key.replace("audio_ff.net.2.", "audio_ff.proj_out.")
        remapped[new_key] = value
    return remapped


class TwoStagePipeline(TextToVideoPipeline):
    """Two-stage generation: dev model + CFG at half-res, upscale, distilled LoRA refine.

    Stage 1: Dev model + CFG guidance at half resolution (Euler sampler).
    Stage 2: Dev + distilled LoRA fused, simple denoising at full resolution.

    Requires ``dev_transformer`` and ``distilled_lora`` — the two-stage pipeline
    needs the dev model for quality generation at half resolution with CFG.

    Args:
        model_dir: Path to model weights or HuggingFace repo ID.
        gemma_model_id: Gemma model for text encoding.
        low_memory: Aggressive memory management.
        dev_transformer: Dev transformer filename (e.g. ``transformer-dev.safetensors``).
        distilled_lora: Distilled LoRA filename for Stage 2.
        distilled_lora_strength: LoRA fusion strength (default 1.0).
    """

    def __init__(
        self,
        model_dir: str,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        dev_transformer: str = "transformer-dev.safetensors",
        distilled_lora: str = "ltx-2.3-22b-distilled-lora-384.safetensors",
        distilled_lora_strength: float = 1.0,
    ):
        super().__init__(model_dir, gemma_model_id=gemma_model_id, low_memory=low_memory)
        self._dev_transformer = dev_transformer
        self._distilled_lora = distilled_lora
        self._distilled_lora_strength = distilled_lora_strength
        self.upsampler: LatentUpsampler | None = None
        self.vae_encoder: VideoEncoder | None = None

    def _load_dev_transformer(self) -> LTXModel:
        """Load the dev (non-distilled) transformer weights."""
        dev_path = self.model_dir / self._dev_transformer
        if not dev_path.exists():
            raise FileNotFoundError(
                f"Dev transformer not found: {dev_path}\n"
                "Two-stage requires the dev model + distilled LoRA.\n"
                "Use: --model dgrauet/ltx-2.3-mlx-q8"
            )
        dit = LTXModel()
        weights = load_split_safetensors(dev_path, prefix="transformer.")
        apply_quantization(dit, weights)
        dit.load_weights(list(weights.items()))
        aggressive_cleanup()
        return dit

    def _fuse_distilled_lora(self, dit: LTXModel) -> None:
        """Fuse distilled LoRA weights into a loaded transformer in-place."""
        lora_path = self.model_dir / self._distilled_lora
        if not lora_path.exists():
            raise FileNotFoundError(
                f"Distilled LoRA not found: {lora_path}\n"
                "Two-stage requires the distilled LoRA for Stage 2.\n"
                "Use: --model dgrauet/ltx-2.3-mlx-q8"
            )
        lora_raw = dict(mx.load(str(lora_path)))
        lora_remapped = _remap_lora_keys(lora_raw)

        import mlx.utils

        flat_params = mlx.utils.tree_flatten(dit.parameters())
        flat_model = {k: v for k, v in flat_params if isinstance(v, mx.array)}

        model_sd = StateDict(sd=flat_model, size=0, dtype=set())
        lora_sd = StateDict(sd=lora_remapped, size=0, dtype=set())
        lora_with_strength = LoraStateDictWithStrength(lora_sd, self._distilled_lora_strength)

        fused = apply_loras(model_sd, [lora_with_strength])
        dit.load_weights(list(fused.sd.items()))
        aggressive_cleanup()

    def _load_vae_encoder(self) -> None:
        """Load VAE encoder (for per-channel normalization stats + I2V encoding)."""
        if self.vae_encoder is not None:
            return
        self.vae_encoder = VideoEncoder()
        enc_weights = load_split_safetensors(self.model_dir / "vae_encoder.safetensors", prefix="vae_encoder.")
        enc_weights = {
            k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
            for k, v in enc_weights.items()
        }
        self.vae_encoder.load_weights(list(enc_weights.items()))
        aggressive_cleanup()

    def _load_upsampler(self, name: str = "spatial_upscaler_x2_v1_1") -> None:
        """Load upsampler from config and weights."""
        import json

        config_path = self.model_dir / f"{name}_config.json"
        weights_path = self.model_dir / f"{name}.safetensors"

        if config_path.exists():
            config = json.loads(config_path.read_text()).get("config", {})
            self.upsampler = LatentUpsampler.from_config(config)
        else:
            self.upsampler = LatentUpsampler()

        if weights_path.exists():
            weights = load_split_safetensors(weights_path, prefix=f"{name}.")
            self.upsampler.load_weights(list(weights.items()))
        aggressive_cleanup()

    def load(self) -> None:
        """Load DiT + VAE encoder + upsampler (skip decoders for memory).

        Decoders are loaded on-demand in ``generate_and_save()``.
        """
        if self._loaded:
            return

        model_dir = self.model_dir

        # Text encoder + connector
        if self.text_encoder is None:
            from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

            self.text_encoder = GemmaLanguageModel()
            self.text_encoder.load(self._gemma_model_id)
            aggressive_cleanup()

        if self.feature_extractor is None:
            from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2

            self.feature_extractor = GemmaFeaturesExtractorV2()
            connector_weights = load_split_safetensors(model_dir / "connector.safetensors", prefix="connector.")
            self.feature_extractor.connector.load_weights(list(connector_weights.items()))
            aggressive_cleanup()

        # DiT (dev model)
        if self.dit is None:
            if self.low_memory:
                self.text_encoder = None
                aggressive_cleanup()

            self.dit = self._load_dev_transformer()

        # VAE encoder (for denorm/renorm + optional I2V)
        self._load_vae_encoder()

        # Upsampler
        if self.upsampler is None:
            self._load_upsampler()

        self._loaded = True

    def _load_decoders(self) -> None:
        """Load decoders on-demand (VAE decoder, audio decoder, vocoder)."""
        from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
        from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
        from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
        from ltx_core_mlx.utils.weights import remap_audio_vae_keys

        model_dir = self.model_dir

        if self.vae_decoder is None:
            self.vae_decoder = VideoDecoder()
            vae_weights = load_split_safetensors(model_dir / "vae_decoder.safetensors", prefix="vae_decoder.")
            self.vae_decoder.load_weights(list(vae_weights.items()))
            aggressive_cleanup()

        if self.audio_decoder is None:
            self.audio_decoder = AudioVAEDecoder()
            audio_weights = load_split_safetensors(model_dir / "audio_vae.safetensors", prefix="audio_vae.decoder.")
            all_audio = load_split_safetensors(model_dir / "audio_vae.safetensors", prefix="audio_vae.")
            for k, v in all_audio.items():
                if k.startswith("per_channel_statistics."):
                    audio_weights[k] = v
            audio_weights = remap_audio_vae_keys(audio_weights)
            self.audio_decoder.load_weights(list(audio_weights.items()))
            aggressive_cleanup()

        if self.vocoder is None:
            self.vocoder = VocoderWithBWE()
            vocoder_weights = load_split_safetensors(model_dir / "vocoder.safetensors", prefix="vocoder.")
            self.vocoder.load_weights(list(vocoder_weights.items()))
            aggressive_cleanup()

    def generate_two_stage(
        self,
        prompt: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        stage1_steps: int = 20,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = 0.0,
        image: str | None = None,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Generate video using two-stage pipeline.

        Args:
            prompt: Text prompt.
            height: Final video height.
            width: Final video width.
            num_frames: Number of frames.
            seed: Random seed.
            stage1_steps: Denoising steps for stage 1 (default: 20).
            stage2_steps: Denoising steps for stage 2.
            cfg_scale: CFG guidance scale for stage 1 (default: 3.0).
            stg_scale: STG guidance scale for stage 1 (default: 0.0).
            image: Optional reference image for I2V conditioning.
            video_guider_params: Optional full guider params for video.
            audio_guider_params: Optional full guider params for audio.

        Returns:
            Tuple of (video_latent, audio_latent) at full resolution.
        """
        # --- Text encoding ---
        if self.text_encoder is None or self.feature_extractor is None:
            model_dir = self.model_dir
            if self.text_encoder is None:
                from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

                self.text_encoder = GemmaLanguageModel()
                self.text_encoder.load(self._gemma_model_id)
                aggressive_cleanup()
            if self.feature_extractor is None:
                from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2

                self.feature_extractor = GemmaFeaturesExtractorV2()
                conn_weights = load_split_safetensors(model_dir / "connector.safetensors", prefix="connector.")
                self.feature_extractor.connector.load_weights(list(conn_weights.items()))
                aggressive_cleanup()

        video_embeds, audio_embeds = self._encode_text(prompt)

        # Encode negative prompt for CFG
        from ltx_pipelines_mlx.utils.constants import DEFAULT_NEGATIVE_PROMPT

        neg_video_embeds, neg_audio_embeds = self._encode_text(DEFAULT_NEGATIVE_PROMPT)

        mx.eval(video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds)

        # Free text encoder before loading heavy components
        self.text_encoder = None
        self.feature_extractor = None
        aggressive_cleanup()

        # --- Load DiT + VAE encoder + upsampler ---
        if self.dit is None:
            self.dit = self._load_dev_transformer()

        self._load_vae_encoder()
        if self.upsampler is None:
            self._load_upsampler()

        assert self.dit is not None
        assert self.vae_encoder is not None
        assert self.upsampler is not None

        # --- Stage 1: Half resolution with CFG ---
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half)
        audio_positions = compute_audio_positions(audio_T)

        video_state = create_initial_state(video_shape, seed, positions=video_positions_1)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # I2V conditioning at half resolution
        enc_h_half = H_half * 32
        enc_w_half = W_half * 32
        conditionings_1: list[VideoConditionByLatentIndex] = []
        if image is not None:
            img_tensor = prepare_image_for_encoding(image, enc_h_half, enc_w_half)
            img_tensor = img_tensor[:, :, None, :, :]
            ref_latent = self.vae_encoder.encode(img_tensor)
            ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
            conditionings_1.append(
                VideoConditionByLatentIndex(
                    frame_indices=[0],
                    clean_latent=ref_tokens,
                    strength=1.0,
                )
            )

        if conditionings_1:
            video_state = apply_conditioning(video_state, conditionings_1, (F, H_half, W_half))
            video_state = noise_latent_state(video_state, sigma=1.0, seed=seed)
            audio_state = noise_latent_state(audio_state, sigma=1.0, seed=seed + 1)

        # Stage 1 sigma schedule (dynamic for dev model)
        num_tokens = F * H_half * W_half
        sigmas_1 = ltx2_schedule(stage1_steps, num_tokens=num_tokens)
        x0_model = X0Model(self.dit)

        # Build guider params
        if video_guider_params is None:
            video_guider_params = MultiModalGuiderParams(cfg_scale=cfg_scale, stg_scale=stg_scale)
        if audio_guider_params is None:
            audio_guider_params = MultiModalGuiderParams(cfg_scale=cfg_scale, stg_scale=stg_scale)

        video_factory = create_multimodal_guider_factory(video_guider_params, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_guider_params, negative_context=neg_audio_embeds)

        output_1 = guided_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas_1,
        )
        if self.low_memory:
            aggressive_cleanup()

        # --- Fuse distilled LoRA for Stage 2 ---
        self._fuse_distilled_lora(self.dit)

        # --- Upscale with denormalize/renormalize ---
        video_half = self.video_patchifier.unpatchify(output_1.video_latent, (F, H_half, W_half))

        video_mlx = video_half.transpose(0, 2, 3, 4, 1)  # (B,C,F,H,W) -> (B,F,H,W,C)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        mx.eval(video_upscaled)

        # Derive full-resolution dims from actual upscaled shape
        H_full = H_half * 2
        W_full = W_half * 2

        # I2V conditioning at full resolution for Stage 2
        conditionings_2: list[VideoConditionByLatentIndex] = []
        if image is not None:
            enc_h_full = H_full * 32
            enc_w_full = W_full * 32
            img_tensor = prepare_image_for_encoding(image, enc_h_full, enc_w_full)
            img_tensor = img_tensor[:, :, None, :, :]
            ref_latent = self.vae_encoder.encode(img_tensor)
            ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
            conditionings_2.append(
                VideoConditionByLatentIndex(
                    frame_indices=[0],
                    clean_latent=ref_tokens,
                    strength=1.0,
                )
            )

        # Free VAE encoder + upsampler before Stage 2 denoising
        if self.low_memory:
            self.vae_encoder = None
            self.upsampler = None
            aggressive_cleanup()

        # --- Stage 2: Refine at full resolution (no CFG) ---
        video_tokens, _ = self.video_patchifier.patchify(video_upscaled)

        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        mx.random.seed(seed + 2)
        noise = mx.random.normal(video_tokens.shape).astype(mx.bfloat16)
        noisy_tokens = noise * start_sigma + video_tokens * (1.0 - start_sigma)

        video_positions_2 = compute_video_positions(F, H_full, W_full)

        video_state_2 = LatentState(
            latent=noisy_tokens,
            clean_latent=video_tokens,
            denoise_mask=mx.ones((1, video_tokens.shape[1], 1), dtype=mx.bfloat16),
            positions=video_positions_2,
        )

        if conditionings_2:
            video_state_2 = apply_conditioning(video_state_2, conditionings_2, (F, H_full, W_full))

        audio_tokens_1 = output_1.audio_latent
        audio_state_2 = LatentState(
            latent=audio_tokens_1,
            clean_latent=audio_tokens_1,
            denoise_mask=mx.ones((1, audio_tokens_1.shape[1], 1), dtype=audio_tokens_1.dtype),
            positions=audio_positions,
        )
        audio_state_2 = noise_latent_state(audio_state_2, sigma=start_sigma, seed=seed + 2)

        output_2 = denoise_loop(
            model=x0_model,
            video_state=video_state_2,
            audio_state=audio_state_2,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_2,
        )
        if self.low_memory:
            aggressive_cleanup()

        video_latent = self.video_patchifier.unpatchify(output_2.video_latent, (F, H_full, W_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)

        return video_latent, audio_latent

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        seed: int = 42,
        stage1_steps: int = 20,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        stg_scale: float = 0.0,
        image: str | None = None,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
    ) -> str:
        """Generate two-stage video+audio and save to file."""
        video_latent, audio_latent = self.generate_two_stage(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            image=image,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
        )

        # Free transformer + encoder to make room for decoders
        if self.low_memory:
            self.dit = None
            self.text_encoder = None
            self.feature_extractor = None
            self.vae_encoder = None
            self.upsampler = None
            self._loaded = False
            aggressive_cleanup()

        # Load decoders on-demand
        self._load_decoders()

        # Decode audio first (smaller)
        assert self.audio_decoder is not None
        assert self.vocoder is not None
        mel = self.audio_decoder.decode(audio_latent)
        waveform = self.vocoder(mel)
        if self.low_memory:
            aggressive_cleanup()

        import tempfile

        audio_path = tempfile.mktemp(suffix=".wav")
        self._save_waveform(waveform, audio_path, sample_rate=48000)

        assert self.vae_decoder is not None
        self.vae_decoder.decode_and_stream(video_latent, output_path, fps=24.0, audio_path=audio_path)

        Path(audio_path).unlink(missing_ok=True)
        aggressive_cleanup()

        return output_path
