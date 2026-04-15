"""Prompt enhancement using Qwen3.5 GGUF model."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

# System prompts for different modes
T2V_SYSTEM_PROMPT = """You are a Creative Assistant writing concise, action-focused text-to-video prompts.
Generate a prompt to guide video generation from the user's input.

Guidelines:
- Subject & Action: Identify main subject and primary action.
- Setting: Specify environment and atmosphere.
- Camera: Describe camera movement if requested.
- Audio: Describe complete soundscape.
- Style: Include visual style at the beginning.
- Output: Single continuous paragraph, no Markdown."""

I2V_SYSTEM_PROMPT = """You are a Creative Assistant writing image-to-video prompts.
Generate a prompt to animate the given image.

Guidelines:
- Analyze the image: Identify subject, setting, style.
- Describe only changes from the image.
- Active language: Use present-progressive verbs.
- Audio: Describe soundscape.
- Output: Single concise paragraph."""

JOINT_I2V_SYSTEM_PROMPT = """You are generating an I2V prompt to animate an existing image.

Guidelines:
- The video should be CONTINUOUS MOTION from the static image.
- Every element should come ALIVE with subtle or dynamic motion.
- NO cuts or scene transitions — single continuous shot.
- Output: Single continuous paragraph starting with motion/action."""


class PromptEnhancer:
    """Prompt enhancer using Qwen3.5 GGUF model."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._llm = None

    def _get_llm(self):
        """Lazy-load LLM model."""
        if self._llm is None:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=False,
            )
        return self._llm

    def enhance_prompt(self, raw_prompt: str, mode: str = "t2v", seed: int = 10) -> str:
        """Enhance a prompt using LLM.

        Args:
            raw_prompt: Raw user prompt.
            mode: "t2v" or "i2v".
            seed: Random seed.

        Returns:
            Enhanced prompt.
        """
        import gc

        system_prompt = T2V_SYSTEM_PROMPT if mode == "t2v" else I2V_SYSTEM_PROMPT
        user_content = f"User prompt: {raw_prompt}"

        llm = self._get_llm()
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=512,
            temperature=0.7,
            seed=seed,
        )

        enhanced = result["choices"][0]["message"]["content"].strip()
        gc.collect()
        return enhanced

    def extract_scene_and_motion(self, flux_prompt: str, seed: int = 10) -> Tuple[str, str]:
        """Extract scene keywords and motion from Flux prompt.

        Args:
            flux_prompt: Flux image prompt.
            seed: Random seed.

        Returns:
            Tuple of (scene_keywords, motion_description).
        """
        llm = self._get_llm()

        extract_prompt = """Extract and return ONLY these two lines:

SCENE_KEYWORDS: <keywords>
MOTION: <motion description>

Your output (two lines only):"""

        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": extract_prompt},
                {"role": "user", "content": flux_prompt},
            ],
            max_tokens=256,
            temperature=0.3,
            seed=seed,
        )

        output = result["choices"][0]["message"]["content"].strip()

        scene_keywords = ""
        motion_description = ""

        for line in output.split("\n"):
            if line.startswith("SCENE_KEYWORDS:"):
                scene_keywords = line.replace("SCENE_KEYWORDS:", "").strip()
            elif line.startswith("MOTION:"):
                motion_description = line.replace("MOTION:", "").strip()

        return scene_keywords, motion_description

    def generate_i2v_prompt(self, scene_keywords: str, motion_description: str, seed: int = 10) -> str:
        """Generate I2V prompt from scene and motion.

        Args:
            scene_keywords: Scene keywords.
            motion_description: Motion description.
            seed: Random seed.

        Returns:
            I2V prompt.
        """
        llm = self._get_llm()

        user_content = f"""Scene Keywords: {scene_keywords}
Motion Description: {motion_description}"""

        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": JOINT_I2V_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=512,
            temperature=0.7,
            seed=seed,
        )

        return result["choices"][0]["message"]["content"].strip()

    def enhance_flash_prompt(self, scene_description: str, seed: int = 10) -> str:
        """Enhance flash scene description with motion.

        Args:
            scene_description: Scene description.
            seed: Random seed.

        Returns:
            Enhanced prompt with motion.
        """
        # Simple rule-based enhancement for flash scenes
        enhanced = scene_description

        motion_map = {
            "neon rain": "heavy neon rain pouring down streets, raindrops splashing",
            "flying cars": "flying cars zooming through neon-lit alleys, light trails",
            "dust storm": "massive dust storm swirling violently",
            "neon lights": "neon lights flickering intensely, moving shadows",
            "holographic": "holographic displays flickering, projecting moving figures",
            "crowds": "busy crowds rushing past, moving chaotically",
        }

        for keyword, motion in motion_map.items():
            if keyword.lower() in enhanced.lower():
                enhanced = enhanced.replace(keyword, motion)

        if "camera" not in enhanced.lower():
            enhanced = enhanced + ", dynamic tracking camera movement"

        return enhanced
