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

    def enhance_flash_prompt(self, scene_description: str, seed: int = 10, mode: str = "t2v") -> str:
        """Enhance flash scene description with motion.

        Args:
            scene_description: Scene description.
            seed: Random seed.
            mode: "t2v" or "i2v" - I2V needs more motion emphasis.

        Returns:
            Enhanced prompt with motion.
        """
        # For I2V mode, emphasize specific motions to animate the static image
        if mode == "i2v":
            # I2V-specific motion patterns - focus on animating existing elements
            motion_patterns = {
                "neon rain": "raindrops falling continuously, water rippling on surfaces, neon reflections shimmering, puddles splashing with each drop",
                "flying cars": "vehicles moving smoothly through frame, lights trailing behind, slight camera pan following motion",
                "dust storm": "dust particles swirling in wind, debris floating past, atmospheric haze shifting",
                "neon lights": "lights pulsing gently, shadows shifting subtly, glow breathing in and out",
                "holographic": "hologram flickering with static, projection wavering slightly, colors shifting",
                "crowds": "people walking naturally, subtle head movements, clothes swaying, ambient motion",
                "city": "distant lights twinkling, traffic flowing in background, atmospheric movement",
                "street": "leaves rustling, distant movement, ambient street activity",
                "building": "lights turning on/off, windows flickering, subtle environmental motion",
                "sky": "clouds drifting slowly, light changing gradually, atmospheric shift",
                "bazaars": "vendors moving slowly, fabric swaying, steam rising, ambient market sounds",
                "vault": "machines humming, lights flickering, steam venting, mechanical sounds",
                "skyline": "lights blinking, distant movement, atmospheric haze shifting",
                "graveyard": "rust settling, debris floating, wind blowing through metal, eerie silence",
                "ruins": "hologram glitching, static interference, projection unstable, artifacts appearing",
            }
            
            enhanced = scene_description
            
            # Apply motion patterns
            motion_applied = False
            for keyword, motion in motion_patterns.items():
                if keyword.lower() in enhanced.lower():
                    enhanced = enhanced + f", {motion}"
                    motion_applied = True
                    break
            
            # If no specific pattern matched, add generic subtle motion
            if not motion_applied:
                enhanced = enhanced + ", subtle ambient motion, gentle environmental movement, natural animation"
            
            # Add camera motion for I2V (gentle, not jarring)
            if "camera" not in enhanced.lower():
                enhanced = enhanced + ", smooth subtle camera drift"
            
            return enhanced
        
        # For T2V mode, use original aggressive motion enhancement
        else:
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
    
    def enhance_flux_prompt(self, scene_description: str, seed: int = 10) -> str:
        """Enhance Flux prompt for better image quality.

        Args:
            scene_description: Scene description.
            seed: Random seed.

        Returns:
            Enhanced prompt for Flux image generation.
        """
        # Add quality and detail keywords for Flux
        quality_keywords = [
            "highly detailed",
            "intricate details",
            "sharp focus",
            "professional photography",
            "8k resolution",
            "cinematic lighting",
            "dramatic atmosphere",
        ]
        
        # Scene-specific enhancements
        scene_enhancements = {
            "bazaars": "bustling market stalls, colorful fabrics, steam and smoke, crowded aisles",
            "vault": "dark industrial interior, mechanical machinery, glowing panels, metal surfaces",
            "skyline": "towering buildings, neon signs, flying vehicles, atmospheric perspective",
            "graveyard": "rusted metal hulks, overgrown vegetation, scattered debris, moody lighting",
            "ruins": "crumbling structures, holographic projections, glitching displays, decay",
        }
        
        enhanced = scene_description
        
        # Add scene-specific details
        for keyword, details in scene_enhancements.items():
            if keyword.lower() in enhanced.lower():
                enhanced = enhanced + f", {details}"
                break
        
        # Add quality keywords
        enhanced = enhanced + ", " + ", ".join(quality_keywords[:3])
        
        return enhanced
