import streamlit as st
import subprocess
import os
import time
import json
from pathlib import Path

FLUX_SCRIPT_DIR = "/Users/junhui/work/flux-gguf-image-gen"
FLUX_MODEL_DIR = os.path.join(FLUX_SCRIPT_DIR, "models")
FLUX_OUTPUT_DIR = "outputs/webui/flux"

MODEL_DIR = "models/ltx-2.3-mlx-q8"
OUTPUT_DIR = "outputs/webui"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FLUX_OUTPUT_DIR, exist_ok=True)

# Joint workflow state
if "joint_workflow" not in st.session_state:
    st.session_state.joint_workflow = {
        "flux_image": None,
        "flux_params": {},
        "flux_prompt": "",
        "scene_keywords": "",
        "motion_description": "",
        "i2v_prompt": None,
        "video_path": None,
        "step": None,
    }

if "flash_workflow" not in st.session_state:
    st.session_state.flash_workflow = {
        "theme_words": "",
        "scene_count": 5,
        "frames_per_segment": 49,
        "seed": 42,
        "scenes": [],
        "segments": [],
        "status": "idle",
        "current_segment": 0,
        "output_path": None,
        "temp_dir": None,
    }

st.set_page_config(page_title="LTX-2 & FLUX Generator", page_icon="🎬", layout="wide")

st.markdown(
    """
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton > button {
        background-color: #4A90D9;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #357ABD;
    }
    .success-box {
        background-color: #D4EDDA;
        border: 1px solid #C3E6CB;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
    }
    .error-box {
        background-color: #F8D7DA;
        border: 1px solid #F5C6CB;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_video_info(video_path):
    """Get video file info using ffprobe"""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout)
        video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
        duration = float(data["format"].get("duration", 0))
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        return {"duration": f"{duration:.1f}", "width": width, "height": height}
    except Exception:
        return {"duration": "N/A", "width": 0, "height": 0}


QWEN_MODEL_PATH = os.path.expanduser("/Users/junhui/models/Qwopus3.5-27B-v3-Q4_K_M.gguf")

T2V_SYSTEM_PROMPT = """You are a Creative Assistant writing concise, action-focused text-to-video prompts. Given a user Raw Input Prompt, generate a prompt to guide video generation.

#### Guidelines:
- Subject & Action: Identify main subject and primary action.
- Setting: Specify environment and atmosphere.
- Camera: Describe any camera movement if requested by user.
- Audio: Describe complete soundscape (ambient sounds, music, etc.).
- For ANY speech-related input (talking, conversation, singing, etc.), ALWAYS include exact words in quotes with voice characteristics.
- Specify language if not English and accent if relevant.
- Style: Include visual style at the beginning: "Style: <style>, <rest of prompt>." Default to cinematic-realistic if unspecified. Omit if unclear.
- Visual and audio only: NO non-visual/auditory senses (smell, taste, touch).
- Restrained language: Avoid dramatic/exaggerated terms. Use mild, natural phrasing.
    - Colors: Use plain terms ("red dress"), not intensified ("vibrant blue," "bright red").
    - Lighting: Use neutral descriptions ("soft overhead light"), not harsh ("blinding light").

#### Important notes:
- Camera motion: DO NOT invent camera motion unless requested by the user.
- Speech: DO NOT modify user-provided character dialogue unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Format: DO NOT use phrases like "The scene opens with...". Start directly with Style (optional) and chronological scene description.
- Format: DO NOT start your response with special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- If the user's raw input prompt is highly detailed, chronological and in the requested format: DO NOT make major edits or introduce new elements. Add/enhance audio descriptions if missing.

#### Output Format (Strict):
- Single continuous paragraph in natural language (English).
- NO titles, headings, prefaces, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications."""

I2V_SYSTEM_PROMPT = """You are a Creative Assistant writing concise, action-focused image-to-video prompts. Given an image (first frame) and user Raw Input Prompt, generate a prompt to guide video generation from that image.

#### Guidelines:
- Analyze the Image: Identify Subject, Setting, Elements, Style and Mood.
- Follow user Raw Input Prompt: Include all requested motion, actions, camera movements, audio, and details.
- Describe only changes from the image: Don't reiterate established visual details.
- Active language: Use present-progressive verbs ("is walking," "speaking").
- Chronological flow: Use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape throughout the prompt alongside actions.
- Speech (only when requested): Provide exact words in quotes with character's visual/voice characteristics.
- Style: Include visual style at beginning: "Style: <style>, <rest of prompt>." If unclear, omit.
- Visual and audio only: NO smell, taste, or tactile sensations.
- Restrained language: Avoid dramatic terms. Use mild, natural phrasing.

#### Important notes:
- Camera motion: DO NOT invent camera motion unless requested by the user.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Format: DO NOT use phrases like "The scene opens with...". Start directly with Style and description.
- Format: Never start your output with punctuation marks or special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.

#### Output Format (Strict):
- Single concise paragraph in natural English. NO titles, headings, prefaces, sections, code fences, or Markdown."""

JOINT_I2V_SYSTEM_PROMPT = """You are a Creative Assistant generating image-to-video prompts.

Given scene keywords and motion description, create an I2V prompt that animates the existing image.

#### Guidelines:
- The video should be a CONTINUOUS MOTION from the static image — NOT a scene change or transition
- Every element in the image should come ALIVE with subtle or dynamic motion
- If describing camera motion, treat it as a SINGLE continuous shot
- The scene setting and subject from the image should REMAIN CONSISTENT throughout
- Expand the motion description into specific, filmable actions that make the image feel alive
- Ambient elements should have continuous subtle motion (wind, water, light shifts, etc.)

#### Important:
- NO cuts or scene transitions — single continuous shot
- The video should feel like the still image "woke up" and is happening now
- Camera motion should feel natural, not like a camera change

#### Output Format (Strict):
Single continuous paragraph describing the animated scene.
Start directly with the motion or action.
NO titles, headings, or Markdown formatting."""

FLASH_SCENE_PLANNER_PROMPT = """OUTPUT ONLY. NO EXPLANATION. NO COMMENTARY. ONLY the {count} scenes separated by "---".

Example output (2 scenes):
Neon rain-soaked street, cyberpunk woman walks alone, glowing implants, reflections on pavement
Flying cars through holographic skyscrapers, aerial drone view

Your themes:
{themes}

Output (exactly {count} scenes, "---" between each, nothing else):"""

_llm = None


def _get_llm():
    """Lazy-load Qwen3.5 GGUF model."""
    global _llm
    if _llm is None:
        from llama_cpp import Llama

        _llm = Llama(
            model_path=QWEN_MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )
    return _llm


def _extract_enhanced_prompt(text: str) -> str:
    """Extract enhanced prompt from LLM output, handling think blocks."""
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    style_positions = [m.start() for m in re.finditer(r"Style:", text)]

    if style_positions:
        for pos in reversed(style_positions):
            paragraph = text[pos:]
            end_idx = paragraph.find("\n\n")
            if end_idx != -1:
                paragraph = paragraph[:end_idx]
            for marker in [
                "\nThis captures",
                "\nKey elements",
                "\nI need to",
                "\nLet me",
                "\nI should",
                "\n---",
            ]:
                idx = paragraph.find(marker)
                if idx != -1:
                    paragraph = paragraph[:idx]
            paragraph = paragraph.strip()
            if len(paragraph) > 100:
                return paragraph

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    if paragraphs:
        return max(paragraphs, key=len)

    return text


def enhance_prompt(raw_prompt: str, mode: str = "t2v", seed: int = 10) -> str:
    """Enhance a prompt using Qwen3.5 GGUF model via llama-cpp-python."""
    import gc

    system_prompt = T2V_SYSTEM_PROMPT if mode == "t2v" else I2V_SYSTEM_PROMPT
    user_content = f"User Raw Input Prompt: {raw_prompt}" if mode == "i2v" else f"user prompt: {raw_prompt}"

    llm = _get_llm()
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=512,
        temperature=0.7,
        seed=seed,
    )
    raw_output = result["choices"][0]["message"]["content"].strip()
    enhanced = _extract_enhanced_prompt(raw_output)
    gc.collect()
    return enhanced


EXTRACT_SCENE_MOTION_PROMPT = """Extract and return ONLY these two lines. Nothing else.

Example output:
SCENE_KEYWORDS: temple, cherry blossoms, morning mist, golden light
MOTION: gentle camera pan left, petals drifting, mist slowly rising

Your output (two lines only):"""


def extract_scene_and_motion(flux_prompt: str, seed: int = 10) -> tuple[str, str]:
    """Extract scene keywords and motion description from FLUX prompt."""
    llm = _get_llm()

    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": EXTRACT_SCENE_MOTION_PROMPT},
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


def generate_i2v_prompt(scene_keywords: str, motion_description: str, seed: int = 10) -> str:
    """Generate I2V prompt from scene keywords and motion description."""
    llm = _get_llm()

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

    raw_output = result["choices"][0]["message"]["content"].strip()

    return raw_output


def plan_flash_scenes(theme_words: str, scene_count: int, seed: int = 10) -> list[str]:
    """将主题词列表扩展为场景描述列表"""
    themes = [t.strip() for t in theme_words.strip().split("\n") if t.strip()]

    if len(themes) >= scene_count:
        scenes = themes[:scene_count]
    else:
        scenes = themes.copy()
        while len(scenes) < scene_count:
            for t in themes:
                if len(scenes) >= scene_count:
                    break
                scenes.append(t + " extended")

    return scenes


FLASH_ENHANCE_PROMPT = """.-enhance: {scene}

Output:"""


def enhance_flash_prompt(scene_description: str, seed: int = 10) -> str:
    """扩展场景描述，添加具体动态 - 规则匹配版本"""
    enhanced = scene_description

    # 更大胆的动态描述
    motion_map = {
        "neon rain": "heavy neon rain pouring down streets, raindrops splashing on ground",
        "flying cars": "flying cars zooming fast through neon-lit alleys, leaving light trails",
        "dust storm": "massive dust storm swirling violently around broken machines",
        "neon lights": "neon lights flickering intensely, casting moving shadows",
        "holographic": "holographic displays flickering rapidly, projecting moving figures",
        "crowds": "busy crowds rushing past, moving in chaotic directions",
        "rusty machines": "rusty machines grinding loudly, pistons pumping, sparks flying",
        "cyberpunk city": "cyberpunk city at night, rain pouring down walls",
        "multiple screens": "dozens of screens flickering with code, scrolling rapidly",
        "code": "code streaming down screens, glitching and flashing",
        "executive boardroom": "executives in suits pacing anxiously, city view changing",
        "dystopian view": "dystopian city sprawling below, smog moving across skyline",
        "hacker lab": "underground hacker lab, screens flashing, cables sparking",
        "megacorp tower": "massive megacorp tower, elevator moving fast inside",
        "market": "bustling cyberpunk market, vendors shouting, steam rising",
        "wasteland": "barren wasteland stretching to horizon, wind blowing debris",
        "underground": "dark underground area, flickering lights, water dripping",
        "alley": "narrow alley, neon signs buzzing, steam venting from grates",
    }

    for keyword, motion in motion_map.items():
        if keyword.lower() in enhanced.lower():
            enhanced = enhanced.replace(keyword, motion)

    # 添加全局动态
    if "camera" not in enhanced.lower():
        enhanced = enhanced + ", dynamic tracking camera movement"

    return enhanced


def generate_flash_segment(
    scene_description: str,
    output_image_path: str,
    output_video_path: str,
    width: int = 1024,
    height: int = 1024,
    frames: int = 49,
    seed: int = 42,
) -> tuple[bool, str]:
    """生成单个快闪片段: FLUX图像 + LTX I2V视频"""
    flux_process = run_flux_generation(
        prompt=scene_description,
        output_path=output_image_path,
        width=width,
        height=height,
        steps=4,
        cfg_scale=1.0,
    )

    for line in flux_process.stdout:
        pass

    flux_return = flux_process.wait()

    if flux_return != 0 or not os.path.exists(output_image_path):
        return False, "FLUX 图像生成失败"

    # 扩展动态描述
    i2v_prompt = enhance_flash_prompt(scene_description, seed=seed)

    python_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python3")
    ltx_height = (height // 32) * 32
    ltx_height = max(256, min(544, ltx_height))
    ltx_width = (width // 32) * 32
    ltx_width = max(256, min(832, ltx_width))

    cmd = [
        python_path,
        "-m",
        "ltx_pipelines_mlx",
        "generate",
        "--model",
        "models/ltx-2.3-mlx-q8",
        "--prompt",
        i2v_prompt,
        "--output",
        output_video_path,
        "--height",
        str(ltx_height),
        "--width",
        str(ltx_width),
        "--frames",
        str(frames),
        "--seed",
        str(seed),
        "--steps",
        "8",
        "--image",
        output_image_path,
        "--quiet",
    ]

    ltx_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/Users/junhui/work/ltx-2-mlx"
    )

    for line in ltx_process.stdout:
        pass

    ltx_return = ltx_process.wait()

    if ltx_return != 0 or not os.path.exists(output_video_path):
        return False, "LTX 视频生成失败"

    return True, ""


def concatenate_videos(video_paths: list, output_path: str) -> bool:
    """使用 ffmpeg concat 拼接多个视频 (硬切)"""
    filelist_path = output_path + ".filelist.txt"

    with open(filelist_path, "w") as f:
        for video_path in video_paths:
            f.write(f"file '{video_path}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", filelist_path, "-c", "copy", output_path]

    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        os.remove(filelist_path)
    except:
        pass

    return result.returncode == 0 and os.path.exists(output_path)


def run_generation(
    prompt,
    height,
    width,
    frames,
    seed,
    output_path,
    pipeline_type="one-stage",
    cfg_scale=3.0,
    stage1_steps=30,
    stage2_steps=3,
):
    """Run the video generation command"""
    python_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python3")
    cmd = [python_path, "-m", "ltx_pipelines_mlx", "generate"]

    cmd.extend(["--model", MODEL_DIR])
    cmd.extend(["--prompt", prompt])
    cmd.extend(["--output", output_path])
    cmd.extend(["--height", str(height)])
    cmd.extend(["--width", str(width)])
    cmd.extend(["--frames", str(frames)])
    cmd.extend(["--seed", str(seed)])

    if pipeline_type == "two-stage":
        cmd.append("--two-stage")
        cmd.extend(["--stage1-steps", str(stage1_steps)])
        cmd.extend(["--stage2-steps", str(stage2_steps)])
        cmd.extend(["--cfg-scale", str(cfg_scale)])
    elif pipeline_type == "two-stage-hq":
        cmd.append("--hq")
        cmd.extend(["--stage1-steps", str(stage1_steps)])
        cmd.extend(["--stage2-steps", str(stage2_steps)])
        cmd.extend(["--cfg-scale", str(cfg_scale)])

    cmd.append("--quiet")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/Users/junhui/work/ltx-2-mlx"
    )

    return process


def run_flux_generation(
    prompt,
    output_path,
    width=1024,
    height=1024,
    steps=4,
    cfg_scale=1.0,
):
    """Run FLUX image generation using stable-diffusion.cpp"""
    python_path = os.path.join(FLUX_SCRIPT_DIR, "generate.py")
    cmd = [
        "python3",
        python_path,
        prompt,
        "-o",
        output_path,
        "-w",
        str(width),
        "--height",
        str(height),
        "-s",
        str(steps),
        "--cfg-scale",
        str(cfg_scale),
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=FLUX_SCRIPT_DIR)
    return process


def render_video_tab():
    """Render the LTX-2 video generation tab"""

    with st.sidebar:
        st.header("⚙️ 视频参数设置")

        st.subheader("流水线类型")
        pipeline_type = st.selectbox(
            "Pipeline",
            ["one-stage", "two-stage", "two-stage-hq"],
            format_func=lambda x: {
                "one-stage": "🚀 单阶段（快速，8步）",
                "two-stage": "🎬 两阶段（高质量，CFG+上采样）",
                "two-stage-hq": "✨ 两阶段 HQ（res_2s 采样器，最高质量）",
            }[x],
        )

        st.subheader("视频参数")
        height = st.number_input("Height", min_value=256, max_value=544, value=480, step=32)
        width = st.number_input("Width", min_value=384, max_value=832, value=704, step=32)
        frames = st.number_input("Frames (必须 8k+1)", min_value=9, max_value=193, value=97, step=8)
        seed = st.number_input("Seed (-1 随机)", value=42)

        if pipeline_type == "one-stage":
            st.info(f"💡 帧数必须是 8k+1，如: 9, 25, 41, 49, 65, 81, 97, 121, 145, 161, 193")
            cfg_scale = 3.0
            stage1_steps = 30
            stage2_steps = 3
        else:
            cfg_scale = st.slider("CFG Scale", min_value=1.0, max_value=7.0, value=3.0, step=0.5)
            stage1_steps = st.number_input(
                "Stage 1 Steps", min_value=5, max_value=50, value=15 if pipeline_type == "two-stage-hq" else 30, step=5
            )
            stage2_steps = st.number_input("Stage 2 Steps", min_value=1, max_value=10, value=3, step=1)
            st.info("💡 两阶段：Stage 1 半分辨率 CFG → 上采样 → Stage 2 精炼")

    if "enhanced_prompt" not in st.session_state:
        st.session_state.enhanced_prompt = None

    prompt = st.text_area(
        "📝 提示词",
        height=120,
        placeholder="小猫做饭",
        value=st.session_state.enhanced_prompt
        or "A woman with long black hair walks through a traditional Japanese temple corridor, cherry blossoms falling, warm afternoon sunlight, cinematic 4K, smooth camera tracking shot",
    )

    enhance_mode = st.radio("增强模式", ["t2v", "i2v"], horizontal=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        enhance_btn = st.button("✨ 增强提示词", use_container_width=True)
    with col2:
        generate_btn = st.button("🚀 生成视频", use_container_width=True)

    if enhance_btn:
        if not prompt:
            st.error("请输入提示词")
        else:
            with st.spinner("✨ 正在增强提示词..."):
                enhanced = enhance_prompt(prompt, mode=enhance_mode, seed=seed if seed > 0 else 10)
                st.session_state.enhanced_prompt = enhanced
                st.success("提示词增强完成！")
                st.text_area("📝 增强后的提示词", value=enhanced, height=100, disabled=True)
            st.rerun()

    if generate_btn:
        if not prompt:
            st.error("请输入提示词")
        elif frames % 8 != 1:
            st.error(f"帧数必须是 8k+1，当前值 {frames} 不合法")
        else:
            timestamp = int(time.time())
            output_path = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")

            st.markdown("---")
            st.subheader("📊 生成状态")

            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            log_area = st.empty()
            log_lines = []

            status_placeholder.info("⏳ 正在初始化...")

            process = run_generation(
                prompt,
                height,
                width,
                frames,
                seed,
                output_path,
                pipeline_type=pipeline_type,
                cfg_scale=cfg_scale if pipeline_type != "one-stage" else 3.0,
                stage1_steps=stage1_steps if pipeline_type != "one-stage" else 30,
                stage2_steps=stage2_steps if pipeline_type != "one-stage" else 3,
            )

            start_time = time.time()
            if pipeline_type == "one-stage":
                total_steps = 8
            else:
                total_steps = (stage1_steps if pipeline_type != "one-stage" else 30) + (
                    stage2_steps if pipeline_type != "one-stage" else 3
                )

            current_step = 0
            current_stage = 1

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if line:
                    stripped = line.strip()
                    if stripped:
                        log_lines.append(stripped)
                        if len(log_lines) > 20:
                            log_lines = log_lines[-20:]
                        log_area.text("\n".join(log_lines))

                    if "Stage 2" in line:
                        current_stage = 2

                    if "Denoising:" in line and "guided" not in line:
                        match = line.split("|")
                        if len(match) > 1:
                            step_info = match[1].strip()
                            if "/" in step_info:
                                parts = step_info.split("/")
                                current_step = int(parts[0].strip().replace("%", ""))
                                stage1_done = stage1_steps if pipeline_type != "one-stage" else 30
                                step_num = stage1_done + current_step if current_stage == 2 else current_step
                                progress = step_num / total_steps
                                progress_bar.progress(min(progress, 1.0))
                                stage_label = "Stage 2 精炼" if current_stage == 2 else "Stage 1 生成"
                                status_placeholder.info(f"🔄 {stage_label}... {current_step}/{total_steps} 步")

                    elif "Saved to:" in line:
                        progress_bar.progress(1.0)

            return_code = process.wait()
            elapsed = time.time() - start_time

            if return_code == 0 and os.path.exists(output_path):
                status_placeholder.success(f"✅ 生成完成! 耗时: {elapsed:.1f} 秒")

                st.markdown("---")
                st.subheader("🎥 视频预览")

                st.video(output_path)

                info = get_video_info(output_path)
                file_size = os.path.getsize(output_path) / (1024 * 1024)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("文件大小", f"{file_size:.1f} MB")
                col2.metric("时长", f"{info['duration']}s")
                col3.metric("分辨率", f"{info['width']}x{info['height']}")
                col4.metric("耗时", f"{elapsed:.1f}s")

                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 下载视频", data=f, file_name=f"ltx2_video_{timestamp}.mp4", mime="video/mp4"
                    )

                folder_path = os.path.dirname(output_path)
                st.markdown(f"📁 输出目录: `{folder_path}`")

            else:
                status_placeholder.error("❌ 生成失败，请检查参数或内存")


def render_flash_tab():
    """快闪视频生成 Tab"""
    st.markdown("### ⚡ 快闪视频")
    st.caption("多个主题词 → AI 规划 → FLUX图像 → LTX视频 → ffmpeg拼接")

    state = st.session_state.flash_workflow

    theme_words = st.text_area(
        "主题词列表 (每行一个)",
        height=120,
        placeholder="neon city rain\nholographic ads flying cars\nunderground hacker lab\nmegacorp boardroom",
        value=state.get("theme_words", ""),
        key="flash_theme_words",
    )
    state["theme_words"] = theme_words

    col_count, col_frames, col_seed = st.columns(3)
    with col_count:
        scene_count = st.selectbox("片段数量", [3, 4, 5, 6, 7, 8, 9, 10], index=2, key="flash_scene_count")
    with col_frames:
        frames_per_segment = st.selectbox("每段帧数", [9, 25, 41, 49, 65, 81, 97], index=3, key="flash_frames")
    with col_seed:
        seed = st.number_input("Seed (-1=随机)", value=42, key="flash_seed")

    state["scene_count"] = scene_count
    state["frames_per_segment"] = frames_per_segment
    state["seed"] = seed

    generate_btn = st.button("⚡ 生成快闪视频", key="flash_generate", use_container_width=True)

    if generate_btn:
        if not theme_words.strip():
            st.error("请输入主题词")
        else:
            state["status"] = "planning"
            state["scenes"] = []
            state["segments"] = []
            state["current_segment"] = 0

            with st.spinner("✨ AI 正在规划场景..."):
                scenes = plan_flash_scenes(
                    theme_words=theme_words, scene_count=scene_count, seed=seed if seed > 0 else 42
                )
                state["scenes"] = scenes

            st.success(f"✅ 规划完成! 将生成 {len(scenes)} 个片段")

            import tempfile

            temp_dir = tempfile.mkdtemp(prefix="flash_")
            state["temp_dir"] = temp_dir
            state["status"] = "generating"

            segment_paths = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, scene in enumerate(scenes):
                state["current_segment"] = i
                status_text.text(f"正在生成片段 {i + 1}/{len(scenes)}: {scene[:50]}...")
                progress_bar.progress((i) / len(scenes))

                timestamp = int(time.time())
                image_path = os.path.join(temp_dir, f"segment_{i}_image.png")
                video_path = os.path.join(temp_dir, f"segment_{i}.mp4")

                success, error = generate_flash_segment(
                    scene_description=scene,
                    output_image_path=image_path,
                    output_video_path=video_path,
                    width=1024,
                    height=1024,
                    frames=frames_per_segment,
                    seed=seed + i if seed > 0 else -1,
                )

                if success:
                    segment_paths.append(video_path)
                else:
                    st.error(f"片段 {i + 1} 生成失败: {error}")
                    state["status"] = "error"
                    break

            progress_bar.progress(1.0)

            if state.get("status") != "error" and segment_paths:
                status_text.text("正在拼接视频...")
                state["status"] = "concatenating"

                timestamp = int(time.time())
                output_path = os.path.abspath(os.path.join(OUTPUT_DIR, f"flash_video_{timestamp}.mp4"))

                if concatenate_videos(segment_paths, output_path):
                    state["output_path"] = output_path
                    state["segments"] = segment_paths
                    state["status"] = "done"
                    st.success("✅ 快闪视频生成完成!")
                else:
                    st.error("视频拼接失败")
                    state["status"] = "error"

    if state.get("status") == "done" and state.get("output_path"):
        st.markdown("---")
        st.markdown("#### 🎬 视频预览")
        st.video(state["output_path"])

        file_size = os.path.getsize(state["output_path"]) / (1024 * 1024)
        col1, col2 = st.columns(2)
        col1.metric("文件大小", f"{file_size:.1f} MB")
        col2.metric("片段数量", str(len(state.get("scenes", []))))

        with open(state["output_path"], "rb") as f:
            st.download_button(
                label="📥 下载视频", data=f, file_name=f"flash_video_{int(time.time())}.mp4", mime="video/mp4"
            )

        if st.button("🔄 新建任务", key="flash_new_task"):
            if state.get("temp_dir") and os.path.exists(state["temp_dir"]):
                import shutil

                shutil.rmtree(state["temp_dir"], ignore_errors=True)

            st.session_state.flash_workflow = {
                "theme_words": "",
                "scene_count": 5,
                "frames_per_segment": 49,
                "seed": 42,
                "scenes": [],
                "segments": [],
                "status": "idle",
                "current_segment": 0,
                "output_path": None,
                "temp_dir": None,
            }
            st.rerun()


def render_joint_tab():
    """联合生成: FLUX 图像 + LTX-2 I2V 视频"""
    st.markdown("### ➕ 联合生成工作流")
    st.caption("FLUX 生成高质量图像 → LTX-2 I2V 基于图像生成视频")

    state = st.session_state.joint_workflow

    # FLUX 场景描述
    flux_prompt = st.text_area(
        "FLUX 场景描述",
        height=80,
        placeholder="A serene Japanese temple at dawn, cherry blossoms in full bloom...",
        value=state.get("flux_prompt", ""),
        key="joint_flux_prompt",
    )

    # 场景关键词和运动描述 (提取后显示为可编辑文本)
    extracted = state.get("extracted", False)

    # 显式设置 session_state 以确保 widget 显示正确的值
    if extracted and state.get("scene_keywords"):
        st.session_state["joint_scene_keywords"] = state["scene_keywords"]
    if extracted and state.get("motion_description"):
        st.session_state["joint_motion_description"] = state["motion_description"]

    col1, col2 = st.columns(2)
    with col1:
        if extracted and state.get("scene_keywords"):
            scene_keywords = st.text_area(
                "场景关键词 (已提取，可编辑)",
                height=60,
                key="joint_scene_keywords",
            )
        else:
            scene_keywords = st.text_area(
                "场景关键词 (用于AI)",
                placeholder="Japanese temple, cherry blossoms, morning mist, golden sunlight",
                value=state.get("scene_keywords", ""),
                height=60,
                key="joint_scene_keywords",
            )
    with col2:
        if extracted and state.get("motion_description"):
            motion_description = st.text_area(
                "运动描述 (已提取，可编辑)",
                height=60,
                key="joint_motion_description",
            )
        else:
            motion_description = st.text_area(
                "运动描述",
                placeholder="Camera pans slowly left, petals falling...",
                value=state.get("motion_description", ""),
                height=60,
                key="joint_motion_description",
            )

    # 保存输入到状态
    state["flux_prompt"] = flux_prompt
    state["scene_keywords"] = scene_keywords
    state["motion_description"] = motion_description

    # 高级选项 (FLUX 参数)
    with st.expander("⚙️ 高级选项 (FLUX)"):
        col_w, col_h, col_s, col_cfg = st.columns(4)
        with col_w:
            flux_width = st.number_input(
                "Width", min_value=256, max_value=2048, value=1024, step=64, key="joint_flux_width"
            )
        with col_h:
            flux_height = st.number_input(
                "Height", min_value=256, max_value=2048, value=1024, step=64, key="joint_flux_height"
            )
        with col_s:
            flux_steps = st.number_input("Steps", min_value=1, max_value=50, value=4, step=1, key="joint_flux_steps")
        with col_cfg:
            flux_cfg = st.slider("CFG", min_value=1.0, max_value=10.0, value=1.0, step=0.5, key="joint_flux_cfg")

    # 保存 FLUX 参数
    state["flux_params"] = {
        "width": flux_width,
        "height": flux_height,
        "steps": flux_steps,
        "cfg": flux_cfg,
    }

    flux_generate_btn = st.button("🖼️ 生成图像", key="joint_flux_generate", use_container_width=True)

    # FLUX 生成
    if flux_generate_btn:
        if not flux_prompt:
            st.error("请输入 FLUX 场景描述")
        else:
            timestamp = int(time.time())
            output_path = os.path.abspath(os.path.join(FLUX_OUTPUT_DIR, f"flux_joint_{timestamp}.png"))

            with st.spinner("🖼️ 正在生成图像..."):
                process = run_flux_generation(
                    prompt=flux_prompt,
                    output_path=output_path,
                    width=flux_width,
                    height=flux_height,
                    steps=flux_steps,
                    cfg_scale=flux_cfg,
                )

                for line in process.stdout:
                    pass

                return_code = process.wait()

            if return_code == 0 and os.path.exists(output_path):
                st.session_state.joint_workflow["flux_image"] = output_path
                st.session_state.joint_workflow["flux_prompt"] = flux_prompt
                st.success("✅ 图像生成完成! 正在分析提示词...")

                # 自动提取场景关键词和运动描述
                with st.spinner("✨ AI 正在分析场景和运动..."):
                    scene_keywords, motion_description = extract_scene_and_motion(flux_prompt, seed=42)
                    st.session_state.joint_workflow["scene_keywords"] = scene_keywords
                    st.session_state.joint_workflow["motion_description"] = motion_description
                    st.session_state.joint_workflow["extracted"] = True

                st.rerun()
            else:
                st.error("❌ 图像生成失败")

    # 图像预览
    if state["flux_image"] and os.path.exists(state["flux_image"]):
        st.markdown("---")
        st.markdown("#### 🖼️ 图像预览")
        st.image(state["flux_image"], width=512)

        col_regen, col_confirm = st.columns(2)
        with col_regen:
            if st.button("🔄 重新生成图像", key="regen_image_btn"):
                if os.path.exists(state["flux_image"]):
                    os.remove(state["flux_image"])
                state["flux_image"] = None
                st.rerun()
        with col_confirm:
            if st.button("✅ 确认并生成视频", key="confirm_video_btn"):
                state["step"] = "generate_video"
                st.rerun()

    # 视频生成阶段
    if state.get("step") == "generate_video":
        st.markdown("---")
        st.markdown("#### 🎬 视频生成")

        # 从 session_state 读取最新的文本框值
        current_scene_keywords = st.session_state.get("joint_scene_keywords", state.get("scene_keywords", ""))
        current_motion_description = st.session_state.get(
            "joint_motion_description", state.get("motion_description", "")
        )

        # AI 生成 I2V 提示词
        if not state.get("i2v_prompt"):
            with st.spinner("✨ AI 正在生成 I2V 提示词..."):
                i2v_prompt = generate_i2v_prompt(
                    scene_keywords=current_scene_keywords,
                    motion_description=current_motion_description,
                    seed=42,
                )
                state["i2v_prompt"] = i2v_prompt

        # 显示生成的提示词
        st.markdown("**I2V 提示词 (AI生成)**")
        st.text_area("提示词预览", value=state["i2v_prompt"] or "", height=100, disabled=True, key="i2v_prompt_display")

        if st.button("🔄 重新生成提示词", key="regen_prompt_btn"):
            state["i2v_prompt"] = None
            st.rerun()

        # LTX-2 参数设置
        flux_w = state["flux_params"].get("width", 1024)
        flux_h = state["flux_params"].get("height", 1024)

        ltx_width = (flux_w // 32) * 32
        ltx_height = (flux_h // 32) * 32
        ltx_width = max(256, min(832, ltx_width))
        ltx_height = max(256, min(544, ltx_height))

        col_frame, col_step, col_seed = st.columns(3)
        with col_frame:
            frames = st.selectbox(
                "Frames (8k+1)", [9, 25, 41, 49, 65, 81, 97, 121, 145, 161, 193], index=6, key="joint_frames"
            )
        with col_step:
            steps = st.number_input("Steps", min_value=1, max_value=50, value=8, step=1, key="joint_steps")
        with col_seed:
            seed = st.number_input("Seed (-1=随机)", value=42, key="joint_seed")

        st.info(f"💡 分辨率自动继承 FLUX: {ltx_width}x{ltx_height}")

        if st.button("🎬 生成视频", key="joint_generate_video", use_container_width=True):
            timestamp = int(time.time())
            video_path = os.path.abspath(os.path.join(OUTPUT_DIR, f"joint_video_{timestamp}.mp4"))

            state["video_path"] = video_path

            with st.spinner("🎬 正在生成视频..."):
                python_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python3")
                cmd = [
                    python_path,
                    "-m",
                    "ltx_pipelines_mlx",
                    "generate",
                    "--model",
                    "models/ltx-2.3-mlx-q8",
                    "--prompt",
                    state["i2v_prompt"],
                    "--output",
                    video_path,
                    "--height",
                    str(ltx_height),
                    "--width",
                    str(ltx_width),
                    "--frames",
                    str(frames),
                    "--seed",
                    str(seed),
                    "--steps",
                    str(steps),
                    "--image",
                    state["flux_image"],
                    "--quiet",
                ]

                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/Users/junhui/work/ltx-2-mlx"
                )

                for line in process.stdout:
                    pass

                return_code = process.wait()

            if return_code == 0 and os.path.exists(video_path):
                st.success("✅ 视频生成完成!")
                st.rerun()
            else:
                st.error("❌ 视频生成失败")

    # 视频预览
    if state.get("video_path") and os.path.exists(state["video_path"]):
        st.markdown("---")
        st.markdown("#### 🎬 视频预览")
        st.video(state["video_path"])

        file_size = os.path.getsize(state["video_path"]) / (1024 * 1024)
        col1, col2 = st.columns(2)
        col1.metric("文件大小", f"{file_size:.1f} MB")
        col2.metric("耗时", "N/A")

        with open(state["video_path"], "rb") as f:
            st.download_button(
                label="📥 下载视频", data=f, file_name=f"joint_video_{int(time.time())}.mp4", mime="video/mp4"
            )

        if st.button("🔄 新建任务", key="joint_new_task"):
            st.session_state.joint_workflow = {
                "flux_image": None,
                "flux_params": {},
                "flux_prompt": "",
                "scene_keywords": "",
                "motion_description": "",
                "i2v_prompt": None,
                "video_path": None,
                "step": None,
            }
            st.rerun()


def render_flux_tab():
    """Render the FLUX image generation tab"""

    st.markdown("### 🖼️ FLUX.2-klein-9B 图像生成")

    if not os.path.exists(os.path.join(FLUX_MODEL_DIR, "flux-2-klein-9b-Q8_0.gguf")):
        st.warning("⚠️ FLUX 模型未找到，请先下载:")
        st.code("flux-2-klein-9b-Q8_0.gguf -> https://huggingface.co/leejet/FLUX.2-klein-9B-GGUF")
        st.code("flux2_ae.safetensors -> https://huggingface.co/black-forest-labs/FLUX.2-dev")
        st.code("Qwen3-8B-Q4_1.gguf -> https://huggingface.co/unsloth/Qwen3-8B-GGUF")
        return

    prompt = st.text_area(
        "📝 提示词",
        height=100,
        placeholder="a beautiful landscape at sunset",
        value="A serene Japanese temple at dawn, cherry blossoms in full bloom, morning mist rising from the surrounding forest, soft golden light filtering through the branches, traditional wooden architecture, photorealistic, 8K detailed",
    )

    col_width, col_height, col_steps = st.columns(3)
    with col_width:
        width = st.number_input("宽度", min_value=256, max_value=2048, value=1024, step=64)
    with col_height:
        height = st.number_input("高度", min_value=256, max_value=2048, value=1024, step=64)
    with col_steps:
        steps = st.number_input("步数", min_value=1, max_value=50, value=4, step=1)

    cfg_scale = st.slider("CFG Scale", min_value=1.0, max_value=10.0, value=1.0, step=0.5)

    generate_btn = st.button("🖼️ 生成图像", use_container_width=True)

    if generate_btn:
        if not prompt:
            st.error("请输入提示词")
        else:
            timestamp = int(time.time())
            output_path = os.path.abspath(os.path.join(FLUX_OUTPUT_DIR, f"flux_{timestamp}.png"))

            st.markdown("---")
            st.subheader("📊 生成状态")

            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            log_area = st.empty()
            log_lines = []

            status_placeholder.info("⏳ 正在初始化...")

            process = run_flux_generation(
                prompt=prompt,
                output_path=output_path,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
            )

            start_time = time.time()

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if line:
                    stripped = line.strip()
                    if stripped:
                        log_lines.append(stripped)
                        if len(log_lines) > 20:
                            log_lines = log_lines[-20:]
                        log_area.text("\n".join(log_lines))

                    if "%" in line:
                        import re

                        matches = re.findall(r"(\d+)%", line)
                        if matches:
                            progress_bar.progress(int(matches[-1]) / 100)

                if os.path.exists(output_path):
                    progress_bar.progress(1.0)

            return_code = process.wait()
            elapsed = time.time() - start_time

            # Check in both locations (relative to cwd and absolute)
            flux_output = os.path.join(FLUX_SCRIPT_DIR, output_path) if not os.path.isabs(output_path) else output_path
            if not os.path.exists(flux_output):
                flux_output = output_path

            if return_code == 0 and os.path.exists(flux_output):
                status_placeholder.success(f"✅ 生成完成! 耗时: {elapsed:.1f} 秒")

                st.markdown("---")
                st.subheader("🖼️ 图像预览")

                st.image(flux_output)

                file_size = os.path.getsize(flux_output) / (1024 * 1024)
                img_info = f"{width}x{height}"

                col1, col2, col3 = st.columns(3)
                col1.metric("文件大小", f"{file_size:.1f} MB")
                col2.metric("分辨率", img_info)
                col3.metric("耗时", f"{elapsed:.1f}s")

                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 下载图像", data=f, file_name=f"flux_image_{timestamp}.png", mime="image/png"
                    )

                folder_path = os.path.dirname(output_path)
                st.markdown(f"📁 输出目录: `{folder_path}`")

            else:
                status_placeholder.error("❌ 生成失败，请检查模型文件是否完整")


def main():
    st.title("🎬 LTX-2 & 🖼️ FLUX Generator")
    st.markdown("Apple Silicon 视频/图像生成工具")

    tab1, tab2, tab3, tab4 = st.tabs(["🎬 视频生成 (LTX-2)", "🖼️ 图像生成 (FLUX)", "➕ 联合生成", "⚡ 快闪"])

    with tab1:
        render_video_tab()

    with tab2:
        render_flux_tab()

    with tab3:
        render_joint_tab()

    with tab4:
        render_flash_tab()


if __name__ == "__main__":
    main()
