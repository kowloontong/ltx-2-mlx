"""Refactored Web UI for LTX-2 video generation.

Features:
- Modular architecture with separate config, state, and generators
- Default to two-stage HQ for all workflows
- Support for joint workflow (Flux + LTX I2V)
- Support for flash workflow (multi-segment, I2V or T2V mode)
- Support for I2V workflow (image upload + LTX I2V)
- Mode-specific parameter panels
"""

import streamlit as st
from pathlib import Path

# Load configuration
from webui.config import WebUIConfig
from webui.state import StateManager
from webui.generators.ltx import LTXGenerator
from webui.generators.flux import FluxGenerator
from webui.prompts import PromptEnhancer

# Initialize config
config = WebUIConfig.load()

# Initialize generators
ltx_gen = LTXGenerator(config.ltx, config.pipeline)
flux_gen = FluxGenerator(config.flux)
prompt_enhancer = PromptEnhancer(config.qwen_model_path)

# Page config
st.set_page_config(
    page_title="LTX-2 & FLUX Generator (Refactored)",
    page_icon="🎬",
    layout="wide",
)

# Custom styles
st.markdown(
    """
<style>
    .main { background-color: #f8f9fa; }
    .stButton > button {
        background-color: #4A90D9;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #357ABD; }
</style>
""",
    unsafe_allow_html=True,
)

# Tab navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 T2V 生成",
    "🎨 联合工作流",
    "⚡ 快闪工作流",
    "🖼️ I2V 生成",
])

# Tab 1: T2V Generation
with tab1:
    st.header("🎬 文本生成视频 (T2V)")
    
    # Two-column layout: parameters on left, content on right
    col_params, col_content = st.columns([1, 2])
    
    with col_params:
        st.subheader("⚙️ 参数设置")
        use_hq = st.checkbox("使用 HQ 模式（推荐）", value=True, key="t2v_hq")
        use_two_stage = st.checkbox("使用两阶段", value=True, disabled=use_hq, key="t2v_two_stage")
        
        st.markdown("**视频尺寸**")
        height = st.number_input("Height", 256, 544, 480, 32, key="t2v_height")
        width = st.number_input("Width", 384, 832, 704, 32, key="t2v_width")
        frames = st.number_input("Frames", 9, 193, 97, 8, key="t2v_frames")
        seed = st.number_input("Seed", value=42, key="t2v_seed")
        
        if use_hq:
            pipeline_type = "two-stage-hq"
            st.markdown("**HQ 参数**")
            cfg_scale = st.slider("CFG Scale", 1.0, 7.0, 3.5, 0.5, key="t2v_cfg")
            stage1_steps = st.number_input("Stage 1 Steps", 5, 50, 20, 5, key="t2v_s1")
            stage2_steps = st.number_input("Stage 2 Steps", 1, 10, 5, 1, key="t2v_s2")
        elif use_two_stage:
            pipeline_type = "two-stage"
            st.markdown("**两阶段参数**")
            cfg_scale = st.slider("CFG Scale", 1.0, 7.0, 3.5, 0.5, key="t2v_cfg")
            stage1_steps = st.number_input("Stage 1 Steps", 5, 50, 30, 5, key="t2v_s1")
            stage2_steps = st.number_input("Stage 2 Steps", 1, 10, 5, 1, key="t2v_s2")
        else:
            pipeline_type = "one-stage"
            cfg_scale = 3.0
            stage1_steps = 30
            stage2_steps = 3
    
    with col_content:
        prompt = st.text_area("Prompt", height=100, key="t2v_prompt", placeholder="输入视频描述...")
        
        if st.button("🚀 生成视频", type="primary", key="t2v_gen"):
            if not prompt:
                st.error("请输入 Prompt")
            else:
                output_path = config.ltx.output_dir / "t2v" / "output.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with st.spinner("生成中..."):
                    success, error = ltx_gen.generate_t2v(
                        prompt=prompt,
                        output_path=str(output_path),
                        height=height,
                        width=width,
                        frames=frames,
                        seed=seed,
                        pipeline_type=pipeline_type,
                        cfg_scale=cfg_scale,
                        stage1_steps=stage1_steps,
                        stage2_steps=stage2_steps,
                    )
                
                if success:
                    st.success("✅ 生成成功！")
                    st.video(str(output_path))
                else:
                    st.error(f"❌ 生成失败: {error}")

# Tab 2: Joint Workflow
with tab2:
    st.header("🎨 联合工作流：Flux + LTX I2V")
    st.markdown("生成高质量 Flux 图像，然后用 LTX I2V 动画化（默认两阶段 HQ）")
    
    # Two-column layout
    col_params, col_content = st.columns([1, 2])
    
    with col_params:
        st.subheader("⚙️ 参数设置")
        use_hq = st.checkbox("使用 HQ 模式（推荐）", value=True, key="joint_hq")
        use_two_stage = st.checkbox("使用两阶段", value=True, disabled=use_hq, key="joint_two_stage")
        seed = st.number_input("Seed", value=42, key="joint_seed")
        
        if use_hq:
            pipeline_type = "two-stage-hq"
        elif use_two_stage:
            pipeline_type = "two-stage"
        else:
            pipeline_type = "one-stage"
    
    with col_content:
        flux_prompt = st.text_area("Flux Prompt", height=100, key="joint_flux_prompt", placeholder="输入图像描述...")
        
        if st.button("🚀 生成联合视频", type="primary", key="joint_gen"):
            if not flux_prompt:
                st.error("请输入 Flux Prompt")
            else:
                output_dir = config.ltx.output_dir / "joint"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Step 1: Generate Flux image
                with st.spinner("生成 Flux 图像..."):
                    flux_output = output_dir / "flux_image.png"
                    success, error = flux_gen.generate(
                        prompt=flux_prompt,
                        output_path=str(flux_output),
                    )
                
                if not success:
                    st.error(f"Flux 生成失败: {error}")
                else:
                    st.image(str(flux_output), caption="Flux 图像", use_column_width=True)
                    
                    # Step 2: Extract scene and motion
                    with st.spinner("提取场景和运动..."):
                        scene_keywords, motion_description = prompt_enhancer.extract_scene_and_motion(
                            flux_prompt, seed=seed
                        )
                    
                    # Step 3: Generate I2V prompt
                    with st.spinner("生成 I2V Prompt..."):
                        i2v_prompt = prompt_enhancer.generate_i2v_prompt(
                            scene_keywords, motion_description, seed=seed
                        )
                    
                    st.markdown(f"**I2V Prompt:**\n```\n{i2v_prompt}\n```")
                    
                    # Step 4: Generate I2V video
                    with st.spinner("生成 I2V 视频..."):
                        video_output = output_dir / "joint_output.mp4"
                        success, error = ltx_gen.generate_i2v(
                            prompt=i2v_prompt,
                            image_path=str(flux_output),
                            output_path=str(video_output),
                            pipeline_type=pipeline_type,
                            seed=seed,
                        )
                    
                    if success:
                        st.success("✅ 生成成功！")
                        st.video(str(video_output))
                    else:
                        st.error(f"I2V 生成失败: {error}")

# Tab 3: Flash Workflow
with tab3:
    st.header("⚡ 快闪工作流")
    st.markdown("快速生成多段视频并拼接（支持 I2V 和 T2V 两种模式，默认两阶段 HQ）")
    
    # Two-column layout
    col_params, col_content = st.columns([1, 2])
    
    with col_params:
        st.subheader("⚙️ 参数设置")
        mode = st.radio(
            "生成模式",
            ["i2v", "t2v"],
            format_func=lambda x: {
                "i2v": "🎨 I2V 模式（Flux + LTX I2V）",
                "t2v": "🎬 T2V 模式（纯 LTX T2V）",
            }[x],
            key="flash_mode",
        )
        
        use_hq = st.checkbox("使用 HQ 模式（推荐）", value=True, key="flash_hq")
        use_two_stage = st.checkbox("使用两阶段", value=True, disabled=use_hq, key="flash_two_stage")
        
        st.markdown("**场景设置**")
        scene_count = st.number_input("场景数量", 1, 10, 5, key="flash_scenes")
        frames_per_segment = st.number_input("每段帧数", 9, 97, 49, 8, key="flash_frames")
        seed = st.number_input("Seed", value=42, key="flash_seed")
        
        if use_hq:
            pipeline_type = "two-stage-hq"
            st.markdown("**HQ 参数**")
            cfg_scale = st.slider("CFG Scale", 1.0, 7.0, 3.5, 0.5, key="flash_cfg")
            stage1_steps = st.number_input("Stage 1 Steps", 5, 50, 20, 5, key="flash_s1")
            stage2_steps = st.number_input("Stage 2 Steps", 1, 10, 5, 1, key="flash_s2")
        elif use_two_stage:
            pipeline_type = "two-stage"
            st.markdown("**两阶段参数**")
            cfg_scale = st.slider("CFG Scale", 1.0, 7.0, 3.5, 0.5, key="flash_cfg")
            stage1_steps = st.number_input("Stage 1 Steps", 5, 50, 30, 5, key="flash_s1")
            stage2_steps = st.number_input("Stage 2 Steps", 1, 10, 5, 1, key="flash_s2")
        else:
            pipeline_type = "one-stage"
            cfg_scale = 3.0
            stage1_steps = 30
            stage2_steps = 3
    
    with col_content:
        theme_words = st.text_area(
            "主题词（每行一个）",
            height=150,
            key="flash_themes",
            placeholder="输入场景描述，每行一个...\n例如：\nCorrupted data bazaars\nSubterranean biomech vault\nCollapsed megacorp skyline"
        )
        
        if st.button("🚀 生成快闪视频", type="primary", key="flash_gen"):
            if not theme_words:
                st.error("请输入主题词")
            else:
                # Parse theme words into scenes
                scenes = [s.strip() for s in theme_words.strip().split("\n") if s.strip()]
                
                # Extend scenes if needed
                while len(scenes) < scene_count:
                    for s in scenes.copy():
                        if len(scenes) >= scene_count:
                            break
                        scenes.append(s + " extended")
                scenes = scenes[:scene_count]
                
                # Create output directory
                output_dir = config.ltx.output_dir / "flash"
                output_dir.mkdir(parents=True, exist_ok=True)
                temp_dir = output_dir / "temp"
                temp_dir.mkdir(exist_ok=True)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                segment_paths = []
                
                # Generate each segment
                for i, scene in enumerate(scenes):
                    status_text.text(f"生成段 {i+1}/{scene_count}: {scene[:50]}...")
                    progress_bar.progress((i + 0.5) / scene_count)
                    
                    segment_output = temp_dir / f"segment_{i:03d}.mp4"
                    
                    if mode == "i2v":
                        # I2V mode: Flux image + LTX I2V
                        # Step 1: Generate Flux image with enhanced prompt
                        image_path = temp_dir / f"image_{i:03d}.png"
                        
                        # Ensure parent directory exists
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Enhance Flux prompt for better image quality
                        flux_prompt = prompt_enhancer.enhance_flux_prompt(scene, seed=seed + i)
                        
                        success, error = flux_gen.generate(
                            prompt=flux_prompt,
                            output_path=str(image_path),
                        )
                        
                        if not success:
                            st.error(f"段 {i+1} Flux 生成失败: {error}")
                            st.error(f"输出路径: {image_path}")
                            st.error(f"路径存在: {image_path.parent.exists()}")
                            break
                        
                        # Step 2: Enhance prompt (I2V mode - emphasize motion)
                        enhanced_prompt = prompt_enhancer.enhance_flash_prompt(scene, seed=seed + i, mode="i2v")
                        
                        # Step 3: LTX I2V
                        success, error = ltx_gen.generate_i2v(
                            prompt=enhanced_prompt,
                            image_path=str(image_path),
                            output_path=str(segment_output),
                            frames=frames_per_segment,
                            seed=seed + i,
                            pipeline_type=pipeline_type,
                            cfg_scale=cfg_scale,
                            stage1_steps=stage1_steps,
                            stage2_steps=stage2_steps,
                        )
                    else:
                        # T2V mode: Pure LTX T2V
                        enhanced_prompt = prompt_enhancer.enhance_flash_prompt(scene, seed=seed + i, mode="t2v")
                        
                        success, error = ltx_gen.generate_t2v(
                            prompt=enhanced_prompt,
                            output_path=str(segment_output),
                            frames=frames_per_segment,
                            seed=seed + i,
                            pipeline_type=pipeline_type,
                            cfg_scale=cfg_scale,
                            stage1_steps=stage1_steps,
                            stage2_steps=stage2_steps,
                        )
                    
                    if not success:
                        st.error(f"段 {i+1} 生成失败: {error}")
                        break
                    
                    segment_paths.append(str(segment_output))
                    progress_bar.progress((i + 1) / scene_count)
                
                # Concatenate videos
                if len(segment_paths) == scene_count:
                    status_text.text("拼接视频中...")
                    
                    import subprocess
                    output_path = output_dir / "flash_output.mp4"
                    filelist_path = output_dir / "filelist.txt"
                    
                    # Create file list with absolute paths
                    with open(filelist_path, "w") as f:
                        for path in segment_paths:
                            # Convert to absolute path
                            abs_path = Path(path).resolve()
                            f.write(f"file '{abs_path}'\n")
                    
                    # Concatenate with ffmpeg
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", str(filelist_path),
                        "-c", "copy",
                        str(output_path)
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True)
                    
                    # Clean up
                    try:
                        filelist_path.unlink()
                    except:
                        pass
                    
                    if result.returncode == 0 and output_path.exists():
                        progress_bar.progress(1.0)
                        status_text.text("✅ 生成完成！")
                        st.success("✅ 快闪视频生成成功！")
                        st.video(str(output_path))
                        
                        # Show segment info
                        with st.expander("📊 生成详情"):
                            for i, scene in enumerate(scenes):
                                st.markdown(f"**段 {i+1}:** {scene}")
                    else:
                        st.error("视频拼接失败")
                else:
                    st.error("部分段生成失败，无法拼接")

# Tab 4: I2V Generation
with tab4:
    st.header("🖼️ 图像生成视频 (I2V)")
    st.markdown("上传图像，用 LTX I2V 动画化")
    st.info("💡 **提示**: I2V 推荐使用单阶段模式，指令遵循效果更好。两阶段模式 CFG 效果会被稀释。")
    
    # Two-column layout
    col_params, col_content = st.columns([1, 2])
    
    with col_params:
        st.subheader("⚙️ 参数设置")
        use_hq = st.checkbox("使用 HQ 模式", value=False, key="i2v_hq", help="I2V 推荐使用单阶段")
        use_two_stage = st.checkbox("使用两阶段", value=False, disabled=use_hq, key="i2v_two_stage", help="I2V 推荐使用单阶段")
        
        st.markdown("**视频尺寸**")
        height = st.number_input("Height", 256, 544, 480, 32, key="i2v_height")
        width = st.number_input("Width", 384, 832, 704, 32, key="i2v_width")
        frames = st.number_input("Frames", 9, 97, 65, 8, key="i2v_frames")
        seed = st.number_input("Seed", value=42, key="i2v_seed")
        
        if use_hq:
            pipeline_type = "two-stage-hq"
            st.markdown("**HQ 参数**")
            st.warning("⚠️ I2V 两阶段模式 CFG 效果会被稀释")
            cfg_scale = st.slider("CFG Scale", 1.0, 10.0, 6.0, 0.5, key="i2v_cfg", help="I2V 两阶段需要更高的 CFG")
            stage1_steps = st.number_input("Stage 1 Steps", 5, 50, 30, 5, key="i2v_s1")
            stage2_steps = st.number_input("Stage 2 Steps", 1, 10, 3, 1, key="i2v_s2", help="更少步数减少稀释")
        elif use_two_stage:
            pipeline_type = "two-stage"
            st.markdown("**两阶段参数**")
            st.warning("⚠️ I2V 两阶段模式 CFG 效果会被稀释")
            cfg_scale = st.slider("CFG Scale", 1.0, 10.0, 6.0, 0.5, key="i2v_cfg", help="I2V 两阶段需要更高的 CFG")
            stage1_steps = st.number_input("Stage 1 Steps", 5, 50, 30, 5, key="i2v_s1")
            stage2_steps = st.number_input("Stage 2 Steps", 1, 10, 3, 1, key="i2v_s2", help="更少步数减少稀释")
        else:
            pipeline_type = "one-stage"
            st.success("✅ 单阶段模式（推荐）")
            cfg_scale = 3.0
            stage1_steps = 30
            stage2_steps = 3
    
    with col_content:
        uploaded_file = st.file_uploader("上传图像", type=["png", "jpg", "jpeg"], key="i2v_upload")
        prompt = st.text_area("Prompt", height=100, key="i2v_prompt", placeholder="描述视频运动...")
        
        if st.button("🚀 生成视频", type="primary", key="i2v_gen"):
            if not uploaded_file:
                st.error("请上传图像")
            elif not prompt:
                st.error("请输入 Prompt")
            else:
                # Save uploaded file
                output_dir = config.ltx.output_dir / "i2v"
                output_dir.mkdir(parents=True, exist_ok=True)
                image_path = output_dir / "uploaded_image.png"
                
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.image(str(image_path), caption="上传的图像", use_column_width=True)
                
                # Generate I2V
                video_output = output_dir / "i2v_output.mp4"
                
                with st.spinner("生成中..."):
                    success, error = ltx_gen.generate_i2v(
                        prompt=prompt,
                        image_path=str(image_path),
                        output_path=str(video_output),
                        height=height,
                        width=width,
                        frames=frames,
                        seed=seed,
                        pipeline_type=pipeline_type,
                        cfg_scale=cfg_scale,
                        stage1_steps=stage1_steps,
                        stage2_steps=stage2_steps,
                    )
                
                if success:
                    st.success("✅ 生成成功！")
                    st.video(str(video_output))
                else:
                    st.error(f"❌ 生成失败: {error}")

st.markdown("---")
st.markdown("🤖 Generated with CodeArts Agent | LTX-2 MLX Web UI (Refactored)")
