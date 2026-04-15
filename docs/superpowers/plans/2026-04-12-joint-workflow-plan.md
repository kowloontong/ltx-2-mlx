# 联合生成工作流实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ltx-2-mlx/app.py 中新增 "联合生成" Tab，实现 FLUX 图像 → LTX-2 I2V 视频的完整工作流。

**Architecture:** 单 Tab 页，包含三个阶段：输入 → FLUX 图像生成 → LTX-2 I2V 视频生成。AI 辅助生成 I2V 提示词。

**Tech Stack:** Streamlit, subprocess, 现有 `run_flux_generation()`, 现有 `run_generation()`

---

## Chunk 1: 状态和模板定义

**Files:**
- Modify: `app.py` (开头部分，添加状态初始化和 prompt 模板)

- [ ] **Step 1: 添加 joint workflow 状态初始化**

在 `FLUX_OUTPUT_DIR` 定义后添加:

```python
# Joint workflow state
if "joint_workflow" not in st.session_state:
    st.session_state.joint_workflow = {
        "flux_image": None,
        "flux_params": {},
        "scene_keywords": "",
        "motion_description": "",
        "i2v_prompt": None,
        "video_path": None,
    }
```

- [ ] **Step 2: 添加 I2V prompt 模板常量**

在 `T2V_SYSTEM_PROMPT` 和 `I2V_SYSTEM_PROMPT` 后添加:

```python
JOINT_I2V_SYSTEM_PROMPT = """You are a Creative Assistant generating image-to-video prompts.

Given scene keywords and motion description, create a detailed I2V prompt.

#### Guidelines:
- Combine scene keywords into a cohesive visual description
- Expand motion description into specific, filmable actions
- Add secondary ambient motions that enhance realism
- Describe camera behavior explicitly
- Suggest ambient audio for immersion

#### Output Format (Strict):
Style: [style], [scene description].
Primary motion: [expanded motion].
Secondary elements: [ambient motions].
Camera: [camera behavior].
Audio: [ambient sounds].
- Single paragraph.
- NO titles, headings, or Markdown formatting."""
```

---

## Chunk 2: 新增 Tab 渲染函数骨架

**Files:**
- Modify: `app.py` (末尾，添加 `render_joint_tab()` 函数)

- [ ] **Step 1: 添加 render_joint_tab() 函数框架**

在 `render_flux_tab()` 定义前添加:

```python
def render_joint_tab():
    """联合生成: FLUX 图像 + LTX-2 I2V 视频"""
    st.markdown("### ➕ 联合生成工作流")
    st.caption("FLUX 生成高质量图像 → LTX-2 I2V 基于图像生成视频")
```

- [ ] **Step 2: 修改 main() 添加第三个 Tab**

找到 `main()` 函数中的 tabs 定义，修改为:

```python
    tab1, tab2, tab3 = st.tabs(["🎬 视频生成 (LTX-2)", "🖼️ 图像生成 (FLUX)", "➕ 联合生成"])
    
    with tab1:
        render_video_tab()
    
    with tab2:
        render_flux_tab()
    
    with tab3:
        render_joint_tab()
```

---

## Chunk 3: 联合生成 Tab UI 实现 (阶段1: 输入)

**Files:**
- Modify: `app.py` (render_joint_tab 函数内)

- [ ] **Step 1: FLUX 场景描述输入框**

在 `render_joint_tab()` 函数内添加:

```python
    # 获取当前状态
    state = st.session_state.joint_workflow
    
    # FLUX 场景描述
    flux_prompt = st.text_area(
        "FLUX 场景描述",
        height=80,
        placeholder="A serene Japanese temple at dawn, cherry blossoms in full bloom...",
        value=state.get("flux_prompt", ""),
        key="joint_flux_prompt"
    )
```

- [ ] **Step 2: 场景关键词和运动描述输入框**

```python
    col1, col2 = st.columns(2)
    with col1:
        scene_keywords = st.text_input(
            "场景关键词 (用于AI)",
            placeholder="Japanese temple, cherry blossoms, morning mist, golden sunlight",
            value=state.get("scene_keywords", ""),
            key="joint_scene_keywords"
        )
    with col2:
        motion_description = st.text_input(
            "运动描述",
            placeholder="Camera pans slowly left, petals falling...",
            value=state.get("motion_description", ""),
            key="joint_motion_description"
        )
```

- [ ] **Step 3: 保存输入到状态**

```python
    state["flux_prompt"] = flux_prompt
    state["scene_keywords"] = scene_keywords
    state["motion_description"] = motion_description
```

- [ ] **Step 4: 高级选项 (FLUX 参数，折叠)**

```python
    with st.expander("⚙️ 高级选项 (FLUX)"):
        col_w, col_h, col_s, col_cfg = st.columns(4)
        with col_w:
            flux_width = st.number_input("Width", min_value=256, max_value=2048, value=1024, step=64, key="joint_flux_width")
        with col_h:
            flux_height = st.number_input("Height", min_value=256, max_value=2048, value=1024, step=64, key="joint_flux_height")
        with col_s:
            flux_steps = st.number_input("Steps", min_value=1, max_value=50, value=4, step=1, key="joint_flux_steps")
        with col_cfg:
            flux_cfg = st.slider("CFG", min_value=1.0, max_value=10.0, value=1.0, step=0.5, key="joint_flux_cfg")
```

---

## Chunk 4: FLUX 图像生成功能

**Files:**
- Modify: `app.py` (run_flux_generation 需要支持 width/height 参数)

- [ ] **Step 1: 修改 run_flux_generation 支持完整参数**

找到 `run_flux_generation()` 函数，修改为:

```python
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
        "python3", python_path, prompt,
        "-o", output_path,
        "-w", str(width), "--height", str(height),
        "-s", str(steps), "--cfg-scale", str(cfg_scale)
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=FLUX_SCRIPT_DIR
    )
    return process
```

- [ ] **Step 2: 在 render_joint_tab 添加 FLUX 生成按钮**

在高级选项后添加:

```python
    flux_generate_btn = st.button("🖼️ 生成图像", use_container_width=True)
```

- [ ] **Step 3: FLUX 生成按钮逻辑**

```python
    if flux_generate_btn:
        if not flux_prompt:
            st.error("请输入 FLUX 场景描述")
        else:
            timestamp = int(time.time())
            output_path = os.path.abspath(os.path.join(FLUX_OUTPUT_DIR, f"flux_joint_{timestamp}.png"))
            
            st.session_state.joint_workflow["flux_image"] = output_path
            st.session_state.joint_workflow["flux_params"] = {
                "width": flux_width,
                "height": flux_height,
            }
            
            # 调用 FLUX 生成
            with st.spinner("🖼️ 正在生成图像..."):
                process = run_flux_generation(
                    prompt=flux_prompt,
                    output_path=output_path,
                    width=flux_width,
                    height=flux_height,
                    steps=flux_steps,
                    cfg_scale=flux_cfg,
                )
            
            # 等待完成
            for line in process.stdout:
                pass  # 消耗输出
            
            return_code = process.wait()
            
            if return_code == 0 and os.path.exists(output_path):
                st.success("✅ 图像生成完成!")
                st.session_state.joint_workflow["flux_image"] = output_path
                st.rerun()
            else:
                st.error("❌ 图像生成失败")
```

---

## Chunk 5: 图像预览和操作按钮

**Files:**
- Modify: `app.py` (render_joint_tab 函数内)

- [ ] **Step 1: 添加图像预览区域**

在 FLUX 生成按钮逻辑后添加:

```python
    state = st.session_state.joint_workflow
    
    if state["flux_image"] and os.path.exists(state["flux_image"]):
        st.markdown("---")
        st.markdown("#### 🖼️ 图像预览")
        st.image(state["flux_image"], width=512)
        
        col_regen, col_confirm = st.columns(2)
        with col_regen:
            st.button("🔄 重新生成图像", on_click=lambda: None, key="regen_image_btn")
        with col_confirm:
            st.button("✅ 确认并生成视频", on_click=lambda: None, key="confirm_video_btn")
```

- [ ] **Step 2: 重新生成图像按钮逻辑**

在 `if state["flux_image"]` 块内，修改按钮:

```python
        col_regen, col_confirm = st.columns(2)
        with col_regen:
            if st.button("🔄 重新生成图像", key="regen_image_btn"):
                # 删除旧图像
                if os.path.exists(state["flux_image"]):
                    os.remove(state["flux_image"])
                state["flux_image"] = None
                st.rerun()
        with col_confirm:
            if st.button("✅ 确认并生成视频", key="confirm_video_btn"):
                st.session_state.joint_workflow["step"] = "generate_video"
                st.rerun()
```

---

## Chunk 6: AI I2V 提示词生成

**Files:**
- Modify: `app.py` (添加 generate_i2v_prompt 函数)

- [ ] **Step 1: 添加 generate_i2v_prompt 函数**

在 `_extract_enhanced_prompt` 函数后添加:

```python
def generate_i2v_prompt(scene_keywords: str, motion_description: str, seed: int = 10) -> str:
    """Generate I2V prompt from scene keywords and motion description."""
    import gc
    
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
    
    # 简单清理，提取 Style: 开始的内容
    if "Style:" in raw_output:
        return raw_output[raw_output.find("Style:"):]
    return raw_output
```

---

## Chunk 7: 阶段2 - LTX-2 I2V 参数和视频生成

**Files:**
- Modify: `app.py` (render_joint_tab 函数内)

- [ ] **Step 1: 添加确认后显示 I2V 提示词生成区域**

在图像预览区域后添加:

```python
    # 确认后进入视频生成阶段
    if st.session_state.joint_workflow.get("step") == "generate_video":
        st.markdown("---")
        st.markdown("#### 🎬 视频生成")
        
        state = st.session_state.joint_workflow
        
        # AI 生成 I2V 提示词
        if not state.get("i2v_prompt"):
            with st.spinner("✨ AI 正在生成 I2V 提示词..."):
                i2v_prompt = generate_i2v_prompt(
                    scene_keywords=state["scene_keywords"],
                    motion_description=state["motion_description"],
                    seed=42,
                )
                state["i2v_prompt"] = i2v_prompt
        
        # 显示生成的提示词
        st.markdown("**I2V 提示词 (AI生成)**")
        st.text_area(
            "提示词预览",
            value=state["i2v_prompt"],
            height=100,
            disabled=True,
            key="i2v_prompt_display"
        )
        
        # 重新生成提示词按钮
        if st.button("🔄 重新生成提示词"):
            state["i2v_prompt"] = None
            st.rerun()
```

- [ ] **Step 2: LTX-2 参数设置**

在提示词显示后添加:

```python
        # 继承 FLUX 分辨率并调整为 32 倍数
        flux_w = state["flux_params"].get("width", 1024)
        flux_h = state["flux_params"].get("height", 1024)
        
        # 调整为 32 倍数
        ltx_width = (flux_w // 32) * 32
        ltx_height = (flux_h // 32) * 32
        ltx_width = max(256, min(832, ltx_width))
        ltx_height = max(256, min(544, ltx_height))
        
        col_frame, col_step, col_seed = st.columns(3)
        with col_frame:
            frames = st.selectbox(
                "Frames (8k+1)",
                [9, 25, 41, 49, 65, 81, 97, 121, 145, 161, 193],
                index=6,  # 默认 97
                key="joint_frames"
            )
        with col_step:
            steps = st.number_input("Steps", min_value=1, max_value=50, value=8, step=1, key="joint_steps")
        with col_seed:
            seed = st.number_input("Seed (-1=随机)", value=42, key="joint_seed")
        
        st.info(f"💡 分辨率自动继承 FLUX: {ltx_width}x{ltx_height}")
```

- [ ] **Step 3: 生成视频按钮**

```python
        if st.button("🎬 生成视频", use_container_width=True):
            timestamp = int(time.time())
            video_path = os.path.abspath(os.path.join(OUTPUT_DIR, f"joint_video_{timestamp}.mp4"))
            
            st.session_state.joint_workflow["video_path"] = video_path
            
            with st.spinner("🎬 正在生成视频..."):
                # 构建 LTX 命令
                python_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python3")
                cmd = [
                    python_path, "-m", "ltx_pipelines_mlx", "generate",
                    "--model", "models/ltx-2.3-mlx-q8",
                    "--prompt", state["i2v_prompt"],
                    "--output", video_path,
                    "--height", str(ltx_height),
                    "--width", str(ltx_width),
                    "--frames", str(frames),
                    "--seed", str(seed),
                    "--steps", str(steps),
                    "--image", state["flux_image"],
                    "--quiet"
                ]
                
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    cwd="/Users/junhui/work/ltx-2-mlx"
                )
                
                # 等待完成
                for line in process.stdout:
                    pass
                
                return_code = process.wait()
                
                if return_code == 0 and os.path.exists(video_path):
                    st.success("✅ 视频生成完成!")
                    st.rerun()
                else:
                    st.error("❌ 视频生成失败")
```

---

## Chunk 8: 视频预览和下载

**Files:**
- Modify: `app.py` (render_joint_tab 函数内)

- [ ] **Step 1: 添加视频预览区域**

在视频生成逻辑后，函数末尾添加:

```python
    # 视频预览
    state = st.session_state.joint_workflow
    if state.get("video_path") and os.path.exists(state["video_path"]):
        st.markdown("---")
        st.markdown("#### 🎬 视频预览")
        st.video(state["video_path"])
        
        file_size = os.path.getsize(state["video_path"]) / (1024 * 1024)
        col1, col2 = st.columns(2)
        col1.metric("文件大小", f"{file_size:.1f} MB")
        col2.metric("耗时", "N/A")  # 可选：记录时间
        
        # 下载按钮
        with open(state["video_path"], "rb") as f:
            st.download_button(
                label="📥 下载视频",
                data=f,
                file_name=f"joint_video_{int(time.time())}.mp4",
                mime="video/mp4"
            )
        
        # 新建按钮
        if st.button("🔄 新建任务"):
            # 重置状态
            st.session_state.joint_workflow = {
                "flux_image": None,
                "flux_params": {},
                "scene_keywords": "",
                "motion_description": "",
                "i2v_prompt": None,
                "video_path": None,
                "step": None,
            }
            st.rerun()
```

---

## Chunk 9: 测试

**Files:**
- 无需创建测试文件，但需要手动验证

- [ ] **Step 1: 语法检查**

```bash
cd /Users/junhui/work/ltx-2-mlx
.venv/bin/python -c "import app; print('OK')"
```

- [ ] **Step 2: 启动 Streamlit 手动测试**

```bash
cd /Users/junhui/work/ltx-2-mlx
uv run streamlit run app.py
```

测试流程:
1. 选择 "联合生成" Tab
2. 输入 FLUX 场景描述
3. 输入场景关键词和运动描述
4. 点击 "生成图像"
5. 确认图像后点击 "确认并生成视频"
6. 查看 AI 生成的 I2V 提示词
7. 生成视频并预览
