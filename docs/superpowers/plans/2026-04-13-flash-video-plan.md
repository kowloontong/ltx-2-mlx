# 快闪视频工作流实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development to implement this plan.

**Goal:** 在 app.py 中新增 "⚡ 快闪" Tab，实现主题词 → FLUX图像 → LTX-I2V视频 → ffmpeg拼接 的完整快闪视频工作流。

**Architecture:** 新增 Tab + 若干函数。复用 `run_flux_generation`、`run_generation`（加 `--image`）、ffmpeg concat。

**Tech Stack:** Streamlit, subprocess, 现有 `run_flux_generation()`, 现有 `run_generation()`

---

## Chunk 1: 常量和 Prompt 定义

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 添加 ScenePlanner Prompt**

在 `JOINT_I2V_SYSTEM_PROMPT` 后添加:

```python
FLASH_SCENE_PLANNER_PROMPT = """你是一个快闪视频策划。根据主题词列表，生成 N 个片段描述。

要求:
- 每个片段 2-4 句话，包含主体、环境、镜头运动
- 强调视觉冲击和场景氛围
- 运动描述要具体可执行
- 保持赛博朋克风格

输出格式 (每行一个片段，用---分隔):
[片段1描述]
---
[片段2描述]
---
...

重要: 输出只有片段描述，不要其他文字。片段数量: {count}"""
```

---

## Chunk 2: Flash 状态和 Tab 框架

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 添加 flash workflow 状态**

在 `if "joint_workflow"` 后添加:

```python
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
    }
```

- [ ] **Step 2: 添加 render_flash_tab() 函数框架**

在 `render_joint_tab()` 定义前添加:

```python
def render_flash_tab():
    """快闪视频生成 Tab"""
    st.markdown("### ⚡ 快闪视频")
    st.caption("多个主题词 → AI 规划 → FLUX图像 → LTX视频 → ffmpeg拼接")
```

- [ ] **Step 3: 修改 main() 添加第四个 Tab**

找到 tabs 定义，添加 flash tab:

```python
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎬 视频生成 (LTX-2)", 
        "🖼️ 图像生成 (FLUX)", 
        "➕ 联合生成",
        "⚡ 快闪"
    ])
    
    with tab4:
        render_flash_tab()
```

---

## Chunk 3: Flash Tab UI (输入部分)

**Files:**
- Modify: `app.py` (render_flash_tab 函数内)

- [ ] **Step 1: 主题词输入**

在 `render_flash_tab()` 函数内添加:

```python
    state = st.session_state.flash_workflow

    theme_words = st.text_area(
        "主题词列表 (每行一个)",
        height=120,
        placeholder="neon city rain\nholographic ads flying cars\nunderground hacker lab\nmegacorp boardroom",
        value=state.get("theme_words", ""),
        key="flash_theme_words"
    )
    state["theme_words"] = theme_words
```

- [ ] **Step 2: 参数设置**

```python
    col_count, col_frames, col_seed = st.columns(3)
    with col_count:
        scene_count = st.selectbox(
            "片段数量",
            [3, 4, 5, 6, 7, 8, 9, 10],
            index=2,
            key="flash_scene_count"
        )
    with col_frames:
        frames_per_segment = st.selectbox(
            "每段帧数",
            [9, 25, 41, 49, 65, 81, 97],
            index=3,
            key="flash_frames"
        )
    with col_seed:
        seed = st.number_input("Seed (-1=随机)", value=42, key="flash_seed")

    state["scene_count"] = scene_count
    state["frames_per_segment"] = frames_per_segment
    state["seed"] = seed
```

- [ ] **Step 3: 生成按钮**

```python
    generate_btn = st.button("⚡ 生成快闪视频", key="flash_generate", use_container_width=True)
```

---

## Chunk 4: AI 场景规划函数

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 添加 plan_flash_scenes() 函数**

在 `generate_i2v_prompt()` 后添加:

```python
def plan_flash_scenes(theme_words: str, scene_count: int, seed: int = 10) -> list[str]:
    """将主题词列表扩展为场景描述列表"""
    llm = _get_llm()

    prompt = FLASH_SCENE_PLANNER_PROMPT.format(count=scene_count)

    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": theme_words},
        ],
        max_tokens=512,
        temperature=0.8,
        seed=seed,
    )

    output = result["choices"][0]["message"]["content"].strip()

    # 按 --- 分割片段
    scenes = [s.strip() for s in output.split("---") if s.strip()]

    # 如果数量不对，截断或补充
    if len(scenes) > scene_count:
        scenes = scenes[:scene_count]
    elif len(scenes) < scene_count and scenes:
        # 复制最后一个片段填充
        while len(scenes) < scene_count:
            scenes.append(scenes[-1])

    return scenes
```

---

## Chunk 5: 单片段生成函数

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 添加 generate_flash_segment() 函数**

```python
def generate_flash_segment(
    scene_description: str,
    output_image_path: str,
    output_video_path: str,
    width: int = 1024,
    height: int = 1024,
    frames: int = 49,
    seed: int = 42,
) -> tuple[bool, str]:
    """生成单个快闪片段: FLUX图像 + LTX I2V视频
    
    Returns:
        (success, error_message)
    """
    # Step 1: FLUX 生成图像 (场景描述作为 prompt)
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

    # Step 2: LTX I2V 生成视频 (8步单阶段)
    python_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python3")
    ltx_height = (height // 32) * 32
    ltx_height = max(256, min(544, ltx_height))
    ltx_width = (width // 32) * 32
    ltx_width = max(256, min(832, ltx_width))

    cmd = [
        python_path, "-m", "ltx_pipelines_mlx", "generate",
        "--model", "models/ltx-2.3-mlx-q8",
        "--prompt", scene_description,
        "--output", output_video_path,
        "--height", str(ltx_height),
        "--width", str(ltx_width),
        "--frames", str(frames),
        "--seed", str(seed),
        "--steps", "8",
        "--image", output_image_path,
        "--quiet"
    ]

    ltx_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd="/Users/junhui/work/ltx-2-mlx"
    )

    for line in ltx_process.stdout:
        pass

    ltx_return = ltx_process.wait()

    if ltx_return != 0 or not os.path.exists(output_video_path):
        return False, "LTX 视频生成失败"

    return True, ""
```

---

## Chunk 6: ffmpeg 拼接函数

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 添加 concatenate_videos() 函数**

```python
def concatenate_videos(video_paths: list, output_path: str) -> bool:
    """使用 ffmpeg concat 拼接多个视频 (硬切)"""
    # 创建临时 filelist
    filelist_path = output_path + ".filelist.txt"

    with open(filelist_path, "w") as f:
        for video_path in video_paths:
            f.write(f"file '{video_path}'\n")

    # ffmpeg concat
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", filelist_path,
        "-c", "copy",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # 清理 filelist
    try:
        os.remove(filelist_path)
    except:
        pass

    return result.returncode == 0 and os.path.exists(output_path)
```

---

## Chunk 7: Flash 生成逻辑

**Files:**
- Modify: `app.py` (render_flash_tab 函数内，generate_btn 处理)

- [ ] **Step 1: 添加生成按钮逻辑**

在 generate_btn 定义后添加:

```python
    if generate_btn:
        if not theme_words.strip():
            st.error("请输入主题词")
        else:
            state["status"] = "planning"
            state["scenes"] = []
            state["segments"] = []
            state["current_segment"] = 0

            # AI 规划场景
            with st.spinner("✨ AI 正在规划场景..."):
                scenes = plan_flash_scenes(
                    theme_words=theme_words,
                    scene_count=scene_count,
                    seed=seed if seed > 0 else 42
                )
                state["scenes"] = scenes

            st.success(f"✅ 规划完成! 将生成 {len(scenes)} 个片段")

            # 创建临时目录
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="flash_")
            state["temp_dir"] = temp_dir
            state["status"] = "generating"

            # 逐个生成片段
            segment_paths = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, scene in enumerate(scenes):
                state["current_segment"] = i
                status_text.text(f"正在生成片段 {i+1}/{len(scenes)}: {scene[:50]}...")
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
                    st.error(f"片段 {i+1} 生成失败: {error}")
                    state["status"] = "error"
                    break

            progress_bar.progress(1.0)

            if state["status"] != "error" and segment_paths:
                # 拼接视频
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
```

---

## Chunk 8: 预览和下载 UI

**Files:**
- Modify: `app.py` (render_flash_tab 函数内，生成逻辑之后)

- [ ] **Step 1: 添加预览和下载区域**

在生成逻辑之后添加:

```python
    # 预览和下载
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
                label="📥 下载视频",
                data=f,
                file_name=f"flash_video_{int(time.time())}.mp4",
                mime="video/mp4"
            )

        if st.button("🔄 新建任务", key="flash_new_task"):
            # 清理临时文件
            if state.get("temp_dir") and os.path.exists(state["temp_dir"]):
                import shutil
                shutil.rmtree(state["temp_dir"], ignore_errors=True)

            # 重置状态
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
            }
            st.rerun()
```

---

## Chunk 9: 测试

**Files:**
- 无需创建测试文件

- [ ] **Step 1: 语法检查**

```bash
cd /Users/junhui/work/ltx-2-mlx
.venv/bin/python -c "import ast; ast.parse(open('app.py').read()); print('Syntax OK')"
```

- [ ] **Step 2: 手动测试流程**

```bash
cd /Users/junhui/work/ltx-2-mlx
uv run streamlit run app.py
```

测试:
1. 选择 "⚡ 快闪" Tab
2. 输入主题词 (每行一个)
3. 设置片段数量和每段帧数
4. 点击 "⚡ 生成快闪视频"
5. 观察生成进度
6. 预览并下载最终视频
