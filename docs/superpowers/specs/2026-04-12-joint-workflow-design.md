# 联合生成工作流设计

**日期**: 2026-04-12
**项目**: ltx-2-mlx + flux-gguf-image-gen 联合生成

## 概述

使用 FLUX 生成高质量图像作为首帧，LTX-2 I2V 基于该图像生成视频。用户输入场景描述、场景关键词和运动描述，AI 自动分离场景和运动提示词。

## 工作流程

```
步骤 1: 用户输入
├── FLUX 场景描述 (自由文本) → 直接作为 FLUX 提示词
├── 场景关键词 (逗号分隔) → AI 用于生成 I2V 提示词
└── 运动描述 (自由文本) → AI 用于生成 I2V 提示词

步骤 2: FLUX 生成图像
├── 显示图像预览
├── [重新生成图像] → 回到步骤 2
└── [确认并生成视频] → 步骤 3

步骤 3: AI 生成 I2V 提示词
├── 分析场景关键词 + 运动描述
└── 输出 I2V 提示词 (只读显示)

步骤 4: LTX-2 I2V 生成视频
├── 分辨率自动继承 FLUX (调整为 8x 倍数)
├── 用户设置 Frames, Steps, Seed
└── 显示视频预览 + 下载
```

## UI 布局

新增第三个 Tab: "➕ 联合生成"

```
┌────────────────────────────────────────────────────────────────────┐
│  🎬 LTX-2 │ 🖼️ FLUX │ ➕ 联合生成                                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ╭─────────────── FLUX 场景描述 ────────────────╮                  │
│  │  A serene Japanese temple at dawn...          │                  │
│  ╰──────────────────────────────────────────────╯                  │
│                                                                    │
│  ╭─────────────── 场景关键词 (用于AI) ──────────╮                  │
│  │  Japanese temple, cherry blossoms, mist...     │                  │
│  ╰──────────────────────────────────────────────╯                  │
│                                                                    │
│  ╭─────────────── 运动描述 ─────────────────────╮                  │
│  │  Camera pans slowly left, petals falling...   │                  │
│  ╰──────────────────────────────────────────────╯                  │
│                                                                    │
│  [🖼️ 生成图像]                                                      │
│                                                                    │
│  ════════════════════ 图像预览 ════════════════════                  │
│  │                        │                                        │
│  │    [图像预览区域]        │                                        │
│  │                        │                                        │
│  ╰────────────────────────────────────────────────────╯           │
│                                                                    │
│  [🔄 重新生成图像]        [✅ 确认并生成视频]                          │
│                                                                    │
│  ── I2V 提示词 (AI生成) ──────────────────────────────              │
│  │  Style: cinematic, Japanese temple, cherry blossoms...  │         │
│  │  Primary motion: camera pans left...                   │         │
│  ╰──────────────────────────────────────────────────────╯          │
│                                                                    │
│  视频参数: Frames [97▼] Steps [8▼] Seed [42]                       │
│                                                                    │
│  [🎬 生成视频]                                                      │
│                                                                    │
│  ════════════════════ 视频预览 ════════════════════                  │
│  │                        │                                        │
│  │    [视频预览区域]        │                                        │
│  │                        │                                        │
│  ╰────────────────────────────────────────────────────╯           │
│                                                                    │
│  [📥 下载视频]                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## AI 提示词生成

### I2V 提示词模板

```
Style: [cinematic/cinematic-realistic], [场景关键词拼接].
Primary motion: [运动描述扩展].
Secondary elements: [基于场景的次要运动元素，如风吹、阴影等].
Camera: [相机运动详情].
Audio: [场景环境音建议].
```

### 示例

**输入:**
- 场景关键词: "Japanese temple, cherry blossoms, morning mist, golden sunlight"
- 运动描述: "Camera pans slowly left, petals falling, figure walks through corridor"

**生成提示词:**
```
Style: cinematic, Japanese temple, cherry blossoms, morning mist, golden sunlight.
Primary motion: Camera pans slowly left as cherry blossom petals drift in the gentle breeze.
Secondary elements: Morning mist swirls gently, sunlight creates long shadows across the courtyard.
Camera: Smooth leftward pan maintaining focus on the central figure, shallow depth of field.
Audio: Soft wind through branches, distant bird calls, footsteps on wooden floorboards.
```

## 参数设置

### FLUX 参数 (高级折叠)
- Width: 256-2048, 默认 1024
- Height: 256-2048, 默认 1024
- Steps: 1-50, 默认 4
- CFG Scale: 1.0-10.0, 默认 1.0

### LTX-2 参数
- 分辨率: 自动继承 FLUX (调整为 32 倍数，限制 256-832)
- Frames: 8k+1, 默认 97
- Steps: 1-50, 默认 8 (one-stage)
- Seed: -1 或具体值, 默认 42

## 状态管理

```python
if "joint_workflow" not in st.session_state:
    st.session_state.joint_workflow = {
        "flux_image": None,           # str, FLUX生成的图像路径
        "flux_params": {},            # dict, FLUX参数
        "scene_keywords": "",         # str, 用户输入的场景关键词
        "motion_description": "",     # str, 用户输入的运动描述
        "i2v_prompt": None,           # str, AI生成的I2V提示词
        "video_path": None,           # str, LTX生成的视频路径
    }
```

## 实现清单

- [ ] 新增 Tab: "联合生成"
- [ ] FLUX 场景描述输入框
- [ ] 场景关键词输入框
- [ ] 运动描述输入框
- [ ] FLUX 生成图像功能 (复用 `run_flux_generation`)
- [ ] 图像预览区域
- [ ] 重新生成图像按钮
- [ ] 确认并生成视频按钮
- [ ] AI I2V 提示词生成 (新函数 `generate_i2v_prompt`)
- [ ] I2V 提示词只读显示
- [ ] LTX-2 I2V 参数设置 (分辨率自动继承)
- [ ] LTX-2 I2V 生成功能 (复用 `run_generation` + `--image`)
- [ ] 视频预览区域
- [ ] 下载视频按钮
- [ ] 状态重置逻辑

## 依赖

- 现有 `run_flux_generation()` 函数
- 现有 `run_generation()` 函数 (需支持 `--image` 参数)
- 现有 `enhance_prompt()` 函数 (作为 I2V 提示词生成参考)
