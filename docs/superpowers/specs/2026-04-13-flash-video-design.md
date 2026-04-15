# 快闪视频工作流设计

**日期**: 2026-04-13
**项目**: ltx-2-mlx + flux-gguf-image-gen 快闪风格视频拼接

## 概述

将多个赛博朋克主题词通过 AI 组合成多个片段，每个片段经过 FLUX 图像生成 → LTX-2 I2V 视频生成，最终用 ffmpeg 硬切拼接成完整快闪视频。

## 工作流程

```
步骤 1: 用户输入
├── 主题词列表 (每行一个)
├── 片段数量 (3-10)
├── 每段帧数 (49 默认)
└── Seed

步骤 2: AI 场景规划
├── ScenePlanner 将主题词扩展为片段描述
└── 每个片段包含: 详细描述、运动提示词

步骤 3: 逐个生成片段
├── FLUX 生成首帧图像
├── LTX-2 I2V 生成视频 (8步单阶段)
└── 保存到临时目录

步骤 4: ffmpeg 硬切拼接
├── 创建 filelist.txt
└── ffmpeg concat 拼接

步骤 5: 预览和下载
├── 显示完整视频
└── 下载按钮
```

## AI 提示词

### ScenePlanner Prompt

```
你是一个快闪视频策划。根据主题词列表，生成 N 个片段描述。

要求:
- 每个片段 2-4 句话
- 包含主体、环境、镜头运动
- 强调视觉冲击和场景氛围
- 运动描述要具体可执行

输出格式 (每行一个片段):
[片段1描述]
---
[片段2描述]
---
...

片段数量: {count}
```

### I2V Motion Prompt

用户不单独输入运动描述，由 AI 根据场景自动推断自然动态。

## 技术架构

| 组件 | 说明 |
|------|------|
| `FlashWorkflowState` | 会话状态管理 |
| `plan_flash_scenes()` | AI 场景规划 |
| `generate_flash_segment()` | 单个片段生成 (FLUX + LTX) |
| `concatenate_videos()` | ffmpeg 拼接 |
| `render_flash_tab()` | UI 渲染 |

## 参数

| 参数 | 默认值 | 范围 |
|------|--------|------|
| 片段数量 | 5 | 3-10 |
| 每段帧数 | 49 (8k+1) | 9-193 |
| Seed | 42 | -1 或正整数 |

## 拼接方式

ffmpeg concat (硬切，无过渡):

```bash
ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.mp4
```

## 状态管理

```python
if "flash_workflow" not in st.session_state:
    st.session_state.flash_workflow = {
        "theme_words": "",        # 用户输入的主题词
        "scene_count": 5,         # 片段数量
        "frames_per_segment": 49, # 每段帧数
        "seed": 42,
        "scenes": [],             # AI 规划的片段
        "segments": [],           # 生成的片段路径
        "status": "idle",         # idle, planning, generating, concatenating, done
        "current_segment": 0,     # 当前生成片段索引
        "output_path": None,
    }
```

## UI 布局

新增 Tab: "⚡ 快闪"

```
┌────────────────────────────────────────────────────────────────────┐
│  主题词列表 (每行一个)                                                │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  neon city rain                                            │   │
│  │  holographic ads flying cars                               │   │
│  │  underground hacker lab                                    │   │
│  │  megacorp boardroom                                        │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  片段数量 [5▼]  每段帧数 [49▼]  Seed [42]                         │
│                                                                    │
│                              [⚡ 生成快闪视频]                       │
│                                                                    │
│  生成进度: [2/5] 正在生成: underground hacker lab                   │
│  ████████████████░░░░░░░░  72%                                    │
│  图像: ✅  视频: 🔄                                                  │
│                                                                    │
│  预览: [视频播放]                                                    │
│                                                                    │
│  [📥 下载]  [🔄 新建]                                               │
└────────────────────────────────────────────────────────────────────┘
```

## 实现清单

- [ ] 新增 Tab: "⚡ 快闪"
- [ ] 主题词输入 text_area (多行)
- [ ] 参数设置 (片段数量、帧数、seed)
- [ ] AI 场景规划函数 `plan_flash_scenes()`
- [ ] 片段生成函数 `generate_flash_segment()`
- [ ] ffmpeg 拼接函数 `concatenate_videos()`
- [ ] 进度显示 UI
- [ ] 视频预览和下载
- [ ] 状态重置逻辑
