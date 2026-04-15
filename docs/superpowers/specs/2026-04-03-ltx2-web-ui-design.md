# LTX-2 Video Generator Web UI Design

## Project Overview

**Project name**: LTX-2 Video Generator  
**Type**: Web Application (Streamlit)  
**Purpose**: 提供一个可视化界面，通过 Web 前端调用 ltx-2-mlx 生成视频  
**Target users**: 需要使用 LTX-2 进行视频生成的用户

---

## Technical Stack

- **Frontend**: Streamlit (Python)
- **Backend**: 直接调用 `ltx_pipelines_mlx` CLI
- **Deployment**: 本地运行 (`streamlit run app.py`)

---

## UI/UX Specification

### Layout Structure

```
┌─────────────────────────────────────────────────┐
│            LTX-2 Video Generator                │
│                    (Header)                     │
├─────────────────────────────────────────────────┤
│                                                 │
│  Prompt Input                                   │
│  ┌─────────────────────────────────────────┐   │
│  │ Enter your prompt here...               │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  Generation Mode                                │
│  ○ One-stage q8 (Fast, 8 steps)                │
│  ○ Two-stage q4 (Quality, dev model + CFG)     │
│                                                 │
│  Parameters                                     │
│  Height: [480]  Width: [704]                   │
│  Frames: [97]    Seed: [42]                    │
│                                                 │
│  [Generate Video]                               │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  Generation Status                              │
│  Status: Ready / Generating / Complete         │
│  Progress: [███████████████] 100%              │
│  Time: 112.6s                                   │
│                                                 │
│  Video Result                                   │
│  ┌─────────────────────────────────────────┐   │
│  │                                         │   │
│  │           [Video Player]                │   │
│  │                                         │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  Video Info:                                    │
│  - Size: 2.9MB                                  │
│  - Duration: 4.0s                               │
│  - Resolution: 704x480                         │
│                                                 │
│  [Download] [Open Folder]                      │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Visual Design

- **Primary color**: #4A90D9 (蓝色)
- **Secondary color**: #2D3748 (深灰)
- **Background**: #FFFFFF (白色卡片)
- **Accent**: #48BB78 (绿色，成功状态)
- **Font**: System default (Streamlit 主题)

### Components

| Component | Description |
|-----------|-------------|
| Textarea | 提示词输入，支持多行 |
| Radio buttons | 模式选择 (One-stage / Two-stage) |
| Number inputs | 分辨率、帧数、seed |
| Button | 触发生成 |
| Progress bar | 显示生成进度 |
| Video player | 预览生成结果 |
| Download button | 下载视频文件 |

---

## Functionality Specification

### Core Features

1. **提示词输入**
   - 多行文本输入
   - 默认提示词示例

2. **生成模式选择**
   - One-stage q8: 蒸馏模型，8 步，速度快 (~2min)
   - Two-stage q4: dev 模型 + CFG + LoRA，质量更好 (~5min)

3. **参数配置**
   - Height: 默认 480，可选 256/320/480/544
   - Width: 默认 704，可选 384/512/704/832
   - Frames: 默认 97，必须 8k+1 (如 9/25/41/49/65/81/97/121)
   - Seed: 默认 42，-1 为随机

4. **生成执行**
   - 点击按钮触发 CLI 命令
   - 实时显示进度和日志
   - 完成后显示视频预览

5. **结果展示**
   - 内嵌视频播放器
   - 显示文件大小、时长、分辨率
   - 下载按钮

### Edge Cases

- **OOM**: 内存不足时提示用户降低参数
- **生成失败**: 显示错误信息，允许重试
- **参数不合法**: 输入验证 (帧数必须 8k+1)

---

## Acceptance Criteria

1. ✅ 用户可以输入提示词并选择生成模式
2. ✅ 支持配置分辨率、帧数、seed
3. ✅ 点击生成后执行对应 CLI 命令
4. ✅ 实时显示生成进度
5. ✅ 生成完成后展示视频播放器
6. ✅ 显示视频文件信息
7. ✅ 可以下载视频文件

---

## File Structure

```
ltx-2-mlx/
├── app.py                    # Streamlit 主应用
├── requirements.txt          # 依赖 (streamlit)
└── README.md                 # 使用说明
```