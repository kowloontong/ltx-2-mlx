# Phase 2: Storyboard Pipeline 设计文档

## 概述

Storyboard Pipeline 是 ltx-2-mlx 的多镜头故事视频生成层。用户编写 JSON 故事板脚本，系统自动逐镜头生成视频并拼接为完整故事。

## 核心设计

### 每镜头生成流程

| 镜头 | 方法 | 理由 |
|------|------|------|
| 第1镜 | 直接 T2V 生成完整视频 | 省掉多余的 I2V 调用，113s/镜头 |
| 后续镜 | 提取上一镜尾帧 → I2V | 一次到位，保持镜头间视觉连贯 |

### 关键改进

- **Pipeline 复用**: T2V 和 I2V 共享底层模型权重，只在必要时重新加载
- **内存管理**: 每镜生成后调用 `aggressive_cleanup()`，避免 Metal cache 增长
- **降级策略**: 生成失败时自动降级到预览模式 (384x576, 41帧)

## JSON 脚本格式

```json
{
  "title": "故事标题",
  "style_anchor": "cinematic 4K, warm lighting",
  "character_description": "一个年轻女性，长发",
  "settings": {
    "height": 480,
    "width": 704,
    "num_frames": 97,
    "seed_base": 42,
    "crossfade_duration": 0.5,
    "output_dir": "output"
  },
  "shots": [
    {
      "id": 1,
      "prompt": "A woman walks through a temple corridor, cherry blossoms falling",
      "ref_image": "auto",
      "num_frames": 97
    },
    {
      "id": 2,
      "prompt": "She turns to look at the camera and smiles",
      "ref_image": "auto",
      "num_frames": 65
    }
  ]
}
```

## 验证 Gate (Phase 2)

### Step 0: 环境前置检查
- **PRE**: Phase 1 已完成，模型已下载
- **DO**: 检查 `models/ltx-2.3-mlx-q8` 存在，验证 ffmpeg 可用
- **POST**: 模型目录完整，ffmpeg 版本 >= 4.0
- **FAIL**: 重新执行 Phase 1 Step 3

### Step 1: 模块结构验证
- **PRE**: Step 0 PASS
- **DO**: 创建 `storyboard_pipeline.py`，验证 Python 语法
- **POST**: `from storyboard_pipeline import StoryboardPipeline` 无报错
- **FAIL**: 检查语法错误或缺失依赖

### Step 2: JSON 解析验证
- **PRE**: Step 1 PASS
- **DO**: 加载示例 storyboard.json，验证字段类型
- **POST**: 解析正确，边界检查通过 (帧数 ∈ 8k+1，分辨率 % 32 == 0)
- **FAIL**: 修复 JSON schema 或添加默认值

### Step 3: T2V 单镜生成验证
- **PRE**: Step 2 PASS
- **DO**: 单镜 T2V 生成 9帧，监控内存峰值
- **POST**: 输出文件存在，分辨率正确，内存峰值 < 48GB
- **FAIL**: OOM→降帧；crash→检查 Metal 驱动

### Step 4: I2V 扩展验证
- **PRE**: Step 3 PASS
- **DO**: 用首帧扩展为 41帧，验证首帧一致性
- **POST**: 输出 MP4 首帧与参考图视觉相似
- **FAIL**: I2V conditioning 未生效 → 检查参数

### Step 5: 过渡生成验证
- **PRE**: Step 4 PASS
- **DO**: 生成 2 镜过渡，验证帧间连续性
- **POST**: 过渡平滑，无明显跳变
- **FAIL**: 过渡失败 → 降级为硬切

### Step 6: 视频拼接验证
- **PRE**: Step 5 PASS
- **DO**: 拼接 3+ 镜，验证音视频同步
- **POST**: 总时长误差 < 0.1s，音画同步
- **FAIL**: ffmpeg concat 参数错误 → 检查编码

### Step 7: 端到端验证
- **PRE**: Step 6 PASS
- **DO**: 运行完整 3 镜 storyboard.json
- **POST**: 无报错，内存峰值 < 48GB
- **FAIL**: OOM→降级；crash→报告不可用

## 判定标准

| 结论 | 条件 |
|------|------|
| **PASS** | Steps 0-7 全部通过 |
| **PARTIAL** | 核心功能通过，拼接/过渡降级 |
| **FAIL** | 单镜生成失败或 OOM |

## 技术细节

### 帧数边界检查
帧数必须是 8k+1：9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97

### 分辨率边界检查
宽高必须是 32 的倍数

### 内存管理
每镜生成后调用 `aggressive_cleanup()`，监控内存峰值

### 错误处理
生成失败时自动降级到预览模式 (384x576, 41帧)
