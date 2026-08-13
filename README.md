# 面向城市场景的视觉多模态目标检测

> 第八届 AIC 算法大赛 — 算法挑战赛道
> RGB + 红外 + 深度 三模态融合目标检测系统

## 项目简介

本项目实现了一个基于 YOLOv8 架构的多模态目标检测系统，融合可见光（RGB）、热红外（IR）和深度（Depth）三种模态信息，用于城市场景下的 12 类目标检测。核心创新点是**分层模态门控融合（Stage-wise Modality Gating Fusion）**，通过轻量门控网络动态学习各模态在不同网络深度的融合权重。

### 12 类目标

| ID | 类别 | ID | 类别 | ID | 类别 |
|---|---|---|---|---|---|
| 0 | person | 4 | sign | 8 | light |
| 1 | boat | 5 | bicycle | 9 | garbagecan |
| 2 | animal | 6 | car | 10 | uav |
| 3 | seat | 7 | ball | 11 | tricycle |

### 评测指标

mAP@50-95（COCO 口径，IoU 从 0.50 到 0.95，步长 0.05，取 10 个阈值的平均 AP）

## 目录结构

```
multimodal-detection/
├── configs/
│   └── train.yaml              # 训练配置（模型/数据/损失/优化器/推理）
├── src/
│   ├── data/
│   │   └── multimodal_dataset.py   # 多模态数据集 + 同步增强 + collate
│   ├── models/
│   │   ├── multimodal_yolo.py      # 多模态模型（EarlyFusion / StageGateFusion）
│   │   ├── heads.py                # YOLOv8 检测头 + backbone 构建
│   │   ├── stage_gate_fusion.py    # 分层模态门控融合模块
│   │   └── late_fusion.py          # WBF 晚期融合（推理级）
│   ├── losses/
│   │   └── detection_loss.py       # CIoU + DFL + BCE + 辅助损失 + TAL 分配
│   ├── inference/
│   │   └── postprocess.py          # NMS/Soft-NMS + TTA + 格式转换
│   └── utils/
│       ├── io_utils.py             # 图像/标签 IO + letterbox + 坐标转换
│       ├── preprocessing.py        # 伪模态特征生成 + 深度自适应
│       └── metrics.py              # COCO mAP 评估
├── scripts/
│   ├── 00_setup_env.py             # 环境检查
│   ├── 01_preprocess.py            # 数据划分 + EDA + 辅助特征缓存
│   ├── 02_train.py                 # 两阶段训练（核心）
│   ├── 03_validate.py              # 验证集评估 + 可视化
│   ├── 04_infer.py                 # 测试集推理 + TTA + 鲁棒性测试
│   └── 05_export_submission.py     # 提交文件导出（TXT + ZIP）
├── tests/
│   └── smoke_test.py               # 冒烟测试（损失/数据/端到端/推理）
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境检查

```bash
python scripts/00_setup_env.py --data_root ../dataset
```

### 3. 数据预处理

```bash
# 划分 85/15 验证集 + EDA + 缓存辅助特征
python scripts/01_preprocess.py --data_root ../dataset --cache_aux
```

### 4. 训练

```bash
# 完整两阶段训练（Stage1 冻结主干 + Stage2 全网络微调）
python scripts/02_train.py --config configs/train.yaml

# 从断点恢复
python scripts/02_train.py --config configs/train.yaml --resume weights/checkpoints/last.pt

# 仅训练 Stage1（快速基线）
python scripts/02_train.py --config configs/train.yaml --stage1_only

# 使用早期融合（快速对比）
python scripts/02_train.py --config configs/train.yaml --fusion_mode early
```

### 5. 验证

```bash
python scripts/03_validate.py --config configs/train.yaml --weights weights/best/best.pt --visualize
```

### 6. 推理

```bash
# 标准推理
python scripts/04_infer.py --config configs/train.yaml --weights weights/best/best.pt

# TTA 推理（多尺度 + 翻转融合，提分）
python scripts/04_infer.py --config configs/train.yaml --weights weights/best/best.pt --tta

# 模态缺失鲁棒性测试
python scripts/04_infer.py --config configs/train.yaml --weights weights/best/best.pt --robustness_test
```

### 7. 导出提交

```bash
python scripts/05_export_submission.py --predictions outputs/predictions/test_predictions.pt
```

输出 `submission.zip`，包含每张测试图一个 TXT 文件。

## 技术架构

### 融合策略

#### 1. 早期融合（EarlyFusion）

三模态拼接为 5 通道输入 → 标准 YOLOv8 backbone → 检测头

- 优点：最简单、最快，backbone 可直接复用 COCO 预训练权重
- 缺点：模态间信息耦合过早，浅层物理语义未被充分利用

#### 2. 分层门控融合（StageGateFusion）— 核心方案

```
RGB Stem ──→ C2f 分支 ──┐
IR  Stem ──→ C2f 分支 ──┼→ 门控融合(1/4) → 共享主干 P3/P4/P5 → 检测头
Dep Stem ──→ C2f 分支 ──┘
```

- 三模态独立 Stem（各下采样到 1/4）
- 模态特定 C2f 分支提取浅层特征
- **ModalityGate**：1×1 压缩 → 拼接 → Sigmoid 门控 → 加权求和
- 可学习模态偏好偏置（浅层偏 Depth/IR，深层偏 RGB）
- 共享主干（P3/P4/P5）复用 YOLOv8 结构 + 预训练权重

### 损失函数

| 损失 | 说明 | 权重 |
|---|---|---|
| CIoU | 像素空间回归损失（fg 锚点） | 7.5 |
| BCE | 分类损失（全锚点） | 0.5 |
| DFL | 分布焦点损失（特征空间） | 1.5 |
| IR 对比 | 伪模态辅助：IR 温度对比图重建 | 0.30 × 0.15 |
| Depth 掩码 | 伪模态辅助：有效深度掩码重建 | 0.20 × 0.15 |
| Depth 法向 | 伪模态辅助：深度法向梯度重建 | 0.15 × 0.15 |

标签分配：TaskAlignedAssigner（topk=13, alpha=0.5, beta=6.0）

### 数据增强

- 三模态同步 letterbox（640×640）
- 随机水平翻转（三模态 + 标签同步）
- 随机缩放 + 中心裁剪/填充（三模态 + 标签同步）
- RGB 色域增强（HSV 抖动，不影响 IR/Depth 物理语义）
- 模态 Dropout（训练时随机失效单模态，提升鲁棒性）

### 深度图自适应

官方数据集深度图格式混合（16bit 毫米 PNG + 8bit JPEG），所有深度处理函数自动识别并统一处理：

- `depth_is_mm()`: `max > 255` → 16bit 毫米
- `normalize_depth()`: 16bit clip[0,19999]/19999; 8bit /255
- 所有相关函数（掩码/法向/伪彩色/归一化）均自适应

### 坐标系约定

```
原始归一化 YOLO (cx,cy,w,h)
    ↓ yolo_to_xyxy(scale, dw, dh, orig_shape)
letterbox 后 640 像素 xyxy ← 损失/匹配空间
    ↓ letterbox_xyxy_to_norm(scale, dw, dh, orig_shape)
原始归一化 YOLO ← 提交空间
```

非正方形图（1920×1080）的坐标转换使用真实原图宽高，不假设正方形。

## 配置说明

`configs/train.yaml` 关键参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `fusion_mode` | `stage_gate` | 融合策略（early / stage_gate） |
| `model_size` | `n` | 模型规模（n / s / m） |
| `img_size` | `640` | 输入尺寸 |
| `batch_size` | `16` | 批大小（24GB 显存建议 16） |
| `stage1_epochs` | `15` | Stage1 轮数（冻结主干） |
| `stage2_epochs` | `30` | Stage2 轮数（全网络微调） |
| `modal_dropout_prob` | `0.3` | 模态 Dropout 概率 |
| `use_aux` | `true` | 伪模态辅助监督 |
| `ema` | `true` | EMA 权重平滑 |
| `amp` | `true` | 混合精度训练 |
| `early_stopping` | `true` | 早停（patience=10） |

## 提交格式

每张测试图一个 TXT 文件，每行格式：

```
class_id cx cy w h confidence
```

- `class_id`: 0~11
- `cx, cy, w, h`: 归一化坐标 [0, 1]（相对原图宽高）
- `confidence`: 置信度 [0, 1]
- 单图最多 100 个检测框
- 所有 TXT 打包为 ZIP 提交

## 常见问题

### Q: CUDA 不可用怎么办？

系统会自动回退到 CPU。建议使用 GPU 训练，CPU 仅用于调试。可减小 `batch_size` 和 `img_size` 来适应显存。

### Q: 深度图格式混合如何处理？

所有深度处理函数已自适应 8bit/16bit 格式，无需手动处理。`01_preprocess.py` 的 EDA 会输出格式统计。

### Q: 如何提升精度？

1. 启用 TTA 推理：`--tta`（多尺度 + 翻转融合）
2. 增大模型：`model_size: "s"` 或 `"m"`
3. 增大输入：`img_size: 800` 或 `960`
4. 延长训练：增加 `stage2_epochs`
5. 调整 NMS：使用 Soft-NMS（默认已开启）

### Q: 如何断点恢复？

```bash
python scripts/02_train.py --resume weights/checkpoints/last.pt
```

系统会自动恢复模型权重、优化器状态、EMA 和训练轮次。
