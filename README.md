# EATNet

## 环境安装

```bash
pip install -r requirement.txt
```

注意事项见 [requirement.txt](requirement.txt)。要点：

- `torch` / `torchvision` 的 CUDA 构建需自行按驱动版本选择（验证过的组合：`torch 2.12.0+cu130` + `torchvision 0.27.0`）。
- 评估所用的指标库在 PyPI 上的发行名是 **`pysodmetrics`**，而导入名是 **`py_sod_metrics`**，两者不一致。
- 骨干网络的预训练权重放在 [models/pvt_v2_b2.pth](models/)，[net/pvtv2.py](net/pvtv2.py) 会**硬编码**从该路径加载，缺失则无法构建模型。

## 数据准备

在项目根目录下按以下结构放置数据：

```
data/
├── TrainDataset/
│   ├── Imgs/          # 训练图像，*.jpg
│   ├── GT/            # 真值掩码，*.png
│   └── Edge/          # 边缘真值，*.png
└── TestDataset/
    ├── CAMO/{Imgs,GT,Edge}/
    ├── CHAMELEON/{Imgs,GT,Edge}/
    ├── COD10K/{Imgs,GT,Edge}/
    └── NC4K/{Imgs,GT}/          # NC4K 无 Edge，推理不需要
```

> **两个硬性约束**（见 [utils/tdataloader.py](utils/tdataloader.py)）：
>
> 1. 图像文件名**必须**以 `.jpg` 结尾、掩码与边缘**必须**以 `.png` 结尾，加载器是按后缀过滤的，后缀不符的文件会被静默忽略。
> 2. 所有脚本使用相对路径，**必须在项目根目录下运行**。

## 训练

```bash
python etrain.py
```

常用参数：

| 参数           | 默认值              | 说明                                   |
| -------------- | ------------------- | -------------------------------------- |
| `--epoch`      | 40                  | 训练轮数                               |
| `--lr`         | 1e-4                | 初始学习率（Adam + poly 衰减）         |
| `--trainsize`  | 416                 | 输入分辨率                             |
| `--clip`       | 0.5                 | 梯度裁剪阈值                           |
| `--train_save` | EATNet1             | 快照保存到 `checkpoints/<train_save>/` |
| `--train_path` | ./data/TrainDataset | 训练集路径                             |
| `--batchsize`  | 16                  | **目前不生效**，见「已知问题」         |

每 5 个 epoch 存一次快照，命名为 `EATNet-<epoch>.pth`（约 154MB/个）。训练日志同时打印到终端并追加写入 [log/EATNet.txt](log/EATNet.txt)。

损失为三路侧输出的加权 BCE + 加权 IoU，加 3 倍权重的边缘 dice 损失（[etrain.py:64](etrain.py#L64)）。

## 推理

```bash
python etest.py --pth_path ./checkpoints/best.pth
```

依次在 CAMO / CHAMELEON / COD10K / NC4K 上推理，结果写入 `results/EATNet/<数据集>/`。`--testsize` 默认 416。

## 评估

```bash
python eval.py --method EATNet --datasets CAMO CHAMELEON COD10K NC4K
```

`--method` 需与推理输出目录名一致。指标追加写入 [evalresults.txt](evalresults.txt)。
