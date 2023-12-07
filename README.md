# Path Pathology-2021

## 快速开始

### 环境和数据集准备
首先，克隆项目并准备数据集：
```
cd /root/ && git clone --depth 1 https://github.com/Dwl2021/Plant-Pathology-2021.git
cd Plant-Pathology-2021/
unzip data.zip
pip install -r requirements.txt
```

或者，您可以从 [Plant Pathology 2021](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/overview/description) 比赛页面下载数据集。若选择此方式，请注意需要对数据进行重新预处理，并将文件解压到 `/root/plant_pathology` 目录下。

## 代码运行

使用以下命令结构来运行脚本：

```
python main.py --model_name <模型名称> --epochs <训练轮数> --batch_size <批处理大小> --Loss_function <损失函数>
```

### 模型选项 (`--model_name`)

您可以选择以下模型之一进行训练：
- `ResNet50`
- `ResNet101`
- `SEResNet34`
- `SEResNet50`
- `SEResNet101`
- `SG_ViT`
- `Torch_ViT`
- `DeepViT`
- `CrossViT`
- `Deit`
- `ResNeXt`

### 损失函数选项 (`--Loss_function`)

支持的损失函数包括：
- `BCEWithLogitsLoss`
- `FocalLoss`
- `ArcFaceLoss`

### 示例命令

以下命令示例将使用 CrossViT 模型进行100轮训练，批处理大小设置为200，并使用 BCEWithLogitsLoss 损失函数：

```
python main.py --model_name CrossViT --epochs 100 --batch_size 200 --Loss_function BCEWithLogitsLoss
```

> **注意**：如果需要进行数据预处理，请在 `main.py` 文件中取消 `preprocessing()` 函数前的注释，以启用该函数。如果不需要预处理，将使用项目目录下已预处理好的 `data` 文件夹。

