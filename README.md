# Path Pathology-2021

## Quick Start

### 环境和数据集准备
```
git clone --depth 1 https://github.com/Dwl2021/Plant-Pathology-2021.git
cd Plant-Pathology-2021/
unzip data.zip
pip install -r requirements.txt
```

也可以下载比赛的 [Plant Pathology 2021](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/overview/description)，但是这一步要重新数据预处理，因此要将文件解压为`/root/plant_pathology`。



## 代码运行

运行主程序main.py，并且之后的参数依次是，模型名，epochs，batchsize。

```
python main.py ResNet50 100 50
```

模型可选：[ResNet50, ResNet101, SEResNet34, SEResNet50, SEResNet101, SG_ViT, Torch_ViT, DeepViT, CrossViT, Deit, ResNeXt]，注意在输入的时候不要带有引号。

> 如果需要预处理：则把main.py文件中的preprocessing()函数的前面的'#'去掉，让其可以运行，否则就默认运行该目录下自带的已经预处理好的data文件夹。

