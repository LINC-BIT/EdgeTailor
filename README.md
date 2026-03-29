# EdgeTailor

本项目实现了一个面向边缘侧动态场景的长尾持续学习与域自适应的图像分类方法 EdgeTailor，并以航拍船舶图像分类任务作为典型应用示例进行验证。
该方法旨在解决图像分类任务在实际部署中普遍存在的类别不均衡、任务连续到达以及跨域分布偏移等问题。

在航拍船舶场景中，不同类别船舶样本分布不均，不同海域与成像条件带来明显的域偏移，同时数据以流式方式不断到达，这些特性为验证所提方法的有效性提供了具有代表性的应用背景。

整体框架包括两个核心部分：

- **EdgeTailorCL**：用于提升长尾持续学习性能，缓解灾难性遗忘问题  
- **EdgeTailorDA**：用于提升模型在跨域场景下的泛化能力  

---

# 1. 环境配置

```bash
conda create -n edgetailor python=3.8
conda activate edgetailor

pip install -r requirements.txt
```

# 2. 数据集
为全面验证方法性能，本文选取多个公开航拍船舶数据集构建统一实验数据，包括：

- [FGSC-23](https://www.cjig.cn/en/article/doi/10.11834/jig.200261/)：细粒度船舶分类数据集，包含23类船舶目标。该数据集类别间外观相似度高，对模型的细粒度判别能力提出较高要求。

- [FGSR/FGSCR](https://www.sciencedirect.com/org/science/article/pii/S1546221823006203)：与 FGSC-23 类似，同样面向细粒度船舶识别任务，进一步增加了类别复杂度和样本多样性，适用于评估模型在细粒度长尾场景下的性能。

- [ShipRSImageNet](https://www.mdpi.com/1424-8220/22/9/3243)：原始为目标检测数据集。本文通过裁剪标注框区域提取船舶目标，并转换为图像分类数据，用于增强数据规模与类别覆盖范围。

- DSCR：包含多场景、多背景下的航拍船舶图像，具有较强的环境变化特性，有助于评估模型的跨域泛化能力。求。

在数据构建过程中，本文保留各数据集原始类别分布，使整体数据呈现自然长尾特性，从而更贴近真实应用场景中“头类丰富、尾类稀缺”的分布规律。


# 3. 模型

本项目主要支持以下两类模型：

- [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)：该模型由多个卷积层和池化层组成，用于提取图像中的信息。通常，当网络深度较大时，ResNet 容易出现梯度消失（爆炸）现象，性能下降。因此，ResNet 添加了 BatchNorm 来缓解梯度消失（爆炸），并添加了残差连接来缓解性能下降。

- [Vit](https://arxiv.org/abs/2010.11929)：Vision Transformer（ViT）将 Transformer 架构应用于图像识别任务。它将图像分割成多个块，然后将这些小块作为序列数据输入到 Transformer 模型中，利用自注意力机制捕捉图像中的全局和局部信息，从而实现高效的图像分类。目前系统支持了TinyPiT以及TinyViT两个网路。



