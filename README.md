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


# 4. 支持的长尾学习算法

目前系统支持多种经典与最新的长尾学习方法，主要包括基于重采样、数据增强的方法：

- [SRR](https://openaccess.thecvf.com/content_ECCV_2018/html/Dhruv_Mahajan_Exploring_the_Limits_ECCV_2018_paper.html)：本文来自 ECCV (2018)。它通过对类别进行重采样与重加权，缓解长尾分布带来的类别不均衡问题。

- [DCL](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Dynamic_Curriculum_Learning_for_Imbalanced_Data_Classification_ICCV_2019_paper.html)：本文来自 ICCV (2019)。它基于动态课程学习策略，逐步调整训练样本分布，从而提升模型对尾类的学习能力。

- [CMO](https://openaccess.thecvf.com/content/CVPR2022/html/Park_The_Majority_Can_Help_the_Minority_Context-Rich_Minority_Oversampling_for_CVPR_2022_paper.html)：本文来自 CVPR (2022)。它通过上下文感知的过采样策略生成少数类样本，从而改善长尾分布。
  
- [Remix](https://link.springer.com/chapter/10.1007/978-3-030-65414-6_9)：本文来自 ECCV (2020)。它基于 Mixup 的重采样策略，对不同类别样本进行重组以平衡数据分布。

- [OTmix](https://proceedings.neurips.cc/paper_files/paper/2023/hash/bdabb5d4262bcfb6a1d529d690a6c82b-Abstract-Conference.html)：本文来自 NIPS (2023)。它利用最优传输引导的样本混合策略，实现更合理的类别间数据重分布。

- [GLMC](https://openaccess.thecvf.com/content/CVPR2023/html/Du_Global_and_Local_Mixture_Consistency_Cumulative_Learning_for_Long-Tailed_Visual_CVPR_2023_paper.html)：本文来自 CVPR (2023)。它通过全局与局部一致性约束进行数据增强与表示优化，从而提升长尾场景下的分类性能。
