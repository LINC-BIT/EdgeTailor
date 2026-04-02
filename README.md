# EdgeTailor

面向航拍船舶识别的边缘动态学习系统。该系统将航拍船舶识别任务建模为一种面向边缘环境的动态视觉学习问题。在该系统设定中，模型不仅需要处理持续到达的数据并实现持续学习，以适应任务类别的不断扩展与演化，同时还需应对由拍摄条件变化引起的显著域偏移问题，实现跨场景的稳定泛化能力。具体而言，一方面，系统需在长尾分布数据下进行在线更新，在学习新类别或新任务的同时有效缓解灾难性遗忘问题；另一方面，模型需具备一定的域自适应能力，在不同航拍高度、光照条件及设备差异所带来的分布变化下，仍能够保持对各类船舶目标的可靠识别性能。此外，考虑到边缘设备在计算资源与存储能力上的限制，所设计的方法还需兼顾模型的轻量化与高效性，从而实现准确性、稳定性与实时性的综合平衡。基于上述需求，本系统从长尾学习、持续学习与域泛化三个方面出发，构建了一种适用于复杂动态环境的边缘智能船舶识别方法，为实际航拍应用提供有效支撑。

<img src="https://github.com/LINC-BIT/FCLOnMDefenseData/blob/main/System.png" width="100%">


# 1. 数据集
为全面验证方法性能，本文选取多个公开航拍船舶数据集构建统一实验数据，包括：

- [FGSC-23](https://www.cjig.cn/en/article/doi/10.11834/jig.200261/)：细粒度船舶分类数据集，包含23类船舶目标。该数据集类别间外观相似度高，对模型的细粒度判别能力提出较高要求。

- [FGSR/FGSCR](https://www.sciencedirect.com/org/science/article/pii/S1546221823006203)：与 FGSC-23 类似，同样面向细粒度船舶识别任务，进一步增加了类别复杂度和样本多样性，适用于评估模型在细粒度长尾场景下的性能。

- [ShipRSImageNet](https://www.mdpi.com/1424-8220/22/9/3243)：原始为目标检测数据集。本文通过裁剪标注框区域提取船舶目标，并转换为图像分类数据，用于增强数据规模与类别覆盖范围。

- DSCR：包含多场景、多背景下的航拍船舶图像，具有较强的环境变化特性，有助于评估模型的跨域泛化能力。求。

在数据构建过程中，本文保留各数据集原始类别分布，使整体数据呈现自然长尾特性，从而更贴近真实应用场景中“头类丰富、尾类稀缺”的分布规律。


# 2. 模型

本项目主要支持以下两类模型：

- [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)：该模型由多个卷积层和池化层组成，用于提取图像中的信息。通常，当网络深度较大时，ResNet 容易出现梯度消失（爆炸）现象，性能下降。因此，ResNet 添加了 BatchNorm 来缓解梯度消失（爆炸），并添加了残差连接来缓解性能下降。

- [Vit](https://arxiv.org/abs/2010.11929)：Vision Transformer（ViT）将 Transformer 架构应用于图像识别任务。它将图像分割成多个块，然后将这些小块作为序列数据输入到 Transformer 模型中，利用自注意力机制捕捉图像中的全局和局部信息，从而实现高效的图像分类。目前系统支持了TinyPiT以及TinyViT两个网路。


# 3. 支持的长尾学习算法

目前系统支持多种经典与最新的长尾学习方法，主要包括基于重采样、数据增强的方法：

- [SRR](https://openaccess.thecvf.com/content_ECCV_2018/html/Dhruv_Mahajan_Exploring_the_Limits_ECCV_2018_paper.html)：本文来自 ECCV (2018)。它通过对类别进行重采样与重加权，缓解长尾分布带来的类别不均衡问题。

- [DCL](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Dynamic_Curriculum_Learning_for_Imbalanced_Data_Classification_ICCV_2019_paper.html)：本文来自 ICCV (2019)。它基于动态课程学习策略，逐步调整训练样本分布，从而提升模型对尾类的学习能力。

- [CMO](https://openaccess.thecvf.com/content/CVPR2022/html/Park_The_Majority_Can_Help_the_Minority_Context-Rich_Minority_Oversampling_for_CVPR_2022_paper.html)：本文来自 CVPR (2022)。它通过上下文感知的过采样策略生成少数类样本，从而改善长尾分布。
  
- [Remix](https://link.springer.com/chapter/10.1007/978-3-030-65414-6_9)：本文来自 ECCV (2020)。它基于 Mixup 的重采样策略，对不同类别样本进行重组以平衡数据分布。

- [OTmix](https://proceedings.neurips.cc/paper_files/paper/2023/hash/bdabb5d4262bcfb6a1d529d690a6c82b-Abstract-Conference.html)：本文来自 NIPS (2023)。它利用最优传输引导的样本混合策略，实现更合理的类别间数据重分布。

- [GLMC](https://openaccess.thecvf.com/content/CVPR2023/html/Du_Global_and_Local_Mixture_Consistency_Cumulative_Learning_for_Long-Tailed_Visual_CVPR_2023_paper.html)：本文来自 CVPR (2023)。它通过全局与局部一致性约束进行数据增强与表示优化，从而提升长尾场景下的分类性能。
