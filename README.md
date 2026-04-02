# EdgeTailor

面向航拍船舶识别任务的边缘侧持续学习与域自适应智能系统。该系统针对航拍场景中长尾分布、任务动态演化与跨域分布偏移等多重挑战，构建了基于“数据—模型—训练”三位一体的一体化系统架构。系统以任务与域为核心组织维度，通过模块化分层解耦与统一控制机制，实现多源航拍数据的规范化组织、模型能力的持续演进以及跨场景环境下的稳定泛化。

系统主要包括数据处理模块、模型与算法模块以及航拍船舶训练模块。数据处理模块通过多数据集整合与标准化处理，构建具有长尾分布与跨域差异的数据体系；模型与算法模块以轻量化模型为基础，集成长尾学习、持续学习与域泛化能力，实现对复杂航拍场景的稳定识别；航拍船舶训练模块则通过配置化控制训练流程，并模拟持续学习与域自适应过程，同时统一采集各任务与目标域下的性能与资源指标。

最终，系统形成“数据组织可控、模型能力可演进、训练过程可管理”的闭环框架，为航拍船舶识别任务在边缘环境中的高效部署与动态优化提供全流程支撑。


# 1. 数据集
为全面验证方法性能，本文选取多个公开航拍船舶数据集构建统一实验数据，包括：

- [FGSC-23](https://www.cjig.cn/en/article/doi/10.11834/jig.200261/)：细粒度船舶分类数据集，包含23类船舶目标。该数据集类别间外观相似度高，对模型的细粒度判别能力提出较高要求。

- [FGSR/FGSCR](https://www.sciencedirect.com/org/science/article/pii/S1546221823006203)：与 FGSC-23 类似，同样面向细粒度船舶识别任务，进一步增加了类别复杂度和样本多样性，适用于评估模型在细粒度长尾场景下的性能。

- [ShipRSImageNet](https://www.mdpi.com/1424-8220/22/9/3243)：原始为目标检测数据集。本文通过裁剪标注框区域提取船舶目标，并转换为图像分类数据，用于增强数据规模与类别覆盖范围。

- DSCR：包含多场景、多背景下的航拍船舶图像，具有较强的环境变化特性，有助于评估模型的跨域泛化能力。

## 1.1 长尾分布模拟

基于设定的类别不均衡类型（如指数型或阶梯型），对每个类别的样本数量进行控制，从而构造符合长尾分布的数据划分：

```python
def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
    img_max = len(self.data) / cls_num
    img_num_per_cls = []

    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))

    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))

    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)

    return img_num_per_cls
```

在获得每个类别的目标样本数量后，根据类别索引对原始数据进行裁剪与重组，从而生成符合长尾分布的新数据集：

```python
def gen_imbalanced_data(self, img_num_per_cls):
    new_data = []
    new_targets = []

    targets_np = np.array(self.targets, dtype=np.int64)
    classes = np.unique(targets_np)

    self.num_per_cls_dict = dict()

    for the_class, the_img_num in zip(classes, img_num_per_cls):
        self.num_per_cls_dict[the_class] = the_img_num

        idx = np.where(targets_np == the_class)[0]
        selec_idx = idx[:the_img_num]

        new_data.append(self.data[selec_idx, ...])
        new_targets.extend([the_class] * the_img_num)

    self.data = np.vstack(new_data)
    self.targets = new_targets
```

## 1.2 域划分

为构建具有显著域偏移特性的训练数据，本系统结合图像平均亮度、对比度以及高亮像素比例等多维特征，对数据进行强光与弱光划分。其中，强光样本通常具有较高的对比度及明显的局部高亮区域，而弱光样本则表现为整体亮度较低且对比度较弱。

```python
def compute_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()          
    contrast = gray.std()              
    highlight_ratio = np.mean(gray > 200) 

    return brightness, contrast, highlight_ratio

def classify_domain(brightness, contrast, highlight_ratio):

    if highlight_ratio > 0.05 and contrast > 40:
        return "bright"
    
    if brightness < 90 and contrast < 35:
        return "dark"

    return "dark"
```



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
