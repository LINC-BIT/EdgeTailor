from benchmark.data.datasets.data_aug import cifar_like_image_train_aug, cifar_like_image_test_aug
from benchmark.data.datasets.ab_dataset import ABDataset
from benchmark.data.datasets.dataset_split import train_val_split
from benchmark.data.datasets.registery import dataset_register

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from typing import Dict, List, Optional
import numpy as np


@dataset_register(
    name='ShipRSImageNet',
    classes=None,
    task_type='Image Classification',
    object_type='Ship',
    class_aliases=[],
    shift_type=None
)
class ShipRSImageNet(ABDataset):

    def create_dataset(
        self,
        root_dir: str,
        split: str,
        transform: Optional[Compose],
        classes: List[str],
        ignore_classes: List[str],
        idx_map: Optional[Dict[int, int]]
    ):
        # 数据增强
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
            self.transform = transform

        # 读取裁剪后的分类数据
        dataset = ImageFolder(root=root_dir, transform=transform)
        self.dataset1 = dataset

        # 自动获取类别（推荐）
        if classes is None or len(classes) == 0:
            classes = dataset.classes

        dataset.targets = np.array(dataset.targets)

        # 忽略类别（CL常用）
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                cls_idx = classes.index(ignore_class)

                dataset.samples = [
                    s for s, t in zip(dataset.samples, dataset.targets) if t != cls_idx
                ]
                dataset.targets = dataset.targets[dataset.targets != cls_idx]

        # 类别映射（task incremental）
        if idx_map is not None:
            for i, t in enumerate(dataset.targets):
                dataset.targets[i] = idx_map[t]

        # train / val 切分
        if split != 'test':
            dataset = train_val_split(dataset, split)

        return dataset

    def get_clsnum(self):
        targets = np.array(self.dataset1.targets)
        cls_num = []
        for i in range(len(set(targets))):
            cls_num.append(np.sum(targets == i))
        return cls_num