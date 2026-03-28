from benchmark.data.datasets.data_aug import cifar_like_image_train_aug, cifar_like_image_test_aug
from benchmark.data.datasets.ab_dataset import ABDataset
from benchmark.data.datasets.dataset_split import train_val_split
from benchmark.data.datasets.registery import dataset_register

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from typing import Dict, List, Optional
import numpy as np


@dataset_register(
    name='FGSC23',
    classes=[
        'Aircraft Carrier', 'Bulk Carrier', 'Cargo', 'Cruise Ship', 'Destroyer',
        'Fishing Boat', 'Frigate', 'Hovercraft', 'Icebreaker', 'Landing Ship',
        'Motorboat', 'Oil Tanker', 'Passenger Ship', 'Research Vessel',
        'Sailboat', 'Submarine', 'Tugboat', 'Warship', 'Yacht',
        'Container Ship', 'RORO', 'Dredger', 'Barge'
    ],
    task_type='Image Classification',
    object_type='Ship',
    class_aliases=[],
    shift_type=None
)
class FGSC23(ABDataset):

    def create_dataset(
        self,
        root_dir: str,
        split: str,
        transform: Optional[Compose],
        classes: List[str],
        ignore_classes: List[str],
        idx_map: Optional[Dict[int, int]]
    ):
        # 默认增强
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
            self.transform = transform

        # ImageFolder 读取
        dataset = ImageFolder(root=root_dir, transform=transform)
        self.dataset1 = dataset

        # 转 numpy 方便处理
        dataset.targets = np.array(dataset.targets)

        # 删除类别
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                cls_idx = classes.index(ignore_class)
                dataset.samples = [
                    s for s, t in zip(dataset.samples, dataset.targets) if t != cls_idx
                ]
                dataset.targets = dataset.targets[dataset.targets != cls_idx]

        # 类别重映射（continual learning 用）
        if idx_map is not None:
            for i, t in enumerate(dataset.targets):
                dataset.targets[i] = idx_map[t]

        # train / val 切分
        if split != 'test':
            dataset = train_val_split(dataset, split)

        return dataset

    def get_clsnum(self):
        # 统计每类数量（用于 long-tail 分析）
        targets = np.array(self.dataset1.targets)
        cls_num = []
        for i in range(len(set(targets))):
            cls_num.append(np.sum(targets == i))
        return cls_num