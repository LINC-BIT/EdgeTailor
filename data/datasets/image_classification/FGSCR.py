from benchmark.data.datasets.data_aug import cifar_like_image_train_aug, cifar_like_image_test_aug
from benchmark.data.datasets.ab_dataset import ABDataset
from benchmark.data.datasets.dataset_split import train_val_split
from benchmark.data.datasets.registery import dataset_register

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from typing import Dict, List, Optional
import numpy as np


@dataset_register(
    name='FGSCR',
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
class FGSCR(ABDataset):

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

        # 读取数据（FGSCR = 裁剪后的 ship crops）
        dataset = ImageFolder(root=root_dir, transform=transform)
        self.dataset1 = dataset

        dataset.targets = np.array(dataset.targets)

        # 忽略类别（CL 会用）
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

        # train / val split
        if split != 'test':
            dataset = train_val_split(dataset, split)

        return dataset

    def get_clsnum(self):
        targets = np.array(self.dataset1.targets)
        cls_num = []
        for i in range(len(set(targets))):
            cls_num.append(np.sum(targets == i))
        return cls_num