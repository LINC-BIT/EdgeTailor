from benchmark.data.datasets.data_aug import tinyimagenet_like_image_train_aug,tinyimagenet_like_image_test_aug
from ..ab_dataset import ABDataset
from ..dataset_split import train_val_split
from torchvision.datasets import ImageFolder
import os
from typing import Dict, List, Optional
from torchvision.transforms import Compose

from ..registery import dataset_register

with open(os.path.join(os.path.dirname(__file__), 'tinyimagenet_classes.txt'), 'r') as f:
    classes = [line.split('\t')[1].strip() for line in f.readlines()]
    assert len(classes) == 200

import torchvision
import numpy as np
import os


class IMBALANETINYIMGNET(torchvision.datasets.ImageFolder):
    cls_num = 200

    def __init__(self, root='data/tiny-imagenet-200', imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None):
        split = 'train'
        if not train:
            split = 'val'

        super(IMBALANETINYIMGNET, self).__init__(root, transform=transform, target_transform=target_transform)
        np.random.seed(rand_number)

        if train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.class_freq = img_num_list
            self.gen_imbalanced_data(img_num_list)
        # else:
        #     img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, 1)
        #     self.class_freq = img_num_list
        #     self.gen_imbalanced_data(img_num_list)

        self.labels = self.targets
        print("{} Mode: Contain {} images".format(split, len(self.samples)))

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.samples) / cls_num
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

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            res_list = [self.samples[i] for i in selec_idx]
            new_data.extend(res_list)
            new_targets.extend([the_class, ] * the_img_num)
        self.samples = new_data
        self.targets = new_targets

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



@dataset_register(
    name='TINYIMAGENETC_snow001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_snow001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'snow')
        dataset = IMBALANETINYIMGNET( root=root_dir, imb_type='exp', imb_factor=0.01,train=True if split == 'train' else False,transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


@dataset_register(
    name='TINYIMAGENETC_gaussiannoise001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_gaussiannoise001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'gaussian_noise')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


@dataset_register(
    name='TINYIMAGENETC_shotnoise001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_shotnoise001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'shot_noise')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


@dataset_register(
    name='TINYIMAGENETC_impulsenoise001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_impulsenoise001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'impulse_noise')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


@dataset_register(
    name='TINYIMAGENETC_defocusblur001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_defocusblur001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'defocus_blur')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


@dataset_register(
    name='TINYIMAGENETC_glassblur001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_glassblur001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'glass_blur')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


@dataset_register(
    name='TINYIMAGENETC_motionblur001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_motionblur001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'motion_blur')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


@dataset_register(
    name='TINYIMAGENETC_zoomblur001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_zoomblur001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'zoom_blur')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset

@dataset_register(
    name='TINYIMAGENETC_frost001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_frost001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'frost')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


@dataset_register(
    name='TINYIMAGENETC_fog001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_fog001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'fog')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


@dataset_register(
    name='TINYIMAGENETC_brightness001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_brightness001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'brightness')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset



@dataset_register(
    name='TINYIMAGENETC_contrast001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_contrast001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'contrast')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset



@dataset_register(
    name='TINYIMAGENETC_elastictransform001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_elastictransform001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'elastic_transform')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


@dataset_register(
    name='TINYIMAGENETC_pixelate001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_pixelate001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'pixelate')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset



@dataset_register(
    name='TINYIMAGENETC_jpegcompression001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_jpegcompression001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'jpeg_compression')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset



@dataset_register(
    name='TINYIMAGENETC_specklenoise001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_specklenoise001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'speckle_noise')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset



@dataset_register(
    name='TINYIMAGENETC_gaussianblur001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_gaussianblur001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'gaussian_blur')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


@dataset_register(
    name='TINYIMAGENETC_spatter001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_spatter001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'spatter')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset



@dataset_register(
    name='TINYIMAGENETC_saturate001',
    classes=classes,
    task_type='Image Classification',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class TINYIMAGENETC_saturate001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = tinyimagenet_like_image_train_aug() if split == 'train' else  tinyimagenet_like_image_test_aug()
            self.transform = transform
        root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        root_dir = os.path.join(root_dir, 'saturate')
        dataset = IMBALANETINYIMGNET(root=root_dir, imb_type='exp', imb_factor=0.01,
                                     train=True if split == 'train' else False, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset