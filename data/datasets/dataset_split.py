import torch
import os
import numpy as np
from collections import Counter
from benchmark.long import parameter as p

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys, weighted_alpha=1.0):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.weighted_alpha = weighted_alpha

    def __getitem__(self, key):
        #print(self.keys[key])
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)

    @property
    def targets(self):
        # 返回 underlying_dataset 中 key 对应的 targets
        return [self.underlying_dataset.targets[i] for i in self.keys]

    def get_cls_num_list(self):
        """返回每个类别的样本数量"""
        # 获取所有 targets
        targets = self.targets

        # 使用 Counter 统计每个类别的数量
        class_counts = Counter(targets)

        # 创建类别数量列表
        num_classes = max(targets) + 1  # 假设类别是从 0 开始的整数

        cls_num_list = [class_counts[i] for i in range(num_classes)]

        return cls_num_list

    def get_weighted_sampler(self, weighted_alpha=None):
        """返回一个新的 _SplitDataset，经过权重采样"""
        # 使用传入的 weighted_alpha 或类的属性
        alpha = weighted_alpha if weighted_alpha is not None else self.weighted_alpha

        # 获取每个类别的样本数量
        cls_num_list = self.get_cls_num_list()

        # 计算每个类别的权重
        cls_weight = 1.0 / (np.array(cls_num_list) ** alpha)

        # 归一化权重，使其和为类别数量
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)

        # 为每个样本分配对应的权重
        samples_weight = np.array([cls_weight[t] for t in self.targets])

        # 转换为 torch Tensor 并转换为 double 类型
        samples_weight = torch.from_numpy(samples_weight).double()

        return samples_weight

        # 创建 WeightedRandomSampler 来根据权重进行索引采样
        # sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.targets), replacement=True)
        #
        # # 使用 WeightedRandomSampler 获取采样后的索引
        # sampled_indices = list(sampler)
        #
        # # 根据采样的索引，生成一个新的 _SplitDataset
        # new_keys = [self.keys[i] for i in sampled_indices]
        # new_split_dataset = _SplitDataset(self.underlying_dataset, new_keys, alpha)
        #
        # return new_split_dataset


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset)

    cache_p = f'{n}_{seed}'
    cache_p = os.path.join(os.path.expanduser(
        '~'), '.domain_benchmark_split_dataset_cache_' + str(cache_p)+'.pth')
    if os.path.exists(cache_p):
        keys_1, keys_2 = torch.load(cache_p)
    else:
        keys = list(range(len(dataset)))
        np.random.RandomState(seed).shuffle(keys)
        keys_1 = keys[:n]
        keys_2 = keys[n:]
        torch.save((keys_1, keys_2), cache_p)
    
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def train_val_split(dataset, split):
    assert split in ['train', 'val']
    if split == 'train':
        return split_dataset(dataset, int(len(dataset) * 0.8))[0]
    else:
        return split_dataset(dataset, int(len(dataset) * 0.8))[1]

    
def train_val_test_split(dataset, split):
    assert split in ['train', 'val', 'test']

    train_set, test_set = split_dataset(dataset, int(len(dataset) * 0.8))
    train_set, val_set = split_dataset(train_set, int(len(train_set) * 0.8))
    
    return {'train': train_set, 'val': val_set, 'test': test_set}[split]
