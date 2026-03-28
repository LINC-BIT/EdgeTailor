import enum
from typing import Dict, List, Tuple
import numpy as np
import copy
from ..data.datasets.ab_dataset import ABDataset
from ..data.dataloader import FastDataLoader, InfiniteDataLoader
import pandas as pd


# 生成一个 0 到 1 之间的随机数


class DatasetMetaInfo:
    def __init__(self, name, 
                 known_classes_name_idx_map, unknown_class_idx):
        
        assert unknown_class_idx not in known_classes_name_idx_map.keys()
        
        self.name = name
        self.unknown_class_idx = unknown_class_idx
        self.known_classes_name_idx_map = known_classes_name_idx_map
        
    @property
    def num_classes(self):
        return len(self.known_classes_idx) + 1
        
        
class MergedDataset:
    def __init__(self, datasets: List[ABDataset]):
        self.datasets = datasets
        self.datasets_len = [len(i) for i in self.datasets]
        self.datasets_cum_len = np.cumsum(self.datasets_len)

    def __getitem__(self, idx):
        for i, cum_len in enumerate(self.datasets_cum_len):
            if idx < cum_len:
                #print(self.datasets[i][idx - sum(self.datasets_len[0: i])])
                return self.datasets[i][idx - sum(self.datasets_len[0: i])]
            
    def __len__(self):
        return sum(self.datasets_len)
    
    
class IndexReturnedDataset:
    def __init__(self, dataset: ABDataset):
        self.dataset = dataset
        
    def __getitem__(self, idx):
        res = self.dataset[idx]

        if isinstance(res, (tuple, list)):
            return (*res, idx)
        else:
            return res, idx
            
    def __len__(self):
        return len(self.dataset)
    

class Scenario:
    def __init__(self, config,
                 source_datasets_meta_info: Dict[str, DatasetMetaInfo], target_datasets_meta_info: Dict[str, DatasetMetaInfo],
                 target_source_map: Dict[str, Dict[str, str]], 
                 target_domains_order: List[str],
                 source_datasets: Dict[str, Dict[str, ABDataset]], target_datasets: Dict[str, Dict[str, ABDataset]]):
        
        self.__config = config
        self.__source_datasets_meta_info = source_datasets_meta_info
        self.__target_datasets_meta_info = target_datasets_meta_info
        self.__target_source_map = target_source_map
        self.__target_domains_order = target_domains_order
        self.__source_datasets = source_datasets
        self.__target_datasets = target_datasets
    
    # 1. basic
    def get_config(self):
        return copy.deepcopy(self.__config)
    
    def get_task_type(self):
        return list(self.__source_datasets.values())[0]['train'].task_type
    
    def get_num_classes(self):
        known_classes_idx = []
        unknown_classes_idx = []
        #print(self.__source_datasets_meta_info)
        #print('--------------------------------------')
        for v in self.__source_datasets_meta_info.values():
            known_classes_idx += list(v.known_classes_name_idx_map.values())
            unknown_classes_idx += [v.unknown_class_idx]
        #print(self.__target_datasets_meta_info[0])
        #print('--------------------------------------')
        #print(self.__target_datasets_meta_info)
        #print(type(self.__target_datasets_meta_info))
        for v in self.__target_datasets_meta_info.values():
            known_classes_idx += list(v.known_classes_name_idx_map.values())
            unknown_classes_idx += [v.unknown_class_idx]
        unknown_classes_idx = [i for i in unknown_classes_idx if i is not None]
        # print(known_classes_idx, unknown_classes_idx)
        res = len(set(known_classes_idx)), len(set(unknown_classes_idx)), len(set(known_classes_idx + unknown_classes_idx))
        # print(res)
        assert res[0] + res[1] == res[2]
        return res

    def build_dataloader(self, dataset: ABDataset, batch_size: int, num_workers: int, infinite: bool, shuffle_when_finite: bool,weight=None):
        #print(list(dataset))
        import torch

        if infinite:
            dataloader = InfiniteDataLoader(
                dataset, weight, batch_size, num_workers=0)
        else:
            dataloader = FastDataLoader(
                dataset, batch_size, 0, shuffle=shuffle_when_finite)

        return dataloader

    def get_clsnum(self,dataset: ABDataset):
        return dataset.get_clsnum()

    def build_sub_dataset(self, dataset: ABDataset, indexes: List[int]):
        from ..data.datasets.dataset_split import _SplitDataset
        dataset.dataset = _SplitDataset(dataset.dataset, indexes)
        return dataset
    
    def build_index_returned_dataset(self, dataset: ABDataset):
        return IndexReturnedDataset(dataset)
        
    # 2. source
    def get_source_datasets_meta_info(self):
        return self.__source_datasets_meta_info
    
    def get_source_datasets_name(self):
        return list(self.__source_datasets.keys())
    
    def get_merged_source_dataset(self, split):
        source_train_datasets = {n: d[split] for n, d in self.__source_datasets.items()}

        return MergedDataset(list(source_train_datasets.values()))

    def get_offline_source_merged_dataset(self, split):
        source_train_datasets = {n: d[split] for n, d in self.__source_datasets.items()}
        #print (source_train_datasets.values(),'_________________________')
        return MergedDataset(list(source_train_datasets.values()))
    
    def get_source_datasets(self, split):
        source_train_datasets = {n: d[split] for n, d in self.__source_datasets.items()}
        return source_train_datasets

    def get_offline_source_datasets(self, split):
        source_train_datasets = {n: d[split] for n, d in self.__source_datasets.items()}
        return source_train_datasets
    
    # 3. target **domain**
    # (do we need such API `get_ith_target_domain()`?)
    def get_target_domains_meta_info(self):
        return self.__source_datasets_meta_info
    
    def get_target_domains_order(self):
        return self.__target_domains_order
    
    def get_corr_source_datasets_name_of_target_domain(self, target_domain_name):
        return self.__target_source_map[target_domain_name]
    
    def get_limited_target_train_dataset(self):
        if len(self.__target_domains_order) > 1:
            raise RuntimeError('this API is only for pass-in scenario in user-defined online DA algorithm')

        return list(self.__target_datasets.values())[0]['train']

    def get_domain_index(self):
        return list(self.__target_datasets.keys())[0]

    def get_target_datasets(self,split):
        target_train_datasets = {n: d[split] for n, d in self.__target_datasets.items()}
        return target_train_datasets
    
    def get_target_domains_iterator(self, split):
        for target_domain_index, target_domain_name in enumerate(self.__target_domains_order):
            In_learn=0
            if In_learn and split=='test':
                current_target_dataset = self.__target_datasets[target_domain_name][split]

                # 获取所有小于当前索引的目标域数据集
                previous_target_datasets = [
                    self.__target_datasets[self.__target_domains_order[i]][split]
                    for i in range(target_domain_index)
                ]
                print('target_domain_index:',target_domain_index)
                # 源域数据集
                source_train_datasets = {n: d[split] for n, d in self.__source_datasets.items()}

                # 合并数据集
                target_dataset = [current_target_dataset] +  previous_target_datasets +list(source_train_datasets.values())
            else:
                target_dataset = [self.__target_datasets[target_domain_name][split]]
            #print('target_dataset:',type(target_dataset))

            #print(self.__target_datasets_meta_info,type(self.__target_datasets),target_domain_name)
            #xx=0 if target_domain_name=='SLT10'
            target_domain_meta_info = self.__target_datasets_meta_info[target_domain_name]
            
            yield target_domain_index, target_domain_name, target_dataset, target_domain_meta_info
    
    # 4. permission management
    def get_sub_scenario(self, source_datasets_name, source_splits, target_domains_order, target_splits):
        def get_split(dataset, splits):
            res = {}
            for s, d in dataset.items():
                if s in splits:
                    res[s] = d
            return res
        
        return Scenario(
            config=self.__config,
            source_datasets_meta_info={k: v for k, v in self.__source_datasets_meta_info.items() if k in source_datasets_name},
            target_datasets_meta_info={k: v for k, v in self.__target_datasets_meta_info.items() if k in target_domains_order},
            target_source_map={k: v for k, v in self.__target_source_map.items() if k in target_domains_order},
            target_domains_order=target_domains_order,
            source_datasets={k: get_split(v, source_splits) for k, v in self.__source_datasets.items() if k in source_datasets_name},
            target_datasets={k: get_split(v, target_splits) for k, v in self.__target_datasets.items() if k in target_domains_order}
        )
    
    def get_only_source_sub_scenario_for_exp_tracker(self):
        return self.get_sub_scenario(self.get_source_datasets_name(), ['train', 'val', 'test'], [], [])
    
    def get_only_source_sub_scenario_for_alg(self):
        return self.get_sub_scenario(self.get_source_datasets_name(), ['train'], [], [])
    
    def get_one_da_sub_scenario_for_alg(self, target_domain_name):
        return self.get_sub_scenario(self.get_corr_source_datasets_name_of_target_domain(target_domain_name), 
                                     ['train', 'val'], [target_domain_name], ['train'])

    def set_permission(self, stage, user):
        self.__stage = stage
        self.__user = user

# class Scenario:
#     def __init__(self, config,
                 
#                  offline_source_datasets_meta_info: Dict[str, DatasetMetaInfo], 
#                  offline_source_datasets: Dict[str, ABDataset],
                 
#                  online_datasets_meta_info: List[Tuple[Dict[str, DatasetMetaInfo], DatasetMetaInfo]],
#                  online_datasets: Dict[str, ABDataset],
#                  target_domains_order: List[str],
#                  target_source_map: Dict[str, Dict[str, str]]):
        
#         self.config = config
        
#         self.offline_source_datasets_meta_info = offline_source_datasets_meta_info
#         self.offline_source_datasets = offline_source_datasets
        
#         self.online_datasets_meta_info = online_datasets_meta_info
#         self.online_datasets = online_datasets
        
#         self.target_domains_order = target_domains_order
#         self.target_source_map = target_source_map
        
#         self.__stage = 'offline' # 'online'
#         self.__user = 'internal' # 'user'
        
#     def set_permission(self, stage, user):
#         self.__stage = stage
#         self.__user = user
        
#     def verify_permission(self, split, permission: Dict[str, Tuple[List[str], List[str]]]):
#         if split not in permission.keys():
#             raise PermissionError(f'There is no split "{split}" in {self.__stage} stage!')
#         if self.__stage not in permission[split][0] or self.__user not in permission[split][1]:
#             raise PermissionError(f'User {self.__user} is not allowed to use this data in {self.__stage} stage!')
        
#     def get_offline_source_datasets(self, split):
#         self.verify_permission(split, {
#             'train': (['offline'], ['internal', 'user']),
#             'val': (['offline'], ['internal', 'user']),
#             'test': (['offline'], ['internal']),
#         })
#         return {n: d[split] for n, d in self.offline_source_datasets.items()}
    
#     def get_offline_source_merged_dataset(self, split):
#         self.verify_permission(split, {
#             'train': (['offline'], ['internal', 'user']),
#             'val': (['offline'], ['internal', 'user']),
#             'test': (['offline'], ['internal']),
#         })
#         return MergedDataset([d[split] for d in self.offline_source_datasets.values()])
    
#     def get_online_current_corresponding_source_datasets(self, domain_index, split):
#         self.verify_permission(split, {
#             'train': (['online'], ['internal', 'user']),
#             'val': (['online'], ['internal', 'user']),
#             'test': (['online'], ['internal']),
#         })
#         cur_target_domain_name = self.target_domains_order[domain_index]
#         cur_source_datasets_name = list(self.target_source_map[cur_target_domain_name].keys())
#         cur_source_datasets = {n: self.online_datasets[n + '|' + cur_target_domain_name][split] for n in cur_source_datasets_name}
#         return cur_source_datasets
    
#     def get_online_current_corresponding_merged_source_dataset(self, domain_index, split):
#         self.verify_permission(split, {
#             'train': (['online'], ['internal', 'user']),
#             'val': (['online'], ['internal', 'user']),
#             'test': (['online'], ['internal']),
#         })
#         cur_target_domain_name = self.target_domains_order[domain_index]
#         cur_source_datasets_name = list(self.target_source_map[cur_target_domain_name].keys())
#         cur_source_datasets = {n: self.online_datasets[n + '|' + cur_target_domain_name][split] for n in cur_source_datasets_name}
#         return MergedDataset([d for d in cur_source_datasets.values()])
    
#     def get_online_current_target_dataset(self, domain_index, split):
#         self.verify_permission(split, {
#             'train': (['online'], ['internal', 'user']),
#             'test': (['online'], ['internal']),
#         })
#         cur_target_domain_name = self.target_domains_order[domain_index]
#         return self.online_datasets[cur_target_domain_name][split]
    
#     def build_dataloader(self, dataset: ABDataset, batch_size: int, num_workers: int, 
#                          infinite: bool, shuffle_when_finite: bool, to_iterator: bool):
#         if infinite:
#             dataloader = InfiniteDataLoader(
#                 dataset, None, batch_size, num_workers=num_workers)
#         else:
#             dataloader = FastDataLoader(
#                 dataset, batch_size, num_workers, shuffle=shuffle_when_finite)
            
#         if to_iterator:
#             dataloader = iter(dataloader)

#         return dataloader
    
#     def build_sub_dataset(self, dataset: ABDataset, indexes: List[int]):
#         from ..data.datasets.dataset_split import _SplitDataset
#         dataset.dataset = _SplitDataset(dataset.dataset, indexes)
#         return dataset
    
#     def build_index_returned_dataset(self, dataset: ABDataset):
#         return IndexReturnedDataset(dataset)
    
#     def get_config(self):
#         return copy.deepcopy(self.config)
    
#     def get_task_type(self):
#         return list(self.online_datasets.values())[0]['train'].task_type