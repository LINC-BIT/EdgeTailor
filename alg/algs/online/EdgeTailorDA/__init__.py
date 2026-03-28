from ....ab_algorithm import ABOnlineDAAlgorithm
from .....scenario.scenario import Scenario
from ....registery import algorithm_register
from .....exp.exp_tracker import OnlineDATracker
from .....exp.alg_model_manager import ABAlgModelsManager
from typing import Dict, List
from sklearn.metrics import confusion_matrix

from schema import Schema
import torch
import torch.optim
from torch.utils.data import TensorDataset
import copy
from .util import mmd_rbf
#from benchmark.longtail import *
import torch.nn.functional as F
import torch
edgetailor=0

from torch import nn
#lws_model = LearnableWeightScaling(num_classes=no_of_classes)

global confusion_cf
d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
import ot
from imblearn.over_sampling import BorderlineSMOTE


#import benchmark.long.parameter as p
from benchmark.long.SquareRoot import square_root_biased_sampling
from benchmark.long.Progressivelybalanced import *
from benchmark.long.remix import *
from benchmark.long.GLMC import GLMC_mixed,SimSiamLoss
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

import time

@algorithm_register(
    name='CUA',
    stage='online',
    supported_tasks_type=['Image Classification']
)
class CUA(ABOnlineDAAlgorithm):
    def __init__(self, models, alg_models_manager: ABAlgModelsManager, hparams: Dict, device, random_seed, res_save_dir):
        super().__init__(models, alg_models_manager, hparams, device, random_seed, res_save_dir)
        
        self.replay_train_set = None
        self.all_replay_x, self.all_replay_y = None, None
        self.met_source_datasets_name = []

        self.raw_ft = copy.deepcopy(self.alg_models_manager.get_model(self.models, 'feature extractor'))
        self.raw_ft.eval()


        
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert all([n in models.keys() for n in ['feature extractor', 'classifier']])
        assert all([isinstance(models[n], tuple) and len(models[n]) > 0 for n in models.keys()]), 'pass the path of model file in'
        
        for func in [
            'get_pseudo_y_from_model_output(self, model_output: torch.Tensor) -> torch.Tensor', 
            'get_feature(self, models, x) -> torch.Tensor'
        ]:
            assert hasattr(self.alg_models_manager, func.split('(')[0]), \
                f'you should implement `{func}` function in the alg_models_manager.'
        
        Schema({
            'batch_size': int,
            'num_workers': int,
            'num_iters': int,
            'optimizer': str,
            'optimizer_args': dict,
            'num_replay_samples_each_domain': int,
            'replay_loss_alpha': float,
            'domain':list,
            'Resampling': list,
            'Augmentation': list,
            'oversample_factor': list,
            'undersample_factor': list,
            'weighted_alpha_r': list,
            'weighted_alpha_a': list,
            'mixup_prob': list,
            'tau': list,
            'kappa': list,
            'beta': float,
        }).validate(hparams)

    def adapt_to_target_domain(self, scenario: Scenario, exp_tracker: OnlineDATracker):
        ft = self.alg_models_manager.get_model(self.models, 'feature extractor')
        self.optimizer = torch.optim.__dict__[self.hparams['optimizer']](
            ft.parameters(), **self.hparams['optimizer_args'])
        
        target_train_set = scenario.get_limited_target_train_dataset()

        target_train_loader = iter(scenario.build_dataloader(target_train_set, self.hparams['batch_size'], self.hparams['num_workers'], True, False))

        source_train_set = scenario.get_source_datasets('train')

        source_train_loaders = [iter(scenario.build_dataloader(d, self.hparams['batch_size'], self.hparams['num_workers'], True, False))
                                for n, d in source_train_set.items()]

        target_domain= scenario.get_domain_index()
        if target_domain in self.hparams['domain'] :
            domain_index = self.hparams['domain'].index(target_domain)

        else:
            print(f"'{target_domain}' 不在列表中")

        print(target_domain)

        #source_train_loaders1= [iter(scenario.build_dataloader(d, self.hparams['batch_size'], self.hparams['num_workers'], True, False)) for n, d in source_train_set.items()]
        # cls_num_list = [scenario.get_clsnum(d) for n, d in source_train_set.items()]
        # cls_num_list = cls_num_list[0]


        if self.hparams['Resampling'][domain_index] == 1:
            #balanced_trainset = square_root_biased_sampling(train_sets, p.oversample_factor, p.undersample_factor)
            # source_train_set1 = {n: square_root_biased_sampling(d.dataset, self.hparams['oversample_factor'], self.hparams['undersample_factor']) for n, d in  source_train_set.items()}
            # #train_loaders = DataLoader(balanced_trainset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=2)
            # source_train_loaders = [
            #     iter(scenario.build_dataloader(d, self.hparams['batch_size'], self.hparams['num_workers'], True, False))
            #     for n, d in source_train_set1.items()]
            # source_train_set=source_train_set1
            # resampled_class_counts = {n: np.bincount(np.array(d.dataset.targets)[source_train_set[n].indices]) for n, d in
            #  source_train_set.items()}
            # print(f"Resampled class distribution: {resampled_class_counts}")
            cls_num_list_t = target_train_set.dataset.get_cls_num_list()
            print("cls_num_list_t0:", cls_num_list_t)
            target_train_set1 = square_root_biased_sampling(target_train_set.dataset, self.hparams['oversample_factor'][domain_index], self.hparams['undersample_factor'][domain_index])
            target_train_loader = iter(scenario.build_dataloader(target_train_set1, self.hparams['batch_size'], self.hparams['num_workers'],
                                          True, False))
            resampled_class_counts = np.bincount(np.array(target_train_set.dataset.targets)[target_train_set1.indices])
            print(f"Resampled class distribution: {resampled_class_counts}")
            target_train_set=target_train_set1

        if self.hparams['Resampling'][domain_index] == 2:
            #unique_classes, class_counts = np.unique(train_sets.dataset.targets, return_counts=True)
            #unique_classes= {n: np.unique(d.dataset.targets, return_counts=True)[0] for n, d in train_sets.items()}
            # class_counts = {n: np.unique(d.dataset.targets, return_counts=True)[1] for n, d in source_train_set.items()}
            # initial_weights ={n: 1.0 / (d ** self.hparams['weighted_alpha']) for n, d in class_counts.items()}
            # #initial_weights = 1.0 / (class_counts ** p.weighted_alpha)
            # class_weights = initial_weights.copy()
            # #balanced_trainset, sampled_indices = progressively_balanced_sampling(train_sets.dataset, class_weights)
            # source_train_set={n: progressively_balanced_sampling(d.dataset, class_weights[n])[0]  for n, d in source_train_set.items()}
            # #train_loaders = DataLoader(balanced_trainset, batch_size= self.hparams['batch_size'], shuffle=True, num_workers=self.hparams['num_workers'])
            # source_train_loaders = [ iter(scenario.build_dataloader(d, self.hparams['batch_size'],
            #                                                    self.hparams['num_workers'], True, False)) for n, d in
            #                  source_train_set.items()]
            #class_counts = get_class_counts(train_sets, sampled_indices)
            #print("Class sample counts in balanced_trainset:", class_counts)

            class_counts = np.unique(target_train_set.dataset.targets, return_counts=True)[1]
            initial_weights = 1.0 / (class_counts ** self.hparams['weighted_alpha_r'][domain_index])
            # initial_weights = 1.0 / (class_counts ** p.weighted_alpha)
            class_weights = initial_weights.copy()
            # balanced_trainset, sampled_indices = progressively_balanced_sampling(train_sets.dataset, class_weights)
            target_train_set =  progressively_balanced_sampling(target_train_set.dataset, class_weights)[0]
            # train_loaders = DataLoader(balanced_trainset, batch_size= self.hparams['batch_size'], shuffle=True, num_workers=self.hparams['num_workers'])
            target_train_loader = iter(scenario.build_dataloader(target_train_set, self.hparams['batch_size'], self.hparams['num_workers'],
                                          True, False))
            # class_counts = get_class_counts(train_sets, sampled_indices)
            #print("Class sample counts in balanced_trainset:", class_counts)
        if (self.hparams['Augmentation'][domain_index] == 1 or self.hparams['Augmentation'][domain_index] == 3 or self.hparams['Augmentation'][domain_index] == 4):
            # cls_num_list={n: d.dataset.get_cls_num_list()  for n, d in source_train_set.items()}
            # #weighted_sampler = train_sets.get_weighted_sampler()
            # weighted_sampler ={n: d.dataset.get_weighted_sampler(weighted_alpha=self.hparams['weighted_alpha'])  for n, d in source_train_set.items()}
            # #print(weighted_sampler)
            # # weighted_train_loader = torch.utils.data.DataLoader(
            # #     train_sets, batch_size=self.hparams['batch_size'],
            # #     num_workers=0, pin_memory=True, sampler=weighted_sampler)
            # weighted_train_loader = [iter(scenario.build_dataloader(d, self.hparams['batch_size'],
            #                                                    self.hparams['num_workers'], True, True,weight=weighted_sampler[n])) for n, d in
            #                  source_train_set.items()]
            # inverse_iter = weighted_train_loader

            cls_num_list_t = target_train_set.dataset.get_cls_num_list()
            # weighted_sampler = train_sets.get_weighted_sampler()
            weighted_sampler_t =  target_train_set.dataset.get_weighted_sampler(weighted_alpha=self.hparams['weighted_alpha_a'][domain_index])

            weighted_train_loader_t = iter(scenario.build_dataloader(target_train_set, self.hparams['batch_size'],
                                                                       self.hparams['num_workers'], True, True,
                                                                       weight=weighted_sampler_t))
            inverse_iter_t = iter(weighted_train_loader_t)
        if self.hparams['Augmentation'][domain_index] == 2:
            # cls_num_list = {n: d.dataset.get_cls_num_list() for n, d in source_train_set.items()}
            # source_train_set = {n: RemixDataset(d.dataset, cls_num_list[n], beta=1.0,tau=self.hparams['tau'], kappa=self.hparams['kappa'])  for n, d in source_train_set.items()}
            #
            # # Create DataLoader
            # #trainloader = DataLoader(remix_trainset, batch_size=p.BATCH_SIZE, shuffle=True, num_workers=2)
            # source_train_loaders = [iter(scenario.build_dataloader(d, self.hparams['batch_size'],
            #                                    self.hparams['num_workers'], True, True)) for n, d in
            #  source_train_set.items()]

            cls_num_list_t =target_train_set.dataset.get_cls_num_list()
            if 200 - len(cls_num_list_t) > 0:
                # 使用 ones 扩展到目标长度
                cls_num_list_t.extend([1] * (200 - len(cls_num_list_t)))
            target_train_set = RemixDataset(target_train_set.dataset, cls_num_list_t, beta=1.0, tau=self.hparams['tau'][domain_index],
                                                kappa=self.hparams['kappa'][domain_index])

            # Create DataLoader
            # trainloader = DataLoader(remix_trainset, batch_size=p.BATCH_SIZE, shuffle=True, num_workers=2)
            target_train_loader = iter(scenario.build_dataloader(target_train_set, self.hparams['batch_size'],
                                                                   self.hparams['num_workers'], True, False))

        cls_num_list_t = target_train_set.dataset.get_cls_num_list()
        if 200- len(cls_num_list_t)> 0:
            # 使用 ones 扩展到目标长度
            cls_num_list_t.extend([0] * (200- len(cls_num_list_t)))


        print("cls_num_list_t:",cls_num_list_t)


        # trainloader = sampler1(trainloader,testloader)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,num_workers=2)





        # if edgetailor:
        #     bufferS = SampleBuffer(buffer_size=4)
        #     bufferT = SampleBuffer(buffer_size=4)

        if self.hparams['replay_loss_alpha'] != 0 and self.all_replay_x is None:
            loaders = {n: iter(scenario.build_dataloader(d, self.hparams['num_replay_samples_each_domain'], self.hparams['num_workers'], True, True))
                                for n, d in source_train_set.items()}

            for dataset_name, loader in loaders.items():
                if dataset_name in self.met_source_datasets_name:
                    continue
                
                cur_replay_x, cur_replay_y = next(loader)[0:2]

                if self.all_replay_x is None:
                    self.all_replay_x, self.all_replay_y = cur_replay_x, cur_replay_y
                else:
                    self.all_replay_x, self.all_replay_y = torch.cat(
                        [self.all_replay_x, cur_replay_x]), torch.cat([self.all_replay_y, cur_replay_y])
                self.met_source_datasets_name += [dataset_name]
            self.replay_train_set = TensorDataset(self.all_replay_x.cpu(), self.all_replay_y.cpu())
        
        if self.replay_train_set is not None:
            replay_train_loader = iter(scenario.build_dataloader(self.replay_train_set, 
                                                                min(len(self.replay_train_set), self.hparams['batch_size']), 
                                                                self.hparams['num_workers'], True, False))




        from example.exp.online.cua import logitadjust

            #print(logit_adjustments.size())
        # cls_num_list = {n: d.dataset.get_cls_num_list() for n, d in source_train_set.items()}
        # start_time = time.time()
        # if self.hparams['Augmentation'] == 3:
        #     cls_num = 10
        #     confusion_cf = np.zeros((cls_num, cls_num), dtype=float)
        #     cf = confusion_cf
        #     cf = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
        #     for l1 in range(10):
        #         cf[l1][l1] = 0

        for iter_index in range(self.hparams['num_iters']):
            # cua_time = time.time()
            # if cua_time-start_time>75:
            #     break
            #print(iter_index,self.hparams['num_iters'])
            #print(iter_index % len(source_train_loaders))
            source_x, source_y = next(source_train_loaders[iter_index % len(source_train_loaders)])
            target_x ,target_y = next(target_train_loader)
            #target_x= torch.tensor([item.cpu().detach().numpy() for item in item_[0:1]]).cuda().squeeze(0)
            #target_x = torch.cat([item for item in item_[0:1]], dim=0).cuda().squeeze(0)
            #target_y=torch.tensor([item.cpu().detach().numpy() for item in item_[1:2]]).cuda().squeeze(0)
            #target_y = torch.cat([item for item in item_[1:2]], dim=0).cuda().squeeze(0)
            source_x, source_y, target_x = source_x.to(self.device), source_y.to(self.device), target_x.to(self.device)

            self.alg_models_manager.get_model(self.models, 'feature extractor').train()
            self.alg_models_manager.get_model(self.models, 'classifier').eval()

            if self.hparams['Augmentation'][domain_index] == 3 and iter_index > 0:
                cf = confusion_cf
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning)
                cf = cf.astype('float') / (cf.sum(axis=1)[:, np.newaxis]+ 1e-10)
                current_size = cf.shape

                pad_left = 0
                pad_right = 200 - current_size[0]
                pad_top = 0
                pad_bottom = 200 - current_size[1]
                cf = F.pad(torch.tensor(cf), (pad_left, pad_right, pad_top, pad_bottom), value=0)

                for l1 in range(200):
                    cf[l1][l1] = 0

            if (self.hparams['Augmentation'][domain_index] == 1 or self.hparams['Augmentation'][domain_index] == 3):


                try:
                    input2_t, target2_t = next(inverse_iter_t)
                except:
                    inverse_iter_t = iter(weighted_train_loader_t)
                    input2_t, target2_t = next(inverse_iter_t)
                input2_t = input2_t[:target_x.size()[0]]
                target2_t = target2_t[:target_x.size()[0]]
                input2_t = input2_t.cuda(0, non_blocking=True)
                target2_t = target2_t.cuda(0, non_blocking=True)


            # print(y)
            # print(ads)
            if self.hparams['Augmentation'][domain_index] == 1:
                # generate mixed sample
                lam = np.random.beta(1.0, 1.0)

                bbx1, bby1, bbx2, bby2 = rand_bbox(target_x.size(), lam)
                target_x[:, :, bbx1:bbx2, bby1:bby2] = input2_t[:, :, bbx1:bbx2, bby1:bby2]



            elif self.hparams['Augmentation'][domain_index] == 3 and iter_index > 0:
                B, c, w, h = source_x.size()
                net = self.alg_models_manager.get_model(self.models, 'feature extractor')



                f = net.forward_features(source_x).reshape(B, -1)
                f2 = net.forward_features(input2_t).reshape(B, -1)

                tar1 = source_y.cpu().numpy()
                tar2 = target2_t.cpu().numpy()

                # get confusion
                cf_scale = [[0 for t2 in range(B)] for t1 in range(B)]
                for q1 in range(B):
                    for q2 in range(B):
                        cf_scale[q1][q2] = cf[tar2[q1]][tar1[q2]]

                cf_scale = torch.from_numpy(np.array(cf_scale)).cuda(0)

                # get distance
                x_col = f2.unsqueeze(-2)
                y_lin = f.unsqueeze(-3)
                C_feature = 1 - d_cosine(x_col, y_lin)

                # get cost Cij
                M = 0.05 * C_feature + (1 - 0.05) * (1 - cf_scale)

                # get Tij
                x_points = f.shape[-2]
                y_points = f2.shape[-2]
                if f.dim() == 2:
                    batch_size = 1
                else:
                    batch_size = x_points.shape[0]

                a = torch.empty(batch_size, x_points, dtype=torch.float,
                                requires_grad=False).fill_(1 / x_points).cuda().squeeze()
                b = torch.empty(batch_size, y_points, dtype=torch.float,
                                requires_grad=False).fill_(1 / x_points).cuda().squeeze()
                T = ot.sinkhorn(a, b, M, reg=0.01)

                # OTmix operation
                input3_t = torch.zeros_like(input2_t)

                T = T.cpu().detach().numpy()

                lam = np.random.beta(self.hparams['beta'], self.hparams['beta'])
                bbx1, bby1, bbx2, bby2 = rand_bbox(target_x.size(), lam)
                for k in range(B):
                    index = np.argmax(T[k], axis=None)
                    input3_t[k, :, :, :] = target_x[index, :, :, :]
                    input3_t[k, :, bbx1:bbx2, bby1:bby2] = input2_t[k, :, bbx1:bbx2, bby1:bby2]


                target_x=input3_t


            elif self.hparams['Augmentation'][domain_index] == 4:

                # input_org_1 = inputs[0].cpu()
                # input_org_2 = inputs[1].cpu()

                input_org_1 = target_x.cpu()
                input_org_2 = target_x.cpu()
                target_org = target_y.cpu()
                #net = self.alg_models_manager.get_model(self.models, 'main model')



                try:
                    input_invs, target_invs = next(inverse_iter_t)
                except:
                    inverse_iter = weighted_train_loader_t
                    input_invs, target_invs = next(inverse_iter_t)

                input_invs_1 = input_invs[:input_org_1.size()[0]]
                input_invs_2 = input_invs[:input_org_2.size()[0]]

                per_cls_weights = 1.0 / (np.array(cls_num_list_t) ** 1.0)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list_t)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

                one_hot_org = torch.zeros(target_org.size(0), 200).scatter_(1, target_org.view(-1, 1), 1)
                one_hot_org_w = per_cls_weights.cpu() * one_hot_org
                one_hot_invs = torch.zeros(target_invs.size(0), 200).scatter_(1, target_invs.view(-1, 1), 1)
                one_hot_invs = one_hot_invs[:one_hot_org.size()[0]]
                one_hot_invs_w = per_cls_weights.cpu() * one_hot_invs

                input_org_1 = input_org_1.cuda()
                input_org_2 = input_org_2.cuda()
                input_invs_1 = input_invs_1.cuda()
                input_invs_2 = input_invs_2.cuda()

                one_hot_org = one_hot_org.cuda()
                one_hot_org_w = one_hot_org_w.cuda()
                one_hot_invs = one_hot_invs.cuda()
                one_hot_invs_w = one_hot_invs_w.cuda()



                mix_x, cut_x, mixup_y, mixcut_y, mixup_y_w, cutmix_y_w = GLMC_mixed(org1=input_org_1,
                                                                                    org2=input_org_2,
                                                                                    invs1=input_invs_1,
                                                                                    invs2=input_invs_2,
                                                                                    label_org=one_hot_org,
                                                                                    label_invs=one_hot_invs,
                                                                                    label_org_w=one_hot_org_w,
                                                                                    label_invs_w=one_hot_invs_w)

                target_x=mix_x
            #print(f"index: {index}")
            if self.hparams['Augmentation'][domain_index] == 3 :
                net = self.alg_models_manager.get_model(self.models, 'feature extractor')
                net2 = self.alg_models_manager.get_model(self.models, 'classifier')
                output =net2(net(target_x))
                _, pred = torch.max(output, 1)
                #print("_____",pred,target_y)
                confusion_cf = confusion_matrix(target_y.cpu().numpy(), pred.cpu().numpy()).astype(float)





            # if edgetailor:
            #     #print(target_y.size())
            #     tail_class_label =list(range(4, len(cls_num_list) + 1, 1))
            #     num_samples=3
            #     bufferS.add_samples(source_y, source_x)
            #     bufferT.add_samples(target_y, target_x)
            #     source_x,source_y = bufferS.get_samples(tail_class_label, num_samples, source_x,source_y)
            #     target_x,target_y=bufferT.get_samples(tail_class_label,num_samples,target_x,target_y)
            #
            #     z = target_x.size()
            #     #print(target_y)
            #     xx = target_x.reshape(target_x.shape[0], -1)
            #     xx, target_y = ssmote.fit_resample(xx.cpu(), target_y.cpu())
            #     target_x = xx.reshape(-1, z[1], z[2], z[3])
            #     target_x = torch.tensor(target_x).cuda()
            #     target_y = torch.tensor(target_y).cuda().int()
            #
            #     z = source_x.size()
            #     #print(source_y)
            #     xx = source_x.reshape(source_x.shape[0], -1)
            #     xx, source_y = ssmote.fit_resample(xx.cpu(), source_y.cpu())
            #     source_x = xx.reshape(-1, z[1], z[2], z[3])
            #     source_x = torch.tensor(source_x).cuda()
            #     source_y = torch.tensor(source_y).cuda().int()





            source_feature = self.alg_models_manager.get_feature({'feature extractor': self.raw_ft}, source_x)
            target_feature = self.alg_models_manager.get_feature(self.models, target_x)

            mmd_distance = mmd_rbf(source_feature, target_feature)

            if self.replay_train_set is not None:
                replay_x, replay_y = next(replay_train_loader)
                replay_x, replay_y = replay_x.to(self.device), replay_y.to(self.device)
                replay_loss = self.alg_models_manager.forward_to_compute_task_loss(self.models, replay_x, replay_y)
            else:
                replay_loss = 0.

            loss = mmd_distance + self.hparams['replay_loss_alpha'] * replay_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



            exp_tracker.add_losses({ 'MMD': mmd_distance, 'replay': self.hparams['replay_loss_alpha'] * replay_loss }, iter_index)
            exp_tracker.in_each_iteration_of_each_da()



        if self.hparams['replay_loss_alpha'] != 0:
            cur_replay_x = None
            while True:
                #target_x = torch.tensor([item.cpu().detach().numpy() for item in next(target_train_loader)[0:1]]).cuda()
                #target_x = torch.cat([item  for item in next(target_train_loader)[0:1]], dim=0).cuda()
                target_x = next(target_train_loader)[0]
                #target_x, = next(iter(target_train_loader))

                if cur_replay_x is None:
                    cur_replay_x = target_x   #.squeeze(dim=0)

                else:
                    cur_replay_x = torch.cat((cur_replay_x, target_x))
                if cur_replay_x.size(0) > self.hparams['num_replay_samples_each_domain']:
                    cur_replay_x = cur_replay_x[0: self.hparams['num_replay_samples_each_domain']]
                    break
            cur_replay_y = self.alg_models_manager.predict(self.models, cur_replay_x.to(self.device))
            cur_replay_y = self.alg_models_manager.get_pseudo_y_from_model_output(cur_replay_y)
            
            if self.replay_train_set is None:
                self.all_replay_x, self.all_replay_y = cur_replay_x, cur_replay_y
                '''
            else:
                self.all_replay_x=self.all_replay_x.unsqueeze(0)
                print(self.all_replay_x.size(),cur_replay_x.size(),self.all_replay_y.size(), cur_replay_y.size())
                #self.all_replay_x, self.all_replay_y = torch.cat( [self.all_replay_x.cpu(), cur_replay_x.cpu()]),torch.cat([self.all_replay_y.cpu(), cur_replay_y.cpu()])
                self.all_replay_x, self.all_replay_y =self.all_replay_x.cpu(), torch.cat([self.all_replay_y.cpu(), cur_replay_y.cpu()])
            '''
            else:

                cur_replay_x = cur_replay_x.squeeze(dim=0)

                self.all_replay_x, self.all_replay_y = torch.cat(
                    [self.all_replay_x.cpu(), cur_replay_x.cpu()]), torch.cat([self.all_replay_y.cpu(), cur_replay_y.cpu()])

            self.replay_train_set = TensorDataset(self.all_replay_x.cpu(), self.all_replay_y.cpu())

        #import psutil

        #virtual_memory = psutil.virtual_memory()
        #used_memory = virtual_memory.used
        #used_memory_mb = used_memory / (1024 ** 2)
        #print(f"Used Memory: {used_memory_mb:.2f} MB")
