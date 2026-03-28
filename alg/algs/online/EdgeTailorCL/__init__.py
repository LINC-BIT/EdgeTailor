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
from torch.autograd import Variable
import ot

import sys, time, os
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from utils import *
from torch.utils.tensorboard import SummaryWriter
import quadprog
sys.path.append('..')
import torch.nn.functional as F
import torch.nn as nn
import random
criteriton = nn.CrossEntropyLoss()
d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)

from benchmark.alg.algs.online.EdgeTail.search import heuristic_search
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

T_L = {
    "SRR": 13.0,
    "CMO": 3.0,
    "DCL": 17.0,
    "Remix": 3.0,
    "OTmix": 3.0,
    "GLMC": 3.0
}
M_L = {
    "SRR": 300,
    "CMO": 300,
    "DCL": 300,
    "Remix": 300,
    "OTmix": 300,
    "GLMC": 300
}


@algorithm_register(
    name='EdgeTail',
    stage='online',
    supported_tasks_type=['Image Classification']
)
class EdgeTail(ABOnlineDAAlgorithm):
    def __init__(self, models, alg_models_manager: ABAlgModelsManager, hparams: Dict, device, random_seed,
                 res_save_dir):
        super().__init__(models, alg_models_manager, hparams, device, random_seed, res_save_dir)

        self.replay_train_set = None
        self.all_replay_x, self.all_replay_y = None, None
        self.met_source_datasets_name = []


        self.consolidate = True
        self.fisher_estimation_sample_size = 128
        self.loss_log_interval = 30
        self.eval_log_interval = 50

    def verify_args(self, models, alg_models_manager, hparams: dict):
        #assert all([n in models.keys() for n in ['feature extractor', 'classifier']])
        assert all([isinstance(models[n], tuple) and len(models[n]) > 0 for n in
                    models.keys()]), 'pass the path of model file in'

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
            'domain': list,
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
        model = self.alg_models_manager.get_model(self.models, 'main model')
        configure_model(model)
        params, _ = collect_params(model)
        self.optimizer = torch.optim.__dict__[self.hparams['optimizer']](
            model.parameters(), **self.hparams['optimizer_args'])

        target_train_set = scenario.get_limited_target_train_dataset()

        target_train_loader = iter(
            scenario.build_dataloader(target_train_set, self.hparams['batch_size'], self.hparams['num_workers'], True,
                                      False))

        source_train_set = scenario.get_source_datasets('train')

        source_train_loaders = [
            iter(scenario.build_dataloader(d, self.hparams['batch_size'], self.hparams['num_workers'], True, False))
            for n, d in source_train_set.items()]

        target_domain = scenario.get_domain_index()
        if target_domain in self.hparams['domain']:
            domain_index = self.hparams['domain'].index(target_domain)

        else:
            print(f"'{target_domain}' 不在列表中")

        print(target_domain)
        net = model

        if self.consolidate :

            print(
                '=> Estimating diagonals of the fisher information matrix...',
                flush=True, end='',
            )
            net.consolidate(net.estimate_fisher(
                target_train_set, self.fisher_estimation_sample_size,scenario
            ))
            print(' Done!')

        SL = ["SRR", "CMO", "DCL", "Remix", "OTmix", "GLMC"]

        SH = {
            "SRR": {
                "oversample_factor": [0.5, 1.0, 2.0],
                "undersample_factor": [0.1, 0.5, 1.0]
            },
            "CMO": {
                "weighted_alpha_a": [0.1, 0.3, 0.5],
                "mixup_prob": [0.2, 0.5, 0.8]
            },
            "DCL": {
                "weighted_alpha_r": [0.1, 0.5, 1.0]
            },
            "Remix": {
                "tau": [1, 2, 5],
                "kappa": [1, 3, 5]
            },
            "OTmix": {
                "weighted_alpha_a": [0.1, 0.3, 0.5],
                "mixup_prob": [0.2, 0.5, 0.8]
            },
            "GLMC": {
                "weighted_alpha_a": [0.1, 0.3, 0.5],
                "mixup_prob": [0.2, 0.5, 0.8]
            }
        }

        def poly2(vars, D, C, L, T, O, U, A):
            d, l, t, o, u, a = vars
            return (C / d) ** D * l ** L * (1 / t) ** T * (o) ** O * u ** U * a ** A


        class ScalingLawAdapter(nn.Module):
            def __init__(self, input_dim, hidden_dim=128, output_dim=5):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )

            def forward(self, x):
                return self.fc(x)


        adapter = ScalingLawAdapter(input_dim=15)
        adapter.load_state_dict(torch.load("scaling_law_adapter.pth", map_location="cpu"))
        adapter.eval()

        # 方法和超参数 embedding
        method2id = {m: i for i, m in enumerate(["SRR", "CMO", "DCL", "Remix", "OTmix", "GLMC"])}
        hparam2id = {
            "oversample_factor": 0, "undersample_factor": 1,
            "weighted_alpha_a": 2, "mixup_prob": 3, "weighted_alpha_r": 4,
            "tau": 5, "kappa": 6
        }


        def eval_func(L, H, q, rho=0.1):

            method_vec = np.zeros(len(method2id))
            if L:
                method_vec[method2id[L]] = 1.0
            hparam_vec = np.zeros(len(hparam2id))
            if H:
                # H_key = tuple(sorted(H.items()))
                # hparam_vec[hparam2id[H_key[0]]] = H_key[1]
                for k, v in H.items():
                    hparam_vec[hparam2id[k]] = v

            I = np.concatenate([[rho, q], method_vec, hparam_vec])
            I = torch.tensor(I, dtype=torch.float32).unsqueeze(0)


            with torch.no_grad():
                O_pred = adapter(I).squeeze().numpy()
            beta, alphaD, alphal, alphat, alphaq = O_pred


            vars_data = [1000, 50, q, 1, rho, 1]


            acc = poly2(vars_data, beta, 1.0, alphaD, alphal, alphat, alphaq, 1.0)

            if L:
                TL = T_L[L]
                ML = M_L[L]
            else:
                TL =0
                ML =0

            return acc, TL, ML

        # 调用搜索
        L_star, H_star, q_star, acc_star = heuristic_search(
            SL, SH, eval_func,
            T=1.0, M=200,
            T_avail=50, M_avail=150,
            K=20
        )






        if L_star == "SRR":
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

        if L_star == "DCL":
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
        if (L_star=="CMO" or L_star=="GLMC" or L_star =="OTmix"):
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
            inverse_iter = iter(weighted_train_loader_t)
        if L_star=="Remix":
            # cls_num_list = {n: d.dataset.get_cls_num_list() for n, d in source_train_set.items()}
            # source_train_set = {n: RemixDataset(d.dataset, cls_num_list[n], beta=1.0,tau=self.hparams['tau'], kappa=self.hparams['kappa'])  for n, d in source_train_set.items()}
            #
            # # Create DataLoader
            # #trainloader = DataLoader(remix_trainset, batch_size=p.BATCH_SIZE, shuffle=True, num_workers=2)
            # source_train_loaders = [iter(scenario.build_dataloader(d, self.hparams['batch_size'],
            #                                    self.hparams['num_workers'], True, True)) for n, d in
            #  source_train_set.items()]

            cls_num_list_t =target_train_set.dataset.get_cls_num_list()
            print("cls_num_list_t:",cls_num_list_t)
            target_train_set = RemixDataset(target_train_set.dataset, cls_num_list_t, beta=1.0, tau=self.hparams['tau'][domain_index],
                                                kappa=self.hparams['kappa'][domain_index])

            # Create DataLoader
            # trainloader = DataLoader(remix_trainset, batch_size=p.BATCH_SIZE, shuffle=True, num_workers=2)
            target_train_loader = iter(scenario.build_dataloader(target_train_set, self.hparams['batch_size'],
                                                                   self.hparams['num_workers'], True, False))

        cls_list = [0] * 200
        for iter_index in range(self.hparams['num_iters']):
            # cua_time = time.time()
            # if cua_time-start_time>75:
            #     break
            #print(iter_index,self.hparams['num_iters'])
            #print(iter_index % len(source_train_loaders))
            mix_prob = random.random()
            source_x, source_y = next(source_train_loaders[iter_index % len(source_train_loaders)])
            target_x ,target_y = next(target_train_loader)
            #target_x= torch.tensor([item.cpu().detach().numpy() for item in item_[0:1]]).cuda().squeeze(0)
            #target_x = torch.cat([item for item in item_[0:1]], dim=0).cuda().squeeze(0)
            #target_y=torch.tensor([item.cpu().detach().numpy() for item in item_[1:2]]).cuda().squeeze(0)
            #target_y = torch.cat([item for item in item_[1:2]], dim=0).cuda().squeeze(0)
            if L_star== "remix":
                # print(labels)

                labels_a, labels_b, lam = target_y

                target_y = labels_a * lam + labels_b * (1 - lam)
                for i in range(labels_a.size(0)):
                    cls_list[labels_a[i].item()] += lam[i].item()
                for i in range(labels_b.size(0)):
                    cls_list[labels_b[i].item()] += 1 - lam[i].item()
                target_x = torch.tensor(target_x)
                target_y = torch.tensor(target_y)

            source_x, source_y, target_x,target_y = source_x.to(self.device), source_y.to(self.device), target_x.to(self.device),target_y.to(self.device)
            x=target_x
            y=target_y

            if L_star =="OTmix" and iter_index > 0:
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

            if (L_star =="CMO" or L_star =="OTmix") and (
                    H_star['mixup_prob']> mix_prob):
                try:
                    input2, target2 = next(inverse_iter)
                except:
                    inverse_iter = iter(weighted_train_loader_t)
                    input2, target2 = next(inverse_iter)
                input2 = input2[:x.size()[0]]
                target2 = target2[:y.size()[0]]
                input2 = input2.cuda(0, non_blocking=True)
                target2 = target2.cuda(0, non_blocking=True)







            # prepare the data.
            #x,y=x.to(self.device),y.to(self.device)
            x = Variable(x).cuda()
            y = Variable(y).cuda()



            # run the model and backpropagate the errors.


            if (L_star =="CMO" ) and (
                    H_star['mixup_prob']> mix_prob):
                # generate mixed sample
                lam = np.random.beta(1.0, 1.0)
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                x[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                # compute output
                outputs = self.alg_models_manager.get_outputs(self.models, x)
                ce_loss = criteriton(outputs, y) * lam + criteriton(outputs, target2) * (1. - lam)
            elif (L_star =="Remix" ) and (
                    H_star['mixup_prob']> mix_prob):
                outputs = self.alg_models_manager.get_outputs(self.models, x)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                lam = lam.to(device)
                # lam = lam.to(device)
                log_probs = F.log_softmax(outputs, dim=1)
                # 获取每个样本对应标签的 log 概率
                log_probs_for_labels_a = log_probs[range(labels_a.__len__()), labels_a.to(device)]
                # 使用 lambda 加权
                weighted_log_probs_a = log_probs_for_labels_a * lam
                # 计算最终损失
                loss_a = -weighted_log_probs_a.mean()

                log_probs_for_labels_b = log_probs[range(labels_b.__len__()), labels_b.to(device)]
                # 使用 lambda 加权
                weighted_log_probs_b = log_probs_for_labels_b * (1 - lam)
                # 计算最终损失
                loss_b = -weighted_log_probs_b.mean()

                ce_loss = loss_a + loss_b

            elif (L_star =="OTmix" ) and iter_index > 0 and (H_star['mixup_prob']> mix_prob):
                B, c, w, h = x.size()
                net = self.alg_models_manager.get_model(self.models, 'main model')
                model_f = torch.nn.Sequential(*list(net.children())[:-1])
                # get feature
                # input:background image
                # input2:foreground image
                #model_f = torch.nn.Sequential(*list(model.children())[:-1])
                f = model_f(x).reshape(B, -1)
                f2 = model_f(input2).reshape(B, -1)

                tar1 = y.cpu().numpy()
                tar2 = target2.cpu().numpy()

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
                input3 = torch.zeros_like(input2)
                target3 = torch.zeros_like(y)
                T = T.cpu().detach().numpy()

                lam = np.random.beta(self.hparams['beta'], self.hparams['beta'])
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                for k in range(B):
                    index = np.argmax(T[k], axis=None)
                    input3[k, :, :, :] = x[index, :, :, :]
                    input3[k, :, bbx1:bbx2, bby1:bby2] = input2[k, :, bbx1:bbx2, bby1:bby2]
                    target3[k] = y[index]

                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

                # compute output
                output = net(input3)
                ce_loss = criteriton(output, target3) * lam + criteriton(output, target2) * (1. - lam)

            elif (L_star =="GLMC") and (H_star['mixup_prob']> mix_prob):

                # input_org_1 = inputs[0].cpu()
                # input_org_2 = inputs[1].cpu()
                cls_num_list = target_train_set.dataset.get_cls_num_list()
                #cls_num_list = {n: d.dataset.get_cls_num_list() for n, d in train_sets.items()}
                input_org_1 = x.cpu()
                input_org_2 = x.cpu()
                target_org = y.cpu()
                net = self.alg_models_manager.get_model(self.models, 'main model')

                try:
                    input_invs, target_invs = next(inverse_iter)
                except:
                    inverse_iter = iter(weighted_train_loader_t)
                    input_invs, target_invs = next(inverse_iter)

                input_invs_1 = input_invs[:input_org_1.size()[0]]
                input_invs_2 = input_invs[:input_org_2.size()[0]]

                per_cls_weights = 1.0 / (np.array(cls_num_list))
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

                target_org = target_org % 10
                target_invs = target_invs % 10

                one_hot_org = torch.zeros(target_org.size(0), 10).scatter_(1, target_org.view(-1, 1), 1)
                one_hot_org_w = per_cls_weights.cpu() * one_hot_org
                one_hot_invs = torch.zeros(target_invs.size(0), 10).scatter_(1, target_invs.view(-1, 1), 1)
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

                # measure data loading time

                # Data augmentation
                lam = np.random.beta(self.hparams['beta'], self.hparams['beta'])

                mix_x, cut_x, mixup_y, mixcut_y, mixup_y_w, cutmix_y_w = GLMC_mixed(org1=input_org_1,
                                                                                    org2=input_org_2,
                                                                                    invs1=input_invs_1,
                                                                                    invs2=input_invs_2,
                                                                                    label_org=one_hot_org,
                                                                                    label_invs=one_hot_invs,
                                                                                    label_org_w=one_hot_org_w,
                                                                                    label_invs_w=one_hot_invs_w)
                net.setGLMC()
                output_1, output_cb_1, z1, p1 = net.forwardGLMC(mix_x)
                output_2, output_cb_2, z2, p2 = net.forwardGLMC(cut_x)
                contrastive_loss = SimSiamLoss(p1, z2) + SimSiamLoss(p2, z1)

                loss_mix = -torch.mean(torch.sum(F.log_softmax(output_1, dim=1) * mixup_y, dim=1))
                loss_cut = -torch.mean(torch.sum(F.log_softmax(output_2, dim=1) * mixcut_y, dim=1))
                loss_mix_w = -torch.mean(torch.sum(F.log_softmax(output_cb_1, dim=1) * mixup_y_w, dim=1))
                loss_cut_w = -torch.mean(torch.sum(F.log_softmax(output_cb_2, dim=1) * cutmix_y_w, dim=1))

                balance_loss = loss_mix + loss_cut
                rebalance_loss = loss_mix_w + loss_cut_w
                alpha = 0.8

                ce_loss = alpha * balance_loss + (1 - alpha) * rebalance_loss + 2 * contrastive_loss
                outputs = output_1

            else:
                scores = net(x)

                ce_loss = criteriton(scores, y)


            if self.hparams['Augmentation'][domain_index] == 3:
                net = self.alg_models_manager.get_model(self.models, 'main model')
                output = net(x)
                _, pred = torch.max(output, 1)
                # print("_____",pred,target_y)
                confusion_cf = confusion_matrix(y.cpu().numpy(), pred.cpu().numpy()).astype(float)


            loss = ce_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # calculate the training precision.
            # _, predicted = scores.max(1)
            # precision = (predicted == y).sum().float() / len(x)







def collect_params(model):
    """Collect all trainable parameters.
    Walk the model's modules and collect all parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    # print(nm, np)
    return params, names

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = True
            # m.running_mean = None
            # m.running_var = None
        else:
            m.requires_grad_(True)
    return model
