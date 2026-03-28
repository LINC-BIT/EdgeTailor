from ....ab_algorithm import ABOfflineTrainAlgorithm
from .....exp.exp_tracker import OfflineTrainTracker
from .....scenario.scenario import Scenario
from ....registery import algorithm_register

from schema import Schema
import torch
import tqdm
from sklearn.metrics import confusion_matrix
from benchmark.longtail.WB import MaxNorm_via_PGD

thresh = 0.1  # threshold value
# pgdFunc = MaxNorm_via_PGD(thresh=thresh)

import random

# from benchmark.longtail import *


import torch.nn.functional as F
# import benchmark.long.parameter as p
from benchmark.long.SquareRoot import square_root_biased_sampling
from benchmark.long.Progressivelybalanced import *
from benchmark.long.remix import *
from benchmark.long.GLMC import GLMC_mixed, SimSiamLoss
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remix_criterion(pred, y, criterion):
    y1, y2, lam = y
    y1 = y1.to(device)
    y2 = y2.to(device)
    pred = pred.to(device)
    lam = lam.to(device)
    loss = lam * criterion(pred, y1) + (1 - lam) * criterion(pred, y2)
    return loss


criterion = nn.CrossEntropyLoss()


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


global confusion_cf

d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
import ot


@algorithm_register(
    name='OFFSourceTrain',
    stage='offline',
    supported_tasks_type=['Image Classification', 'Object Detection']
)
class OFFSourceTrain(ABOfflineTrainAlgorithm):
    fisher_estimation_sample_size = 128

    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert 'main model' in models.keys()

        Schema({
            'batch_size': int,
            'num_workers': int,
            'num_iters': int,
            'optimizer': str,
            'optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict,
            'Resampling': int,
            'Augmentation': int,
            'oversample_factor': float,
            'undersample_factor': float,
            'weighted_alpha_a': float,
            'weighted_alpha_r': float,
            'mixup_prob': float,
            'tau': float,
            'kappa': float,
            'beta': float,
        }).validate(hparams)

    def train(self, scenario: Scenario, exp_tracker: OfflineTrainTracker):
        model = self.alg_models_manager.get_model(self.models, 'main model')
        model = model.to(self.device)

        optimizer = torch.optim.__dict__[self.hparams['optimizer']](model.parameters(),
                                                                    **self.hparams['optimizer_args'])
        # optimizer = torch.optim.__dict__[self.hparams['optimizer']](model.parameters(), **self.hparams['optimizer_args'])

        scheduler = torch.optim.lr_scheduler.__dict__[
            self.hparams['scheduler']](optimizer, **self.hparams['scheduler_args'])

        train_sets = scenario.get_source_datasets('train')
        train_loaders = {n: iter(scenario.build_dataloader(d, self.hparams['batch_size'],
                                                           self.hparams['num_workers'], True, True)) for n, d in
                         train_sets.items()}
        # print(train_loaders.size())
        if self.hparams['Resampling'] == 1:
            # balanced_trainset = square_root_biased_sampling(train_sets, p.oversample_factor, p.undersample_factor)
            balanced_trainset = {n: square_root_biased_sampling(d.dataset, self.hparams['oversample_factor'],
                                                                self.hparams['undersample_factor']) for n, d in
                                 train_sets.items()}
            # train_loaders = DataLoader(balanced_trainset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=2)
            train_loaders = {n: iter(scenario.build_dataloader(d, self.hparams['batch_size'],
                                                               self.hparams['num_workers'], True, True)) for n, d in
                             balanced_trainset.items()}

            resampled_class_counts = {n: np.bincount(np.array(d.dataset.targets)[balanced_trainset[n].indices]) for n, d
                                      in
                                      train_sets.items()}
            train_sets = balanced_trainset
            print(f"Resampled class distribution: {resampled_class_counts}")

        if self.hparams['Resampling'] == 2:
            # unique_classes, class_counts = np.unique(train_sets.dataset.targets, return_counts=True)
            unique_classes = {n: np.unique(d.dataset.targets, return_counts=True)[0] for n, d in train_sets.items()}
            class_counts = {n: np.unique(d.dataset.targets, return_counts=True)[1] for n, d in train_sets.items()}
            initial_weights = {n: 1.0 / (d ** self.hparams['weighted_alpha']) for n, d in class_counts.items()}
            # initial_weights = 1.0 / (class_counts ** p.weighted_alpha)
            class_weights = initial_weights.copy()
            # balanced_trainset, sampled_indices = progressively_balanced_sampling(train_sets.dataset, class_weights)
            balanced_trainset = {n: progressively_balanced_sampling(d.dataset, class_weights[n])[0] for n, d in
                                 train_sets.items()}
            # train_loaders = DataLoader(balanced_trainset, batch_size= self.hparams['batch_size'], shuffle=True, num_workers=self.hparams['num_workers'])
            train_loaders = {n: iter(scenario.build_dataloader(d, self.hparams['batch_size'],
                                                               self.hparams['num_workers'], True, True)) for n, d in
                             balanced_trainset.items()}
            train_sets = balanced_trainset
            # class_counts = get_class_counts(train_sets, sampled_indices)
            print("Class sample counts in balanced_trainset:", class_counts)
        if (self.hparams['Augmentation'] == 1 or self.hparams['Augmentation'] == 3 or self.hparams[
            'Augmentation'] == 4):
            cls_num_list = {n: d.dataset.get_cls_num_list() for n, d in train_sets.items()}
            # weighted_sampler = train_sets.get_weighted_sampler()
            weighted_sampler = {n: d.dataset.get_weighted_sampler(weighted_alpha=self.hparams['weighted_alpha']) for
                                n, d in train_sets.items()}
            # print(weighted_sampler)
            # weighted_train_loader = torch.utils.data.DataLoader(
            #     train_sets, batch_size=self.hparams['batch_size'],
            #     num_workers=0, pin_memory=True, sampler=weighted_sampler)
            weighted_train_loader = {n: iter(scenario.build_dataloader(d, self.hparams['batch_size'],
                                                                       self.hparams['num_workers'], True, True,
                                                                       weight=weighted_sampler[n])) for n, d in
                                     train_sets.items()}
            inverse_iter = weighted_train_loader  # {n: iter(d) for n, d in  weighted_train_loader.items()}

        # trainloader = sampler1(trainloader,testloader)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,num_workers=2)

        if self.hparams['Augmentation'] == 2:
            cls_num_list = {n: d.dataset.get_cls_num_list() for n, d in train_sets.items()}
            remix_trainset = {n: RemixDataset(d.dataset, cls_num_list[n], beta=1.0, tau=self.hparams['tau'],
                                              kappa=self.hparams['kappa']) for n, d in train_sets.items()}

            # Create DataLoader
            # trainloader = DataLoader(remix_trainset, batch_size=p.BATCH_SIZE, shuffle=True, num_workers=2)
            train_loaders = {n: iter(scenario.build_dataloader(d, self.hparams['batch_size'],
                                                               self.hparams['num_workers'], True, True)) for n, d in
                             remix_trainset.items()}
            # trainloader, remix_dataset = create_dataloader(trainset,cls_num_list, batch_size=p.BATCH_SIZE,shuffle=True, num_workers=2)

        # cls_num_list =[scenario.get_clsnum(d)  for n, d in train_sets.items()]
        # cls_num_list=cls_num_list[0]

        exp_tracker.start_train()

        cls_list = [0] * 10

        for iter_index in tqdm.tqdm(range(self.hparams['num_iters']), desc='iterations',
                                    leave=False, dynamic_ncols=True):

            losses = {}
            mix_prob = random.random()
            if self.hparams['Augmentation'] == 3 and iter_index > 0:
                cf = confusion_cf
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning)
                cf = cf.astype('float') / (cf.sum(axis=1)[:, np.newaxis] + 1e-10)
                current_size = cf.shape

                pad_left = 0
                pad_right = 10 - current_size[0]
                pad_top = 0
                pad_bottom = 10 - current_size[1]
                cf = F.pad(torch.tensor(cf), (pad_left, pad_right, pad_top, pad_bottom), value=0)

                for l1 in range(10):
                    cf[l1][l1] = 0

            for train_loader_name, train_loader in train_loaders.items():
                model.train()
                self.alg_models_manager.set_model(self.models, 'main model', model)

                # x, y = next(train_loader)
                xx = next(train_loader)
                x, y = xx[:2]
                # print(xx)

                if (self.hparams['Augmentation'] == 1 or self.hparams['Augmentation'] == 3) and (
                        self.hparams['mixup_prob'] > mix_prob):
                    try:
                        input2, target2 = next(inverse_iter[train_loader_name])
                    except:
                        inverse_iter = {n: iter(d) for n, d in weighted_train_loader.items()}
                        input2, target2 = next(inverse_iter[train_loader_name])
                    input2 = input2[:x.size()[0]]
                    target2 = target2[:y.size()[0]]
                    input2 = input2.cuda(0, non_blocking=True)
                    target2 = target2.cuda(0, non_blocking=True)

                if self.hparams['Augmentation'] == 2:
                    # print(labels)

                    labels_a, labels_b, lam = y

                    y = labels_a * lam + labels_b * (1 - lam)
                    for i in range(labels_a.size(0)):
                        cls_list[labels_a[i].item()] += lam[i].item()
                    for i in range(labels_b.size(0)):
                        cls_list[labels_b[i].item()] += 1 - lam[i].item()

                x, y = x.to(self.device), y.to(self.device)
                # print(y)
                # print(ads)
                if self.hparams['Augmentation'] == 1 and (self.hparams['mixup_prob'] > mix_prob):
                    # generate mixed sample
                    lam = np.random.beta(1.0, 1.0)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                    x[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                    # compute output
                    outputs = self.alg_models_manager.get_outputs(self.models, x)
                    task_loss = criterion(outputs, y) * lam + criterion(outputs, target2) * (1. - lam)
                elif self.hparams['Augmentation'] == 2 and (self.hparams['mixup_prob'] > mix_prob):
                    outputs = self.alg_models_manager.get_outputs(self.models, x)
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

                    task_loss = loss_a + loss_b

                elif self.hparams['Augmentation'] == 3 and iter_index > 0 and (self.hparams['mixup_prob'] > mix_prob):
                    B, c, w, h = x.size()
                    net = self.alg_models_manager.get_model(self.models, 'main model')
                    model_f = torch.nn.Sequential(*list(net.children())[:-1])
                    # get feature
                    # input:background image
                    # input2:foreground image
                    f = net.forward_features(x).reshape(B, -1)
                    f2 = net.forward_features(input2).reshape(B, -1)

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
                    task_loss = criterion(output, target3) * lam + criterion(output, target2) * (1. - lam)

                elif self.hparams['Augmentation'] == 4 and (self.hparams['mixup_prob'] > mix_prob):

                    # input_org_1 = inputs[0].cpu()
                    # input_org_2 = inputs[1].cpu()
                    cls_num_list = {n: d.dataset.get_cls_num_list() for n, d in train_sets.items()}
                    input_org_1 = x.cpu()
                    input_org_2 = x.cpu()
                    target_org = y.cpu()
                    net = self.alg_models_manager.get_model(self.models, 'main model')

                    try:
                        input_invs, target_invs = next(inverse_iter[train_loader_name])
                    except:
                        inverse_iter = {n: iter(d) for n, d in weighted_train_loader.items()}
                        input_invs, target_invs = next(inverse_iter[train_loader_name])

                    input_invs_1 = input_invs[:input_org_1.size()[0]]
                    input_invs_2 = input_invs[:input_org_2.size()[0]]

                    per_cls_weights = 1.0 / (np.array(cls_num_list[train_loader_name]) ** 1.0)
                    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list[train_loader_name])
                    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

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

                    output_1, output_cb_1, z1, p1 = net(mix_x, Aug=4)
                    output_2, output_cb_2, z2, p2 = net(cut_x, Aug=4)
                    contrastive_loss = SimSiamLoss(p1, z2) + SimSiamLoss(p2, z1)

                    loss_mix = -torch.mean(torch.sum(F.log_softmax(output_1, dim=1) * mixup_y, dim=1))
                    loss_cut = -torch.mean(torch.sum(F.log_softmax(output_2, dim=1) * mixcut_y, dim=1))
                    loss_mix_w = -torch.mean(torch.sum(F.log_softmax(output_cb_1, dim=1) * mixup_y_w, dim=1))
                    loss_cut_w = -torch.mean(torch.sum(F.log_softmax(output_cb_2, dim=1) * cutmix_y_w, dim=1))

                    balance_loss = loss_mix + loss_cut
                    rebalance_loss = loss_mix_w + loss_cut_w
                    alpha = 0.8

                    task_loss = alpha * balance_loss + (1 - alpha) * rebalance_loss + 2 * contrastive_loss
                    outputs = output_1

                else:
                    task_loss = self.alg_models_manager.forward_to_compute_task_loss(self.models, x, y)

                if self.hparams['Augmentation'] == 3:
                    net = self.alg_models_manager.get_model(self.models, 'main model')
                    output = net(x)
                    _, pred = torch.max(output, 1)
                    # print("_____",pred,target_y)
                    confusion_cf = confusion_matrix(y.cpu().numpy(), pred.cpu().numpy()).astype(float)

                optimizer.zero_grad()


                task_loss.backward()
                optimizer.step()

                losses[train_loader_name] = task_loss

            exp_tracker.add_losses(losses, iter_index)
            if iter_index % 10 == 0:
                exp_tracker.add_running_perf_status(iter_index)

            scheduler.step()

            if iter_index % 500 == 0:
                met_better_model = exp_tracker.add_val_accs(iter_index)
                if met_better_model:
                    exp_tracker.add_models()
        print(
            '=> Estimating diagonals of the fisher information matrix...',
            flush=True, end='',
        )
        net = self.alg_models_manager.get_model(self.models, 'main model')
        net.consolidate(net.estimate_fisher(
            list(train_sets.values())[0], self.fisher_estimation_sample_size, scenario
        ))
        print(' Done!')
        exp_tracker.end_train()