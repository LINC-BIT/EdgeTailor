from curses import raw
from model_fbs import DomainDynamicConv2d, boost_raw_model_with_filter_selection, set_pruning_rate,\
    get_accmu_flops, get_cached_raw_w, get_cached_w, get_l1_reg_in_model, start_accmu_flops, train_only_gate
from utils.common.log import logger
import tqdm
import os
import shutil
import sys

from abc import abstractmethod, ABC
from typing import List
import torch.nn as nn
import copy
import random
import torch
import glob
import numpy as np
import math
# from methods.utils.data import get_source_dataloader
# from models.resnet_cifar.model_manager import ResNetCIFARManager
from utils.common.others import get_cur_time_str
from utils.common.data_record import write_json
from utils.dl.common.env import set_random_seed
from utils.dl.common.model import ModelSaveMethod, get_model_size, get_module, save_model, set_module
from utils.third_party.nni_new.common.graph_utils import TorchModuleGraph
from torchvision import transforms

# from models.resnet_cifar.resnet_cifar_3 import resnet18


# if __name__ == '__main__':
def run():
    
    # set_random_seed(0)
    # print(model)
    # exit(0)
    
    if len(sys.argv) == 1:
        run_tag = ''
    else:
        run_tag = '-' + sys.argv[1]
    
    cur_time_str = get_cur_time_str()
    raw_res_save_dir = res_save_dir = f'logs/experiments_trial/CIFAR10/ours_v1_pretrain/{cur_time_str[0:8]}/{cur_time_str[8:]}{run_tag}'
    os.makedirs(res_save_dir)
    logger.info(res_save_dir)
    
    shutil.copytree(os.path.dirname(__file__), os.path.join(res_save_dir, 'method'), 
                    ignore=shutil.ignore_patterns('*.pt', '*.pth', 'log', '__pycache__'))
    logger.info(f'res save dir: {res_save_dir}')

    batch_size = 128
    datasets_name = ['CIFAR10']
    datasets_num_classes = [10]
    device = 'cuda'
    
    warmup_lrs = [1e-3, 1e-3, 1e-3]
    warmup_num_iterses = [4000, 4000, 4000]
    warmup_milestoneses = [
        [2000],
        [2000],
        [2000]
    ]
    warmup_gamma = [0.1]
    
    lrs = [1e-3, 1e-2, 1e-2]
    affine_lrs = [1e-1, 1e-1, 1e-1]
    wds = [5e-4] * 3
    l1_wds = [1e-8, 3e-8, 5e-8]
    weight_reg_wds = [5e-4] * 3
    val_freq = 2000
    num_iterses = [80000 // 2, 80000 // 2, 80000]
    milestoneses = [
        [25000 // 2, 50000 // 2, 70000 // 2],
        [25000 // 2, 50000 // 2, 70000 // 2],
        [25000, 50000, 70000],
    ]
    gamma = 0.2
    
    max_sparsities = [0.2, 0.5, 0.8]
    
    last_sparsity_ckpt = 'logs/experiments_trial/CIFAR10/ours_v1_pretrain/20221103/224325/0.50/best_model_0.50.pt'
    last_sparsity = 0.5
    
    train_dataloaders = [
        get_source_dataloader(dataset_name, batch_size, 8, 'train', True, None, True) for dataset_name in datasets_name
    ]
    test_dataloaders = [
        get_source_dataloader(dataset_name, 256, 8, 'test', False, False, False) for dataset_name in datasets_name
    ]
    
    if last_sparsity_ckpt == '':
        raw_pretrained_ckpt = 'resnet18-raw-pretrained.pt'
        model = torch.load(raw_pretrained_ckpt).to(device)
        raw_model = torch.load(raw_pretrained_ckpt).to(device)
        
        affine_parameters = {}
        for name, p in model.named_parameters():
            if p.dim() > 0:
                affine_parameters[name] = torch.zeros((2, p.size(0))).to(device)
                affine_parameters[name].requires_grad = True
        
        val_accs = []
        y_offset = 0
        for dataset_num_classes, test_dataloader in zip(datasets_num_classes, test_dataloaders):
            val_accs += [ResNetCIFARManager.get_accuracy(model, test_dataloader, device, y_offset)]
            y_offset += dataset_num_classes
        avg_val_acc = sum(val_accs) / len(val_accs)
        print(avg_val_acc, val_accs)
        
        pruned_layers = []
        for i in range(1, 5):
            for j in range(2):
                pruned_layers += [f'layer{i}.{j}.conv1']
        ignore_layers = [layer for layer, m in model.named_modules() if isinstance(m, nn.Conv2d) and layer not in pruned_layers]
        model, conv_bn_map = boost_raw_model_with_filter_selection(model, 0., False, ignore_layers, True, (1, 3, 32, 32))
    
    else:
        raw_pretrained_ckpt = 'resnet18-raw-pretrained.pt'
        model = torch.load(raw_pretrained_ckpt).to(device)
        raw_model = torch.load(raw_pretrained_ckpt).to(device)
        
        pruned_layers = []
        for i in range(1, 5):
            for j in range(2):
                pruned_layers += [f'layer{i}.{j}.conv1']
        ignore_layers = [layer for layer, m in model.named_modules() if isinstance(m, nn.Conv2d) and layer not in pruned_layers]
        _, conv_bn_map = boost_raw_model_with_filter_selection(model, 0., False, ignore_layers, True, (1, 3, 32, 32))
        
        model = torch.load(last_sparsity_ckpt)
        affine_parameters = torch.load(last_sparsity_ckpt + '.affine_p')
        
    
    sparsity_i = 0
    for warmup_lr, warmup_num_iters, warmup_milestones, \
        lr, affine_lr, wd, l1_wd, weight_reg_wd, num_iters, milestones, \
        cur_max_sparsity in zip(warmup_lrs, warmup_num_iterses, warmup_milestoneses, lrs, affine_lrs, wds, l1_wds, weight_reg_wds, num_iterses, milestoneses, max_sparsities):
        
        if last_sparsity_ckpt != '' and last_sparsity >= cur_max_sparsity:
            print(f'skip sparsity {cur_max_sparsity}')
            continue
        sparsities_range = [0., cur_max_sparsity]
        
        res_save_dir = os.path.join(raw_res_save_dir, f'{cur_max_sparsity:.2f}')
        
        from torch.utils.tensorboard import SummaryWriter
        tb_log = SummaryWriter(os.path.join(res_save_dir, f'tb_log'))
        from tensorboard import program
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', os.path.join(res_save_dir, f'tb_log')])
        url = tb.launch()
        print(url)
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(os.path.join(res_save_dir, 'tb_log'))
        
        cur_k = sparsities_range[1]
        set_pruning_rate(model, cur_k)
        
        if warmup_lr > 0:
            gate_params = train_only_gate(model)
            import torch.optim
            optimizer = torch.optim.SGD(gate_params, lr=warmup_lr, momentum=0.9, weight_decay=0.)
            import torch.optim.lr_scheduler
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=warmup_milestones, gamma=warmup_gamma)
            pbar = tqdm.tqdm(range(warmup_num_iters), dynamic_ncols=True)
            for iter_index in pbar:
                tasks_loss = {}
                total_loss = 0.
                y_offset = 0
                for dataset_i, (dataset_name, dataset_num_classes, train_dataloader) in \
                    enumerate(zip(datasets_name, datasets_num_classes, train_dataloaders)):
                    x, y = next(train_dataloader)
                    x, y = x.to(device), y.to(device)
                    y += y_offset
                    
                    task_loss = ResNetCIFARManager.forward_to_gen_loss(model, x, y)
                    tasks_loss[dataset_name] = task_loss
                    total_loss += task_loss
                    
                    y_offset += dataset_num_classes
                
                tb_writer.add_scalar(f'losses/warmup_total_loss_{cur_k:.2f}', total_loss, iter_index)
                tb_writer.add_scalars(f'losses/warmup_tasks_loss_{cur_k:.2f}', tasks_loss, iter_index)
                pbar.set_description(f'(warmup) cur_k: {cur_k:.2f} loss: {total_loss:.6f}')
        
        
        for p in model.parameters():
            p.requires_grad = True
        import torch.optim
        params_group = [
            {'params': [p for n, p in model.named_parameters() if 'filter_selection' in n], 'weight_decay': 0.},
            {'params': [p for n, p in model.named_parameters() if 'filter_selection' not in n], 'weight_decay': wd},
            {'params': [p for p in affine_parameters.values()], 'lr': affine_lr}
        ]
        optimizer = torch.optim.SGD(params_group, lr=lr, momentum=0.9)
        import torch.optim.lr_scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        
        def get_weight_reg_loss(raw_model, feature_reg_model, affine_parameters):
            res = 0.
            for name, p in feature_reg_model.named_parameters():
                if p.dim() == 0:
                    continue
                if 'filter_selection_module' in name:
                    continue
                
                # print(name)
                
                raw_name = name if 'raw_conv2d' not in name else name.replace('raw_conv2d.', '')
                
                if 'raw_bn' in name:
                    raw_name = conv_bn_map[raw_name[0: raw_name.index('.raw_bn')]] + '.' + raw_name.split('.')[-1]
                # print(name, raw_name)
                raw_p = getattr(
                    get_module(raw_model, '.'.join(raw_name.split('.')[0:-1])),
                    raw_name.split('.')[-1]
                )

                affine_parameter = affine_parameters[raw_name]
                res += (affine_parameter[0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * p + \
                    affine_parameter[1].unsqueeze(1).unsqueeze(2).unsqueeze(3) - raw_p.detach()).norm(p=2) ** 2
                
            return res

        
        # from mab import MAB_UCB
        # num_cand_sparsities = 10
        # choose_trained_sparsity_mab = MAB_UCB(num_cand_sparsities, 0.1)
        # cand_sparsities = np.linspace(sparsities_range[0], sparsities_range[1], num=num_cand_sparsities)
        
        pbar = tqdm.tqdm(range(num_iters), dynamic_ncols=True)
        cur_best_acc = 0.
        for iter_index in pbar:
            model.train()
            optimizer.zero_grad()
            
            # ws = []
            tasks_loss = {d: 0. for d in datasets_name}
            l1s_loss = {d: 0. for d in datasets_name}
            weight_reg_losses = {d: 0. for d in datasets_name}
            total_loss = 0.
            
            y_offset = 0
            
            for dataset_i, (dataset_name, dataset_num_classes, train_dataloader) in \
                enumerate(zip(datasets_name, datasets_num_classes, train_dataloaders)):
                
                # -- max --
                set_pruning_rate(model, sparsities_range[0])
                x, y = next(train_dataloader)
                x, y = x.to(device), y.to(device)
                y += y_offset
                
                task_loss = ResNetCIFARManager.forward_to_gen_loss(model, x, y)
                tasks_loss[dataset_name] += task_loss
                total_loss += task_loss
                
                l1_loss = get_l1_reg_in_model(model)
                l1s_loss[dataset_name] += l1_loss * l1_wd
                total_loss += l1_loss * l1_wd
                
                weight_reg_loss = get_weight_reg_loss(raw_model, model, affine_parameters)
                weight_reg_losses[dataset_name] += weight_reg_loss * weight_reg_wd
                total_loss += weight_reg_loss * weight_reg_wd
                
                # raw_model.eval()
                # with torch.no_grad():
                #     raw_model_task_loss = ResNetCIFARManager.forward_to_gen_loss(raw_model, x, y, train=False)
                
                # -- mid --
                for _ in range(2):
                    # cur_sparsity = cand_sparsities[choose_trained_sparsity_mab.choose_objs(1)[0]]
                    cur_sparsity = random.random() * (sparsities_range[1] - sparsities_range[0]) + sparsities_range[0]
                    set_pruning_rate(model, cur_sparsity)
                    
                    x, y = next(train_dataloader)
                    x, y = x.to(device), y.to(device)
                    y += y_offset
                    
                    task_loss = ResNetCIFARManager.forward_to_gen_loss(model, x, y)
                    tasks_loss[dataset_name] += task_loss
                    total_loss += task_loss
                    
                    l1_loss = get_l1_reg_in_model(model)
                    l1s_loss[dataset_name] += l1_loss * l1_wd
                    total_loss += l1_loss * l1_wd
                    
                    weight_reg_loss = get_weight_reg_loss(raw_model, model, affine_parameters)
                    weight_reg_losses[dataset_name] += weight_reg_loss * weight_reg_wd
                    total_loss += weight_reg_loss * weight_reg_wd
                    
                    # reward = (task_loss - raw_model_task_loss) / raw_model_task_loss
                    # choose_trained_sparsity_mab.feedback_reward(reward) # task_loss is larger, the sparsity is easier to be chosen later
                
                # -- min --
                set_pruning_rate(model, sparsities_range[1])
                x, y = next(train_dataloader)
                x, y = x.to(device), y.to(device)
                y += y_offset
                
                task_loss = ResNetCIFARManager.forward_to_gen_loss(model, x, y)
                tasks_loss[dataset_name] += task_loss
                total_loss += task_loss
                
                l1_loss = get_l1_reg_in_model(model)
                l1s_loss[dataset_name] += l1_loss * l1_wd
                total_loss += l1_loss * l1_wd
                
                weight_reg_loss = get_weight_reg_loss(raw_model, model, affine_parameters)
                weight_reg_losses[dataset_name] += weight_reg_loss * weight_reg_wd
                total_loss += weight_reg_loss * weight_reg_wd
                
                y_offset += dataset_num_classes
                # ws += [get_cached_raw_w(model).mean(0)]
            
            # ws = torch.stack(ws)
            # var_domain_w_diff_loss = (ws - ws.mean(0)).norm(1, dim=1).mean()
            # total_loss += var_domain_w_diff_alpha * var_domain_w_diff_loss
            
            total_loss /= (len(datasets_name) * 4)
            tasks_loss = {k: v / (len(datasets_name) * 4) for k, v in tasks_loss.items()}
            l1s_loss = {k: v / (len(datasets_name) * 4) for k, v in l1s_loss.items()}
            weight_reg_losses = {k: v / (len(datasets_name) * 4) for k, v in weight_reg_losses.items()}
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            tb_writer.add_scalar(f'losses/total_loss', float(total_loss), iter_index)
            tb_writer.add_scalars(f'losses/tasks_loss', tasks_loss, iter_index)
            tb_writer.add_scalars(f'losses/l1_loss', l1s_loss, iter_index)
            tb_writer.add_scalars(f'losses/weight_reg_loss', weight_reg_losses, iter_index)
            # tb_writer.add_scalar(f'losses/var_domain_w_diff_loss_{cur_k:.2f}', var_domain_w_diff_loss, iter_index)
            tb_writer.add_scalar(f'lr/lr', optimizer.param_groups[0]['lr'], iter_index)
            pbar.set_description(f'cur_k: {sparsities_range[1]:.2f} loss: {total_loss:.6f} cur_best_acc: {cur_best_acc:.4f}')
            
            # if iter_index % 100 == 0:
            #     sparsity_selection_record = []
            #     for ri, r in enumerate(choose_trained_sparsity_mab.obj_selected_record):
            #         sparsity_selection_record += [cand_sparsities[ri]] * r
                    
            #     tb_writer.add_histogram('mab_selection', np.array(sparsity_selection_record), iter_index)
            
            # subnetwork sampling and training alone
            y_offset = 0
            set_pruning_rate(model, random.random() * (sparsities_range[1] - sparsities_range[0]) + sparsities_range[0])
            for dataset_i, (dataset_name, dataset_num_classes, train_dataloader) in \
                enumerate(zip(datasets_name, datasets_num_classes, train_dataloaders)):
                x, y = next(train_dataloader)
                x, y = x.to(device), y.to(device)
                y += y_offset
                
                for layer in model.modules():
                    if isinstance(layer, DomainDynamicConv2d):
                        layer.static_w = None
                with torch.no_grad():
                    ResNetCIFARManager.forward(model, x)
                for layer in model.modules():
                    if isinstance(layer, DomainDynamicConv2d):
                        layer.static_w = layer.cached_w[0].squeeze().detach()
                
                task_loss = ResNetCIFARManager.forward_to_gen_loss(model, x, y)
                optimizer.zero_grad()
                task_loss.backward()
                optimizer.step()
                # scheduler.step()
                
                y_offset += dataset_num_classes
                
            for layer in model.modules():
                if isinstance(layer, DomainDynamicConv2d):
                    layer.static_w = None
            
            if (iter_index + 1) % val_freq == 0:
                avg_accs_in_var_sparsity = {}
                
                for cur_k in np.linspace(sparsities_range[0], sparsities_range[1], num=10):
                    set_pruning_rate(model, cur_k)
                    val_accs = []
                    y_offset = 0
                    for dataset_num_classes, test_dataloader in zip(datasets_num_classes, test_dataloaders):
                        val_accs += [ResNetCIFARManager.get_accuracy(model, test_dataloader, device, y_offset)]
                        y_offset += dataset_num_classes
                    avg_val_acc = sum(val_accs) / len(val_accs)
                    
                    avg_accs_in_var_sparsity[f'{cur_k:.2f}'] = avg_val_acc
                    
                    # tb_writer.add_scalars(f'accs/val_accs_{cur_k:.2f}', {k: v for k, v in zip(datasets_name, val_accs)}, iter_index)
                tb_writer.add_scalars(f'accs/avg_val_accs', avg_accs_in_var_sparsity, iter_index)
                
                avg_val_acc = sum(list(avg_accs_in_var_sparsity.values())) / len(list(avg_accs_in_var_sparsity.values()))
                if avg_val_acc > cur_best_acc:
                    cur_best_acc = avg_val_acc
                    torch.save(model, os.path.join(res_save_dir, f'best_model_{cur_k:.2f}.pt'))
                    torch.save(affine_parameters, os.path.join(res_save_dir, f'best_model_{cur_k:.2f}.pt.affine_p'))
                
                affine_parameters_weights = []
                for p in affine_parameters.values():
                    affine_parameters_weights += p[0].detach().view(-1).cpu().numpy().tolist()
                affine_parameters_biases = []
                for p in affine_parameters.values():
                    affine_parameters_biases += p[1].detach().view(-1).cpu().numpy().tolist()
                tb_log.add_histogram(f'affine_parameters_weights', torch.tensor(affine_parameters_weights), iter_index)
                tb_log.add_histogram(f'affine_parameters_biases', torch.tensor(affine_parameters_biases), iter_index)
                
                torch.save(model, os.path.join(res_save_dir, f'last_model.pt'))
                torch.save(affine_parameters, os.path.join(res_save_dir, f'last_model.pt.affine_p'))

                pre_acc_file_path = glob.glob(
                    os.path.join(res_save_dir, f'val_acc_*.json'))
                if len(pre_acc_file_path) > 0:
                    os.remove(pre_acc_file_path[0])
                write_json(os.path.join(res_save_dir, f'val_acc_{cur_best_acc:.4f}.json'),
                        avg_accs_in_var_sparsity, backup=False)
