from utils.dl.common.model import get_module
from ....ab_algorithm import ABOfflineTrainAlgorithm
from .....exp.alg_model_manager import ABAlgModelsManager
from .....exp.exp_tracker import OfflineTrainTracker
from .....scenario.scenario import Scenario
from ....registery import algorithm_register, alg_model_manager_register

from schema import Schema
import torch
import tqdm
from torch import nn
from model_fbs import DomainDynamicConv2d, boost_raw_model_with_filter_selection, set_pruning_rate,\
    get_accmu_flops, get_cached_raw_w, get_cached_w, get_l1_reg_in_model, start_accmu_flops, train_only_gate


@alg_model_manager_register(
    name='Ours',
    stage='offline'
)
class OursAlgModelsManager(ABAlgModelsManager):
    pass


@algorithm_register(
    name='Ours',
    stage='offline',
    supported_tasks_type=['Image Classification', 'Object Detection']
)
class Ours(ABOfflineTrainAlgorithm):
    def verify_args(self, models, hparams: dict):
        assert 'main model' in models.keys()
        
        Schema({
            'batch_size': int,
            'num_workers': int,
            
            'sparsity_stages': [float],
            'pruning_layers': [str],
            'model_input_size': (int, int, int, int),
            
            'warmup_lrs': [float],
            'warmup_num_iters': [int],
            'warmup_milestones': [[int]],
            'warmup_gammas': [float],
            
            'lrs': [float],
            'affine_lrs': [float],
            'wds': [float],
            'weight_reg_wds': [float],
            'num_iters': [int],
            'milestones': [[int]],
            'gammas': [float],
            
            'raw_pretrained_model_path': str,
            'resume_sparsity_ckpt_path': str,
            'resume_sparsity': float
        }).validate(hparams)
    
    def train(self, scenario: Scenario, exp_tracker: OfflineTrainTracker):
        model = self.alg_models_manager.get_model(self.models, 'main model')
        model = model.to(self.device)
        
        train_set = scenario.get_offline_source_merged_dataset('train')
        train_loader = scenario.build_dataloader(train_set, self.hparams['batch_size'], self.hparams['num_workers'],
                                                 True, None, True)
        
        exp_tracker.start_train()
        
        # model
        if self.hparams['resume_sparsity_ckpt_path'] == '':
            raw_pretrained_ckpt = self.hparams['raw_pretrained_model_path']
            model = torch.load(raw_pretrained_ckpt).to(self.device)
            raw_model = torch.load(raw_pretrained_ckpt).to(self.device)
            
            affine_parameters = {}
            for name, p in model.named_parameters():
                if p.dim() > 0:
                    affine_parameters[name] = torch.zeros((2, p.size(0))).to(self.device)
                    affine_parameters[name].requires_grad = True
            
            exp_tracker.add_val_accs(0)
            
            pruned_layers = self.hparams['pruning_layers']
            ignore_layers = [layer for layer, m in model.named_modules() if isinstance(m, nn.Conv2d) and layer not in pruned_layers]
            model, conv_bn_map = boost_raw_model_with_filter_selection(model, 0., False, ignore_layers, True, self.hparams['model_input_size'])
        else:
            raw_pretrained_ckpt = self.hparams['raw_pretrained_model_path']
            model = torch.load(raw_pretrained_ckpt).to(self.device)
            raw_model = torch.load(raw_pretrained_ckpt).to(self.device)
            
            pruned_layers = self.hparams['pruning_layers']
            ignore_layers = [layer for layer, m in model.named_modules() if isinstance(m, nn.Conv2d) and layer not in pruned_layers]
            model, conv_bn_map = boost_raw_model_with_filter_selection(model, 0., False, ignore_layers, True, self.hparams['model_input_size'])
            
            model = torch.load(self.hparams['resume_sparsity_ckpt_path'])
            affine_parameters = torch.load(self.hparams['resume_sparsity_ckpt_path'] + '.affine_p')
        
        stage_pbar = exp_tracker.pbared(zip(warmup_lrs, warmup_num_iterses, warmup_milestoneses, warmup_gammas, lrs, affine_lrs, wds, l1_wds, weight_reg_wds, num_iterses, milestoneses, gammas, max_sparsities))
        sparsity_i = 0
        for warmup_lr, warmup_num_iters, warmup_milestones, warmup_gamma, \
            lr, affine_lr, wd, l1_wd, weight_reg_wd, num_iters, milestones, gamma, \
            cur_max_sparsity in stage_pbar:
                
            if self.hparams['resume_sparsity_ckpt_path'] != '' and self.hparams['resume_sparsity'] >= cur_max_sparsity:
                print(f'skip sparsity {cur_max_sparsity}')
                continue
            
            sparsities_range = [0., cur_max_sparsity]
            
            cur_k = sparsities_range[1]
            set_pruning_rate(model, cur_k)
            
            if warmup_lr > 0:
                gate_params = train_only_gate(model)
                import torch.optim
                optimizer = torch.optim.SGD(gate_params, lr=warmup_lr, momentum=0.9, weight_decay=0.)
                import torch.optim.lr_scheduler
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=warmup_milestones, gamma=warmup_gamma)
                pbar = exp_tracker.pbared(range(warmup_num_iters), level=2)
                
                for iter_index in pbar:
                    x, y = next(train_loader)
                    x, y = x.to(self.device), y.to(self.device)
                    
                    task_loss = self.alg_models_manager.forward_to_compute_task_loss(model, x, y)
                    
                    exp_tracker.add_scalar(f'losses/warmup_total_loss_{cur_k:.2f}', task_loss, iter_index)
                    pbar.set_description(f'(warmup) cur_k: {cur_k:.2f} loss: {task_loss:.6f}')
                    
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
            
            iter_pbar = exp_tracker.pbared(num_iters, level=2)
            for iter_index in iter_pbar:
                
                model.train()
                self.alg_models_manager.set_model(self.models, 'main model', model)
                
                total_loss = 0.
                total_task_loss, total_l1_loss, total_weight_reg_loss = 0., 0., 0.
                                
                for sparsity in [
                    sparsities_range[0],
                    random.random() * (sparsities_range[1] - sparsities_range[0]) + sparsities_range[0],
                    random.random() * (sparsities_range[1] - sparsities_range[0]) + sparsities_range[0],
                    sparsities_range[1],
                ]:
                    set_pruning_rate(sparsity)

                    x, y = next(train_loader)
                    x, y = x.to(self.device), y.to(self.device)
                    
                    task_loss = self.alg_models_manager.forward_to_compute_task_loss(self.models, x, y)
                    l1_loss = l1_wd * get_l1_reg_in_model(model)
                    weight_reg_loss = weight_reg_wd * get_weight_reg_loss(raw_model, model, affine_parameters)
                    
                    loss = task_loss + l1_loss + weight_reg_loss
                    
                    total_loss += loss
                    total_task_loss += task_loss
                    total_l1_loss += l1_loss
                    total_weight_reg_loss += weight_reg_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                total_loss /= 4.
                total_task_loss /= 4.
                total_l1_loss /= 4.
                total_weight_reg_loss /= 4.
                
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                # subnetwork sampling
                
                set_pruning_rate(model, random.random() * (sparsities_range[1] - sparsities_range[0]) + sparsities_range[0])

                x, y = next(train_loader)
                x, y = x.to(self.device), y.to(self.device)
                
                for layer in model.modules():
                    if isinstance(layer, DomainDynamicConv2d):
                        layer.static_w = None
                with torch.no_grad():
                    self.alg_models_manager.forward(model, x)
                for layer in model.modules():
                    if isinstance(layer, DomainDynamicConv2d):
                        layer.static_w = layer.cached_w[0].squeeze().detach()
                
                task_loss = self.alg_models_manager.forward_to_compute_task_loss(model, x, y)
                optimizer.zero_grad()
                task_loss.backward()
                optimizer.step()
                    
                for layer in model.modules():
                    if isinstance(layer, DomainDynamicConv2d):
                        layer.static_w = None
                
                exp_tracker.add_scalars('', {'task': total_task_loss, 'subnetwork_task': task_loss, 
                                             'l1': total_l1_loss, 'weight_reg': total_weight_reg_loss, 
                                             'total': total_loss}, iter_index)
                pbar.set_description(f'cur sparsity: {sparsities_range:.2f}, loss: {total_loss:.6f}')
                if iter_index % 10 == 0:
                    exp_tracker.add_running_perf_status(iter_index)
                if iter_index % 500 == 0:
                    met_better_model = exp_tracker.add_val_accs(iter_index)
                    if met_better_model:
                        exp_tracker.add_models()
                        
        exp_tracker.end_train()
