from typing import Dict, List, Type, Union
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

from torch import nn
import time
import os
import torch
import threading
import sys
import json
import shutil
import pprint
import psutil
import pynvml
import platform
import tqdm
from rich.console import Console
from ..exp.rich_markdown import MyMarkdown as Markdown

from benchmark.exp.alg_model_manager import ABAlgModelsManager
from ..scenario.scenario import Scenario

import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import io
import sys
from benchmark.longtail import jetson

has_gpu = torch.cuda.is_available()
#has_gpu=0
if has_gpu:
    if jetson==0:
       pynvml.nvmlInit()
    #cuda.init()


def get_sys_info():
    machine_info = {
        'CPU': platform.processor() + f' ({psutil.cpu_count()} cores)',
        'OS': platform.platform(),
        'RAM (GB)': int(psutil.virtual_memory().total / 1024**3)
    }

    import torchvision
    env_info = {
        'Python': sys.version.split(' ')[0],
        'PyTorch': torch.__version__,
        'TorchVision': torchvision.__version__,
        
    }
    
    if has_gpu:
        env_info = {**env_info, 'CUDA': torch.version.cuda,
            'CUDNN': torch.backends.cudnn.version()}

        a, b = torch.rand(10).cuda(), torch.rand(10).cuda()
        a = a + b
        
        gpu_info = {}
        gpu_info['driver'] = pynvml.nvmlSystemGetDriverVersion().decode()
        gpu_info['CUDA driver'] = torch.version.cuda
        gpu_info['GPUs'] = []
        used_gpus = []

        for di in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(di)
            total_memory = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**3)
            device_info_desc = f'{di}: {pynvml.nvmlDeviceGetName(handle).decode()} ({total_memory:.1f}GB)'
            
            running_processes_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for pi in running_processes_info:
                pid = getattr(pi, 'pid')

                if os.getpid() == pid:
                    used_gpus += [di]
                    break
                
            gpu_info['GPUs'] += [device_info_desc]
            
        if len(used_gpus) > 0:
            for used_gpu in used_gpus:
                gpu_info['GPUs'][used_gpu] += f' (running on)'
    
    res = dict(machine=machine_info, gpu=gpu_info, env=env_info)
    return res
        

current_process_util = psutil.Process(os.getpid())
def get_process_running_status():
    cpu_info = {
        'RAM (MB)': current_process_util.memory_info().rss / 1024**2,
        'CPU': {

            'usage (%)': current_process_util.cpu_percent()
        }
    }
    res = dict(cpu=cpu_info)
    
    if has_gpu:
        used_vram = 0.
        num_competing_processes = 0
        
        for di in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(di)
            running_processes_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for pi in running_processes_info:
                pid = getattr(pi, 'pid')
                if os.getpid() == pid:
                    used_vram +=float(0 if getattr(pi, 'usedGpuMemory') is None else getattr(pi, 'usedGpuMemory') )
                    num_competing_processes += len(running_processes_info) - 1
                    break
        
        used_vram = used_vram / 1024**2
        gpu_info = {
            'VRAM (MB)': used_vram,
            '# competing processes': num_competing_processes
        }
        res['gpu'] = gpu_info
    
    return res


def get_model_size(model: torch.nn.Module, return_MB=False):
    """Get size of a PyTorch model (default in Byte).

    Args:
        model (torch.nn.Module): A PyTorch model.
        return_MB (bool, optional): Return result in MB (/= 1024**2). Defaults to False.

    Returns:
        int: Model size.
    """
    pid = os.getpid()
    tmp_model_file_path = './tmp-get-model-size-{}-{}.model'.format(pid, time.time())
    torch.save(model, tmp_model_file_path)

    model_size = os.path.getsize(tmp_model_file_path)
    os.remove(tmp_model_file_path)
    
    if return_MB:
        model_size /= 1024**2

    return model_size


def eval_models_perf(models, alg_models_manager: ABAlgModelsManager, a_sample, a_label, device):
    models = alg_models_manager.get_deepcopied_models(models)

    def inference_eval(num_inference: int, num_warmup_inference: int, a_sample):
        if device == 'cuda' or 'cuda' in str(device):
            import torch.cuda
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cur_memory = torch.cuda.max_memory_allocated()
        else:
            cur_memory = current_process_util.memory_info().rss
            
        a_sample = a_sample.to(device)
        
        # warm up
        with torch.no_grad():
            for _ in range(num_warmup_inference):
                alg_models_manager.predict(models, a_sample)
                
        infer_time_list = []
                
        if device == 'cuda' or 'cuda' in str(device):
            with torch.no_grad():
                for _ in range(num_inference):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    alg_models_manager.predict(models, a_sample)
                    e.record()
                    torch.cuda.synchronize()
                    cur_model_infer_time = s.elapsed_time(e) / 1000.
                    infer_time_list += [cur_model_infer_time]

        else:
            with torch.no_grad():
                for _ in range(num_inference):
                    start = time.time()
                    alg_models_manager.predict(models, a_sample)
                    cur_model_infer_time = time.time() - start
                    infer_time_list += [cur_model_infer_time]
                    
        avg_infer_time = sum(infer_time_list) / num_inference
        memory = (torch.cuda.max_memory_allocated() if has_gpu else current_process_util.memory_info().rss) - cur_memory
        
        return avg_infer_time, memory
    
    def training_eval(num_training: int, num_warmup_inference: int, a_sample, a_label):
        if device == 'cuda' or 'cuda' in str(device):
            import torch.cuda
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cur_memory = torch.cuda.max_memory_allocated()
        else:
            cur_memory = current_process_util.memory_info().rss
            
        a_sample = a_sample.to(device)
        a_label = a_label.to(device)
        
        # warm up
        with torch.no_grad():
            for _ in range(num_warmup_inference):
                alg_models_manager.predict(models, a_sample)
                
        training_time_list = []
        
        import torch.optim
        params = []
        for n, _ in models.items():
            model = alg_models_manager.get_model(models, n)
            params += list(model.parameters())
        optimizer = torch.optim.SGD(params, lr=1e-3, weight_decay=1e-4)
                
        if device == 'cuda' or 'cuda' in str(device):
            for _ in range(num_training):
                # forward
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()


                loss = alg_models_manager.forward_to_compute_task_loss(models, a_sample, a_label.long())
                # if isinstance(loss, tuple):
                #     loss = loss[0]
                # elif isinstance(loss, dict):
                #     k = list(loss.keys())[0]
                #     loss = loss[k]

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                
                e.record()
                torch.cuda.synchronize()
                cur_model_infer_time = s.elapsed_time(e) / 1000.
                training_time_list += [cur_model_infer_time]
                
        else:
            with torch.no_grad():
                for _ in range(num_training):
                    start = time.time()
                    
                    loss = alg_models_manager.forward_to_compute_loss(models, a_sample, a_label)
                    
                    loss.backward()
                    optimizer.step()
                    training_time_list += [time.time() - start]
                    
        avg_training_time = sum(training_time_list) / num_training
        memory = (torch.cuda.max_memory_allocated() if has_gpu else current_process_util.memory_info().rss) - cur_memory
        
        return avg_training_time, memory 
    
    a_sample, a_label = torch.tensor(a_sample).unsqueeze(0), torch.tensor(a_label).unsqueeze(0)
    predict_latency, predict_memory = inference_eval(100, 50, a_sample)
    avg_training_time, training_memory = training_eval(100, 50, a_sample, a_label)
    model_size = sum([get_model_size(alg_models_manager.get_model(models, n), True) for n in models.keys()])

    return {
        'model(s) size (MB)': model_size,
        
        'predict latency (bs=1) (ms)': predict_latency * 1e3 * 100,
        'predict memory (bs=1) (MB)': predict_memory / 1024**2,
        
        'training latency (bs=1) (ms)': avg_training_time * 1e3 * 100,
        'training memory (bs=1) (MB)': training_memory / 1024**2,
    }


class OfflineTrainTracker(SummaryWriter):
    def __init__(self, res_save_dir):
        super().__init__(log_dir=os.path.join(res_save_dir, 'tb_log'))
        
        self.res_save_dir = res_save_dir
        
        from ..alg.ab_algorithm import ABOfflineTrainAlgorithm
        self.alg: ABOfflineTrainAlgorithm = None 
        self.scenario: Scenario = None
        self.models = None
        
        self.start_time = None
        self.best_val_acc = 0
        self.best_val_accs = None
        self.time_usage_in_eval = 0
        
        self.rich_console = Console()
        
        self._launch_tensorboard()
        
        entry_file_backup_path = os.path.join(self.res_save_dir, 'entry.py')
        shutil.copyfile(sys.argv[0], entry_file_backup_path)
        
    def _get_obj_impl_path(self, obj: object, get_dir=True):
        import inspect
        if get_dir:
            return os.path.dirname(inspect.getfile(obj.__class__))
        else:
            return inspect.getfile(obj.__class__)
        
    def _launch_tensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', os.path.join(self.res_save_dir, 'tb_log')])
        url = tb.launch()
        print(f'TensorBoard is launched in {url} !')
        
    def set_alg(self, alg):
        self.alg = alg
        hparams_str = pprint.pformat(self.alg.hparams)
        
        backup_dir = os.path.join(self.res_save_dir, 'algorithm')
        alg_impl_dir = self._get_obj_impl_path(alg)
        shutil.copytree(alg_impl_dir, backup_dir)
        
        manager_file_backup_path = os.path.join(self.res_save_dir, 'alg_model_manager.py')
        shutil.copyfile(self._get_obj_impl_path(self.alg.alg_models_manager, False), manager_file_backup_path)
        
        desc_str = f"""## Algorithm: {alg.name}

### Random Seed
{self.alg.random_seed}

### Device
{self.alg.device}

### Result Location
{self.alg.res_save_dir}

### Hyperparameter:
```python
{hparams_str}
```
        """
        
        self.add_text('settings/algorithm', desc_str, 0)
        self.rich_console.print(Markdown(desc_str))
        
    def set_scenario(self, scenario: Scenario):
        self.scenario = scenario

        scenario_config = scenario.get_config()
        del scenario_config['transforms']
        del scenario_config['data_dirs']
        del scenario_config['visualize_dir_path']
        
        with open(os.path.join(self.res_save_dir, 'scenario.json'), 'w') as f:
            json.dump(scenario_config, f, indent=2)
        
        desc_str = f"""## Scenario
```python
{pprint.pformat(scenario_config)}
```
        """
        
        self.add_text('settings/scenario', desc_str, 0)
        self.rich_console.print(Markdown(desc_str))
    
    def set_models(self, models: Dict[str, nn.Module]):        
        models_tag = self.alg.alg_models_manager.get_model_desc(models)
        models_tag = '- ' + '\n- '.join(models_tag.split('\n'))

        a_sample, a_label = self.scenario.get_offline_source_merged_dataset('train')[0][:2]
        #print(self.scenario.get_offline_source_merged_dataset('train')[1])
        model_perf_info = eval_models_perf(models, self.alg.alg_models_manager, a_sample, a_label, self.alg.device)
        model_perf_info = pprint.pformat(model_perf_info)
        #print(das)
        networks_structure_str = ''
        for n in models.keys():
            model_str = str(self.alg.alg_models_manager.get_model(models, n))
            networks_structure_str += f'**{n}**:\n{model_str}'

        desc_str = f"""## Model(s)
{models_tag}
### Performance
```python
{model_perf_info}
```
### Network structure
```
{networks_structure_str}
```
        """
        
        os.makedirs(os.path.join(self.res_save_dir, 'models'))
        with open(os.path.join(self.res_save_dir, 'models/models.txt'), 'w') as f:
            f.write(desc_str)
        
        self.add_text('settings/models', desc_str, 0)
        #self.rich_console.print(Markdown(desc_str))


        
    def pbared(self, iterable, level=1):
        return tqdm.tqdm(iterable, dynamic_ncols=True, leave=level == 1, position=level)
    
    def add_losses(self, losses: Dict[str, float], global_step: int):
        self.add_scalars('running/losses', losses, global_step)
        
    def add_running_perf_status(self, global_step: int):
        sys_status = get_process_running_status()
        #self.add_scalar('running_perf/RAM', sys_status['cpu']['RAM (MB)'], global_step)
        #self.add_scalar('running_perf/CPU_core', sys_status['cpu']['CPU']['# cores'], global_step)
        self.add_scalar('running_perf/CPU_usage', sys_status['cpu']['CPU']['usage (%)'], global_step)
        #self.add_scalar('running_perf/VRAM', sys_status['gpu']['VRAM (MB)'], global_step)
        #self.add_scalar('running_perf/num_competing_processes', sys_status['gpu']['# competing processes'], global_step)
        
    def add_val_accs(self, global_step: int) -> bool:
        start_time = time.time()
        
        val_accs = {}
        for dataset_name, val_dataset in self.scenario.get_offline_source_datasets('val').items():
            val_dataloader = self.scenario.build_dataloader(val_dataset, 256, 4, False, False)
            val_accs[dataset_name] = self.alg.alg_models_manager.get_accuracy(self.alg.models, val_dataloader)
        val_dataset = self.scenario.get_offline_source_merged_dataset('val')
        val_dataloader = self.scenario.build_dataloader(val_dataset, 256, 4, False, False)
        val_accs['overall'] = self.alg.alg_models_manager.get_accuracy(self.alg.models, val_dataloader)
        
        self.time_usage_in_eval += time.time() - start_time

        self.add_scalars('running/val_accs', val_accs, global_step)
        
        if val_accs['overall'] > self.best_val_acc:
            self.best_val_acc = val_accs['overall']
            self.best_val_accs = val_accs
            return True
        return False
        
    def add_models(self):
        models = self.alg.models
        for name, model in models.items():
            model = self.alg.alg_models_manager.get_model(models, name)
            torch.save(model, os.path.join(self.res_save_dir, 'models', f'{name}.pt'))
    
    def start_train(self):
        self.start_time = time.time()
        
        sys_info_desc = pprint.pformat(get_sys_info()).replace('running on', '**running on**')
        desc = f"""## Start Training
        
### Time
{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}

### System Info
```python
{sys_info_desc}
```
        """
        self.add_text('running', desc, 0)
        self.rich_console.print(Markdown(desc))
        
        self.scenario.set_permission('offline', 'user')
        
    def end_train(self):
        time_usage = time.time() - self.start_time - self.time_usage_in_eval
        
        self.scenario.set_permission('offline', 'internal')
        
        test_accs = {}
        for dataset_name, test_dataset in self.scenario.get_offline_source_datasets('test').items():
            test_dataloader = self.scenario.build_dataloader(test_dataset, 256, 0, False, False)
            test_accs[dataset_name] = self.alg.alg_models_manager.get_accuracy(self.alg.models, test_dataloader)
        test_dataset = self.scenario.get_offline_source_merged_dataset('test')
        test_dataloader = self.scenario.build_dataloader(test_dataset, 256, 0, False, False)
        test_accs['overall'] = self.alg.alg_models_manager.get_accuracy(self.alg.models, test_dataloader)
        
        summary = {
            'best_val_acc': self.best_val_acc,
            'best_val_accs': self.best_val_accs,
            'test_acc': test_accs['overall'],
            'test_accs': test_accs,
            'time_usage (h)': time_usage / 3600.
        }
        summary_str = pprint.pformat(summary)

        summary_desc = f"""## Summary
```python
{summary_str}
```
        """
        self.add_text('finished/summary', summary_desc, 0)
        self.add_scalars('finished/test_accs', test_accs, 0)
        self.rich_console.print(Markdown(summary_desc))

        with open(os.path.join(self.res_save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
import matplotlib.pyplot as plt
class OnlineDATracker(SummaryWriter):
    def __init__(self, res_save_dir):
        super().__init__(log_dir=os.path.join(res_save_dir, 'tb_log'))
        
        self.res_save_dir = res_save_dir
        
        from ..alg.ab_algorithm import ABOnlineDAAlgorithm
        self.alg: ABOnlineDAAlgorithm = None 
        self.scenario: Scenario = None
        self.models = None
        
        self.cur_target_domain_index = 0
        self.iteration_index = 0
        self.dg_accs = []
        self.before_da_accs = []
        self.after_da_accs = []
        self.time_usages = []
        self.target_domains_iterator_for_before_da_test = None
        self.target_domains_iterator_for_after_da_test = None
        self.cur_da_start_time = None
        
        self.rich_console = Console()
        
        # tb_thread = threading.Thread(target=self._launch_tensorboard)
        # tb_thread.setDaemon(True)
        # tb_thread.start()
        #self._launch_tensorboard()
        
        entry_file_backup_path = os.path.join(self.res_save_dir, 'entry.py')
        shutil.copyfile(sys.argv[0], entry_file_backup_path)

        self.domain=[]
        self.acc=[]

    def _get_obj_impl_path(self, obj: object, get_dir=True):
        import inspect
        if get_dir:
            return os.path.dirname(inspect.getfile(obj.__class__))
        else:
            return inspect.getfile(obj.__class__)
        
    def _launch_tensorboard(self):
        self.tb = program.TensorBoard()
        self.tb.configure(argv=[None, '--logdir', os.path.join(self.res_save_dir, 'tb_log'), '--bind_all'])
        url = self.tb.launch()
        print(f'TensorBoard is launched in {url} !')
        
    def close(self):
        self.tb.close()
        
    def set_alg(self, alg):
        self.alg = alg
        hparams_str = pprint.pformat(self.alg.hparams)
        
        backup_dir = os.path.join(self.res_save_dir, 'algorithm')
        alg_impl_dir = self._get_obj_impl_path(alg)
        shutil.copytree(alg_impl_dir, backup_dir)
        
        manager_file_backup_path = os.path.join(self.res_save_dir, 'alg_model_manager.py')
        shutil.copyfile(self._get_obj_impl_path(self.alg.alg_models_manager, False), manager_file_backup_path)
        
        desc_str = f"""## Algorithm: {alg.name}

### Random Seed
{self.alg.random_seed}

### Device
{'cpu'}

### Result Location
{self.alg.res_save_dir}

### Hyperparameter:
```python
{hparams_str}
```
        """
        
        self.add_text('settings/algorithm', desc_str, 0)
        self.rich_console.print(Markdown(desc_str))
        
    def set_scenario(self, scenario: Scenario):
        self.scenario = scenario
        self.target_domains_iterator_for_before_da_test = self.scenario.get_target_domains_iterator('test')
        self.target_domains_iterator_for_after_da_test = self.scenario.get_target_domains_iterator('test')

        scenario_config = scenario.get_config()
        del scenario_config['transforms']
        del scenario_config['data_dirs']
        del scenario_config['visualize_dir_path']
        
        with open(os.path.join(self.res_save_dir, 'scenario.json'), 'w') as f:
            json.dump(scenario_config, f, indent=2)
            
        scenario_config['da_mode'] = 'open_set_da'
        scenario_config['domain_occur_period'] = '60min'
        scenario_config['source_datasets_name'] = ['CIFAR10LT', 'SVHNLT']
        scenario_config['target_datasets_order'] = ['STL10', 'MNIST'] * 2
        #scenario_config['source_datasets_name'] = ['CIFAR10LT']
        #scenario_config['target_datasets_order'] = ['CIFAR10']
        desc_str = f"""## Scenario
```python
{pprint.pformat(scenario_config)}
```
        """
        if jetson:
           print(desc_str)
        self.add_text('settings/scenario', desc_str, 0)

        self.rich_console.print(Markdown(desc_str))
    
    def set_models(self, models: Dict[str, nn.Module]):        
        models_tag = self.alg.alg_models_manager.get_model_desc(models)
        models_tag = '- ' + '\n- '.join(models_tag.split('\n'))
        #print(self.scenario.get_merged_source_dataset('train')[0])
        a_sample, a_label = self.scenario.get_merged_source_dataset('train')[0][:2]
        #print('...........................',a_sample,'---------------------', a_label)
        #print(asaf)
        model_perf_info = eval_models_perf(models, self.alg.alg_models_manager, a_sample, a_label, self.alg.device)
        model_perf_info = pprint.pformat(model_perf_info)

        networks_structure_str = ''
        for n in models.keys():
            model_str = str(self.alg.alg_models_manager.get_model(models, n))
            networks_structure_str += f'**{n}**:\n{model_str}\n'

        desc_str = f"""## Model(s)
{models_tag}
### Performance
```python
{model_perf_info}
```
### Network structure
```
{networks_structure_str}
```
        """
        if jetson:
           print(desc_str)
        os.makedirs(os.path.join(self.res_save_dir, 'models'))
        with open(os.path.join(self.res_save_dir, 'models/models.txt'),'w',encoding='utf-8-sig') as f:
            f.write(desc_str)
        
        self.add_text('settings/models', desc_str, 0)

        #sys.stdout = io.TextIOWrapper(Markdown(desc_str), encoding='utf8')
        #self.rich_console.print(Markdown(desc_str))
        
    def before_first_da(self):
        test_accs = {}
        
        for target_domain_index, target_domain_name, target_dataset, target_domain_meta_info \
            in self.scenario.get_target_domains_iterator('test'):
                
            if target_domain_name in test_accs.keys():
                continue
            #print('target_dataset',type(target_dataset))
            test_dataloader_list = [self.scenario.build_dataloader(i, 8, 0, False, False) for i in target_dataset]
            #print(type(test_dataloader), '111111')
            test_accs[target_domain_name] = self.alg.alg_models_manager.get_accuracy(self.alg.models, test_dataloader_list)
            
        for target_domain_index, target_domain_name, target_dataset, target_domain_meta_info \
            in self.scenario.get_target_domains_iterator('test'):
            
            self.dg_accs += [test_accs[target_domain_name]]
            
        desc_str = f"""## Running...
        """
        if jetson:
           print(desc_str)
        self.rich_console.print(Markdown(desc_str))
        
    def before_each_da(self):
        # before acc
        target_domain_index, target_domain_name, target_dataset, target_domain_meta_info = \
            next(self.target_domains_iterator_for_before_da_test)
        
        dataloader_list = [self.scenario.build_dataloader(i, 8, 0, False, False) for i in target_dataset]
        before_da_acc = self.alg.alg_models_manager.get_accuracy(self.alg.models, dataloader_list)
        
        # imgs, labels = next(iter(dataloader))
        # imgs, labels = imgs[0: 4], labels[0: 4]
        # k = f'running_in_{self.cur_target_domain_index}-th_target_domain'
        # self.add_images(f'{k}/samples', imgs, 0)

        self.before_da_accs += [before_da_acc]
        self.cur_da_start_time = time.time()
        self.iteration_index = 0
        self.cur_target_domain_index = target_domain_index
        
        return self.dg_accs[self.cur_target_domain_index], before_da_acc
        
    def in_each_iteration_of_each_da(self):
        k = f'running_perf_in_{self.cur_target_domain_index}-th_target_domain'
        
        #sys_status = get_process_running_status()
        # self.add_scalar(f'{k}/RAM', sys_status['cpu']['RAM (MB)'], self.iteration_index)
        # self.add_scalar(f'{k}/CPU_core', sys_status['cpu']['CPU']['# cores'], self.iteration_index)
        #self.add_scalar(f'{k}/CPU_usage', sys_status['cpu']['CPU']['usage (%)'], self.iteration_index)
        # self.add_scalar(f'{k}/VRAM', sys_status['gpu']['VRAM (MB)'], self.iteration_index)
        
        self.iteration_index += 1
    
    def after_each_da(self):
        time_usage = time.time() - self.cur_da_start_time
        self.cur_da_start_time = None
        self.time_usages += [time_usage]
        
        target_domain_index, target_domain_name, target_dataset, target_domain_meta_info = \
            next(self.target_domains_iterator_for_after_da_test)

        dataloader_list =  [self.scenario.build_dataloader(i, 8, 0, False, False) for i in target_dataset]
        after_da_acc = self.alg.alg_models_manager.get_accuracy(self.alg.models, dataloader_list)
        self.after_da_accs += [after_da_acc]
        
        # print(target_domain_index, target_domain_name, self.before_da_accs[-1], self.after_da_accs[-1])
        desc_str = f'{self.cur_target_domain_index}. {target_domain_name}: ' \
            f'({self.dg_accs[self.cur_target_domain_index]:.4f}) {self.before_da_accs[-1]:.4f} -> {self.after_da_accs[-1]:.4f} ({time_usage:.2f}s)'
        self.rich_console.print(Markdown(desc_str))
        if jetson:
           print(desc_str)

        self.domain.append(self.cur_target_domain_index)
        self.acc.append( round(self.after_da_accs[-1],4))

        self.add_scalars('running/accs', {
            'dg': self.dg_accs[target_domain_index],
            'before_da': self.before_da_accs[-1],
            'after_da': self.after_da_accs[-1],
        }, target_domain_index)
        self.add_scalar('running/time', time_usage, target_domain_index)

        # with open(os.path.join(self.res_save_dir, 'domain.txt'), 'w') as f :
        #     f.write(str(round(self.after_da_accs[-1],4)))



        return after_da_acc
        
    def after_last_da(self):
        # get avg data
        avg_dg_acc = sum(self.dg_accs) / len(self.dg_accs)
        avg_before_acc = sum(self.before_da_accs) / len(self.before_da_accs)
        avg_after_acc = sum(self.after_da_accs) / len(self.after_da_accs)
        avg_time_usage, total_time_usage = sum(self.time_usages) / len(self.time_usages), sum(self.time_usages)
        
        res = {
            'avg_dg_acc': avg_dg_acc,
            'avg_before_acc': avg_before_acc,
            'avg_after_acc': avg_after_acc,
            'avg_time_usage': avg_time_usage
        }
        res_str = pprint.pformat(res)


        desc_str = f"""## Summary
```python
{res_str}
```
        """
        self.add_text('finished/summary', desc_str, 0)
        self.rich_console.print(Markdown(desc_str))
        if jetson:
           print(desc_str)
        with open(os.path.join(self.res_save_dir, 'summary.json'), 'w') as f:
            json.dump(res, f, indent=2)

        '''
        self.domaina=['1:slt10','2:mnist','3:slt10','4:mnist']
        plt.plot(self.domaina, self.acc, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='acc')
        # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
        plt.legend(loc="upper right")
        plt.xlabel('domain_index')
        plt.ylabel('acc')

        plt.ylim(0.0000, 1.0000)
        for i in self.domain:
            plt.text(self.domaina[i], self.acc[i], self.acc[i])
        #plt.show()
        '''
        import pandas as pd
        result = {
            'avg_before_acc': avg_before_acc,
            'after_da_accs': self.after_da_accs,
            'avg_after_acc': avg_after_acc,


        }
        output_file = './summary.csv'
        df = pd.DataFrame([result])
        after_da_accs_columns = pd.DataFrame(df['after_da_accs'].to_list(),
                                             columns=[f'after_da_acc_{i + 1}' for i in range(len(self.after_da_accs))])

        # 将展开后的列和原始 DataFrame 合并
        df = pd.concat([df.drop(columns=['after_da_accs']), after_da_accs_columns], axis=1)

        # 如果文件不存在，写入表头并创建文件
        if not os.path.exists(output_file):
            df.to_csv(output_file, mode='w', index=False, header=True)  # 写入表头
        else:
            df.to_csv(output_file, mode='a', index=False, header=False)  # 追加数据，不写表头

        return avg_after_acc
    
    def add_losses(self, losses: Dict[str, float], global_step: int):
        k = f'running_in_{self.cur_target_domain_index}-th_target_domain'
        self.add_scalars(f'{k}/losses', losses, global_step)
        
    def add_models(self):
        models = self.alg.models
        for name, model_info in models.items():
            model = self.alg.alg_models_manager.get_model(models, name)
            torch.save(model, os.path.join(self.res_save_dir, 'models', f'{name}.pt'))
    