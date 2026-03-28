from ..scenario.scenario import Scenario
from .exp_tracker import OfflineTrainTracker, OnlineDATracker
from ..alg.alg import get_algorithm
from ..scenario.registery import static_scenario_registery
from ..alg.alg_models_manager import get_required_alg_models_manager_cls
from ..scenario.build import build_scenario_manually_v2, build_scenario_manually
from ..alg.ab_algorithm import ABOfflineTrainAlgorithm, ABOnlineDAAlgorithm
from .util import set_random_seed, ParameterGrid

import random
import torch
import os
import copy
import shutil
import time


def get_cur_time_str():
    """Get the current timestamp string like '20210618123423' which contains date and time information.

    Returns:
        str: Current timestamp string.
    """
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))


def get_num_existing_logs(log_dir):
    if not os.path.exists(log_dir):
        return 0

    res = 0
    for p1 in os.listdir(log_dir):
        p1 = os.path.join(log_dir, p1)
        
        if not os.path.isdir(p1):
            continue
        
        for p2 in os.listdir(p1):
            p2 = os.path.join(p1, p2)
            
            res += 1
            
    return res


def offline_train(#exp_file,
                  alg_name, scenario_name, models, alg_models_manager,
                  hparams, device, random_seed,res_save_dir):
    
    set_random_seed(random_seed)
    
    cur_time = get_cur_time_str()
    #num_existing_logs = get_num_existing_logs(os.path.join(os.path.dirname(exp_file), f'logs'))
    
    #res_save_dir = os.path.join(os.path.dirname(exp_file), f'logs/{cur_time[0: 8]}/{999999 - num_existing_logs:06d}-{cur_time[8:]}')
    
    alg, supported_tasks_type = get_algorithm(alg_name, 'offline', models, alg_models_manager, hparams, device, random_seed, res_save_dir)
    assert isinstance(alg, ABOfflineTrainAlgorithm)
    
    #required_alg_model_manager_cls = get_required_alg_models_manager_cls(alg_name, 'offline')
    # print(required_alg_model_manager_cls)
    #assert isinstance(alg_models_manager, required_alg_model_manager_cls)

    scenario_def = static_scenario_registery[scenario_name]
    scenario: Scenario = build_scenario_manually_v2(**scenario_def)[0]

    if scenario.get_task_type() not in supported_tasks_type:
        raise RuntimeError(f'Algorithm `{alg_name}` doesn\'t support `{scenario.task_type}` task!')

    exp_tracker = OfflineTrainTracker(res_save_dir)
    exp_tracker.set_scenario(scenario)
    exp_tracker.set_alg(alg)
    exp_tracker.set_models(models)

    scenario.set_permission('offline', 'user')
    alg.train(scenario, exp_tracker)
    
    # finished ideally
    os.rename(res_save_dir, res_save_dir + '-finished')

from benchmark.longtail import *
def offline_da_hp_search(alg_name, scenario_name, models, alg_models_manager,
                         base_hparams, hparams_search_range, random_search_times, legal_hparams_condition, reasonable_da_acc_improvement,
                         device, random_seed, res_save_dir):
    
    set_random_seed(random_seed)
    
    scenario_def = static_scenario_registery[scenario_name]

    scenario: Scenario = build_scenario_manually(**scenario_def)[0]

    search_i = 0
    hparams_metrics_list = []
    result_path = os.path.join(res_save_dir, 'hp_search_results.json')
    
    covered_hparams_list = []
    for covered_hparams in ParameterGrid(hparams_search_range):
        covered_hparams_list += [covered_hparams]
    if random_search_times > 0:
        random.shuffle(covered_hparams_list)
    
    best_hparams = None
    best_metric = 0.

    scaling_y=np.array([])
    for covered_hparams in covered_hparams_list:
        if not legal_hparams_condition(covered_hparams):
            continue

        if random_search_times > 0 and search_i >= random_search_times:
            break
        
        print(f'{search_i}-th trial, hparams: {covered_hparams}')
        
        cur_hparams = copy.deepcopy(base_hparams)
        for k, v in covered_hparams.items():
            t = cur_hparams
            for k_unit in k.split('.')[0: -1]:
                t = cur_hparams[k_unit]
            t[k.split('.')[-1]] = v
        cur_models = alg_models_manager.get_deepcopied_models(models)

        if os.path.exists(os.path.join(res_save_dir, str(search_i))):
            #print(os.path.join(res_save_dir, str(search_i)))
            try:
                shutil.rmtree(os.path.join(res_save_dir, str(search_i)))
            except:
                pass

        
        alg, supported_tasks_type = get_algorithm(alg_name, 'online', cur_models, alg_models_manager, cur_hparams, device, random_seed,
                                                  os.path.join(res_save_dir, str(search_i)))
        assert isinstance(alg, ABOnlineDAAlgorithm)

        if scenario.get_task_type() not in supported_tasks_type:
            raise RuntimeError(f'Algorithm `{alg_name}` doesn\'t support `{scenario.task_type}` task!')

        exp_tracker = OnlineDATracker(os.path.join(res_save_dir, str(search_i)))
        exp_tracker.set_scenario(scenario)
        exp_tracker.set_alg(alg)
        exp_tracker.set_models(models)

        skip_this_trial = False
        
        exp_tracker.before_first_da()
        #print(scenario.get_target_domains_order())
        #print(ghi)
        for target_domain_name in scenario.get_target_domains_order():
            dg_acc, before_da_acc = exp_tracker.before_each_da()
            ################
            alg.adapt_to_target_domain(scenario.get_one_da_sub_scenario_for_alg(target_domain_name), exp_tracker)

            after_da_acc = exp_tracker.after_each_da()

            if after_da_acc - before_da_acc < reasonable_da_acc_improvement and dg_acc - after_da_acc > reasonable_da_acc_improvement:
                print(f'warning! current hparams are too bad ({before_da_acc:.4f} -> {after_da_acc:.4f} '
                      f'({(after_da_acc - before_da_acc):.4f} ↑), (dg acc {dg_acc:.4f})) so we early stop this search trial.')
                skip_this_trial = True
                break
            
        if skip_this_trial:
            # exp_tracker.close()
            continue
            
        avg_after_acc = exp_tracker.after_last_da()
        # exp_tracker.close()
        print(f'avg. acc: {avg_after_acc:.4f}')
        scaling_y = np.append(scaling_y, avg_after_acc)
        hparams_metrics_list += [
            [os.path.join(res_save_dir, str(search_i)), covered_hparams, avg_after_acc]]
        hparams_metrics_list.sort(key=lambda i: i[-1], reverse=True)
        
        import json
        with open(result_path, 'w') as f:
            json.dump(hparams_metrics_list, f, indent=2)
        
        if avg_after_acc > best_metric:
            best_metric = avg_after_acc
            best_hparams = copy.deepcopy(cur_hparams)

        search_i += 1

    if len(hparams_metrics_list) == 0:
        print('No reasonable hparams found. Check the search space and try again!')
        return
    
    #os.rename(hparams_metrics_list[0][0], hparams_metrics_list[0][0] + '-best')
    hparams_metrics_list[0][0] = hparams_metrics_list[0][0] + '-best'
    
    print('search finished. top-5 results:')
    print(f'(full results is in {result_path})')
    for r, p, m in hparams_metrics_list[0: 5]:
        print(f'avg. acc: {m:.4f} in {r}')

    from scipy.optimize import curve_fit
    scaling_x = np.array([ 40, 80, 120,160, 200,240])
    params, covariance = curve_fit(scaling_function, scaling_x, scaling_y,maxfev=1000)
    a, b, c = params
    #from example.exp.online.cua import scaling_logit
    print(scaling_function(5000,a, b, c))
    #scaling_logit=np.append(scaling_logit, scaling_function(5000,a, b, c))

        
    with open(result_path, 'w') as f:
        json.dump(hparams_metrics_list, f, indent=2)

    return best_hparams,scaling_function(5000,a, b, c)


def online_da(alg_name, scenario_name, models, alg_models_manager,
              hparams, device, random_seed, res_save_dir):
    
    set_random_seed(random_seed)
    
    alg, supported_tasks_type = get_algorithm(alg_name, 'online', models, alg_models_manager, hparams, device, random_seed, res_save_dir)
    assert isinstance(alg, ABOnlineDAAlgorithm)
    
    scenario_def = static_scenario_registery[scenario_name]
    scenario: Scenario = build_scenario_manually_v2(**scenario_def)[1]

    if scenario.get_task_type() not in supported_tasks_type:
        raise RuntimeError(f'Algorithm `{alg_name}` doesn\'t support `{scenario.task_type}` task!')

    exp_tracker = OnlineDATracker(res_save_dir)
    exp_tracker.set_scenario(scenario)
    exp_tracker.set_alg(alg)
    exp_tracker.set_models(models)

    exp_tracker.before_first_da()
    for target_domain_name in scenario.get_target_domains_order():
        exp_tracker.before_each_da()
        
        alg.adapt_to_target_domain(scenario.get_one_da_sub_scenario_for_alg(target_domain_name), exp_tracker)

        exp_tracker.after_each_da()
    exp_tracker.after_last_da()
    