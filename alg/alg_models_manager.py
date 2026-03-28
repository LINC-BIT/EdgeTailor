from .algs import *
from .registery import static_offline_alg_model_manager_registery


def get_required_alg_models_manager_cls(alg_name, stage):
    if stage == 'offline':
        return static_offline_alg_model_manager_registery[alg_name]
