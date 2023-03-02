import yaml
from yaml import FullLoader
import ray

import ray.rllib.algorithms.ppo.ppo as ppo
from ray import tune, air
from ray.rllib.utils import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from ray.rllib.algorithms.algorithm import Algorithm

from coverage_2 import CoverageEnv
from model import AdversarialModel
from action_distribution import TorchHomogeneousMultiActionDistribution
# from trainer import MultiPPOTrainer
from trainer import MultiPPOTorchPolicy

torch, _ = try_import_torch()

class MultiPPOAlgorithm(Algorithm):
    @classmethod
    def get_default_policy_class(cls, config):
        return MultiPPOTorchPolicy

def initialise():
    ray.init()
    register_env('coverage', lambda config: CoverageEnv(config))
    ModelCatalog.register_custom_model('adversarial', AdversarialModel)
    ModelCatalog.register_custom_action_dist('hom_multi_action', TorchHomogeneousMultiActionDistribution)

if __name__ == '__main__':
    with open('coverage.yaml', 'r') as handle:
        config = yaml.safe_load(handle)

    initialise()   

    tune.run(
        MultiPPOAlgorithm,
        config={
            'framework': 'troch',
            'env': 'coverage',
            'model': {
                'custom_model': 'adverisarial',
                'custom_action_dict': 'hom_multi_action'
            },
            'env_config': config['env_config']
        }
    )
