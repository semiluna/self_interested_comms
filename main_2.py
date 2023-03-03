import argparse
import ray
from ray import tune

# from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from coverage_2 import CoverageEnv
from model import AdversarialModel
from trainer_2 import MultiPPOTrainer
from action_distribution import TorchHomogeneousMultiActionDistribution

class CustomCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        episode.user_data["rewards"] = {}
        episode.user_data["coverable_area"] = 0.0
    
    def on_episode_step(self, worker, base_env, episode, **kwargs):
        ep_info = episode.last_info_for()
        if ep_info is not None and ep_info:
            for a_id, reward in ep_info['rewards'].items():
                episode.user_data["rewards"][a_id] = episode.user_data['rewards'].get(a_id, []) + [reward]
        episode.user_data['coverable_area'] = ep_info['coverable_area']
    
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        # episode.custom_metrics['rewards'] = {a_id: sum(values) for a_id, values in episode.user_data['rewards'].items()}
        episode.custom_metrics['coverable_area'] = episode.user_data['coverable_area']

def train():
    ray.init()

    register_env("coverage", lambda config: CoverageEnv(config))
    ModelCatalog.register_custom_model("adversarial", AdversarialModel)
    ModelCatalog.register_custom_action_dist(
        "hom_multi_action", TorchHomogeneousMultiActionDistribution
    )

    tune.run(
        MultiPPOTrainer,
        checkpoint_freq=1,
        keep_checkpoints_num=1,
        local_dir="/tmp",
        # callbacks=[WandbLoggerCallback(
        #     project="",
        #     api_key_file="",
        #     log_config=True
        # )],
        stop={"training_iteration": 30},
        config={
            # "callbacks": CustomCallbacks,
            "framework": "torch",
            "env": "coverage",
            "use_gae": True,
            "kl_coeff": 0.5,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
            "train_batch_size": 32,
            "rollout_fragment_length": 32,
            "sgd_minibatch_size": 32,
            "num_sgd_iter": 16,
            "num_gpus": 0,
            "num_workers": 4,
            "num_envs_per_worker": 1,
            "lr": 5e-4,
            "gamma": 0.9,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "model": {
                "custom_model": "adversarial",
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    'graph_layers': 1,
                    'graph_tabs': 2,
                    'graph_edge_features': 1,
                    'graph_features': 32,
                    'cnn_filters': [[8, [4, 4], 2], [16, [4, 4], 2], [32, [3, 3], 2]],
                    'value_cnn_filters': [[8, [4, 4], 2], [16, [4, 4], 2], [32, [4, 4], 2]],
                    'value_cnn_compression': 32,
                    'cnn_compression': 32,
                    'pre_gnn_mlp': [64, 128, 32],
                    'gp_kernel_size': 16,
                    'graph_aggregation': 'sum',
                    'relative': True,
                    'activation': 'relu',
                    'freeze_coop': False,
                    'freeze_greedy': False,
                    'freeze_coop_value': False,
                    'freeze_greedy_value': False,
                    'cnn_residual': False,
                    'agent_split': 1,
                    'greedy_mse_fac': 0.0,
                },
            },
            "env_config": {
                'world_shape': [16, 16],
                'state_size': 16,
                'collapse_state': False,
                'termination_no_new_coverage': 10,
                'max_episode_len': 345, # 24*24*0.6
                'n_agents': 4,
                'disabled_teams_step': [True, False],
                'disabled_teams_comms': [True, False],
                'min_coverable_area_fraction': 0.6,
                'map_mode': 'random',
                'reward_annealing': 0.0,
                'communication_range': 16.0,
                'ensure_connectivity': True,
                'reward_type': 'semi_cooperative', #semi_cooperative/cooperative
                'episode_termination': 'early', # early/fixed/default
                'operation_mode': 'coop_only',
            },
        }
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="RLLib multi-agent with shared NN demo."
    # )

    # args = parser.parse_args()
    train()