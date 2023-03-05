import argparse
import ray
from ray import tune

# from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.callbacks import DefaultCallbacks
# from ray.tune.registry import get_trainable_cls
# from ray.tune.integration.wandb import WandbTrainableMixin

from ray.air.callbacks.wandb import WandbLoggerCallback

from coverage_2 import CoverageEnv
from model import AdversarialModel
from trainer_2 import MultiPPOTrainer
from action_distribution import TorchHomogeneousMultiActionDistribution


# class _WrappedTrainable(WandbTrainableMixin, MultiPPOTrainer):
#     _name = MultiPPOTrainer.__name__ if hasattr(MultiPPOTrainer, "__name__") \
#         else "wrapped_trainable"
    
#     def __init__(self, config=None, logger_creator=None):
#             super().__init__(config=config, logger_creator=logger_creator)

class CustomCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        episode.user_data["rewards"] = {}
    
    def on_episode_step(self, worker, base_env, episode, **kwargs):
        ep_info = episode.last_info_for()
        if ep_info is not None and ep_info:
            for a_id, reward in ep_info['rewards'].items():
                episode.user_data["rewards"][a_id] = episode.user_data['rewards'].get(a_id, 0) + reward
    
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        # episode.custom_metrics['rewards'] = {a_id: sum(values) for a_id, values in episode.user_data['rewards'].items()}
        episode.custom_metrics['agent_reward_max'] = max(episode.user_data['rewards'].values())
        episode.custom_metrics['agent_reward_min'] = min(episode.user_data['rewards'].values())
        episode.custom_metrics['agent_reward_mean'] = sum(episode.user_data['rewards'].values()) / len(episode.user_data['rewards'].keys())

def train(args):
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
        callbacks=[WandbLoggerCallback(
            project="r255-marl",
        )],
        stop={"training_iteration": args.training_iteration},
        config={
            "callbacks": CustomCallbacks,
            "output":'tmp/',
            "framework": "torch",
            "env": "coverage",
            "use_gae": True,
            'use_critic': True,
            "kl_coeff": 0.5,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
            "train_batch_size": args.train_batch_size,
            "rollout_fragment_length": args.rollout_fragment_length,
            "sgd_minibatch_size": args.sgd_minibatch_size,
            "num_sgd_iter": args.num_sgd_iter,
            "num_gpus": args.num_gpus,
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
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
                'state_size': 8,
                'collapse_state': False,
                'termination_no_new_coverage': 10,
                'max_episode_len': 154, # 16 * 16 * 0.6
                'n_agents': 4,
                'disabled_teams_step': [True, False],
                'disabled_teams_comms': [True, False],
                'min_coverable_area_fraction': 0.6,
                'map_mode': 'random',
                'reward_annealing': 0.0,
                'communication_range': 8.0,
                'ensure_connectivity': True,
                'reward_type': 'semi_cooperative', #semi_cooperative/cooperative
                'episode_termination': 'early', # early/fixed/default
                'operation_mode': 'coop_only',
            }
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RLLib multi-agent with differentiable communication channel."
    )

    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=1000)
    parser.add_argument('--rollout_fragment_length', type=int, default=125)
    parser.add_argument('--num_sgd_iter', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_envs_per_worker', type=int, default=1)
    parser.add_argument('--training_iteration', type=int, default=100)
    parser.add_argument('--sgd_minibatch_size', type=int, default=100)
    parser.add_argument('--num_envs_per_worker', type=int, default=1)
    args = parser.parse_args()
    train(args)