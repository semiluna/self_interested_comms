from typing import Dict, List, Optional, Type, Union
import numpy as np

import gymnasium as gym

import ray

from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, AgentID
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing

# from ray.rllib.algorithms.trainer_template import build_trainer
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
)

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy

from ray.rllib.utils.torch_utils import apply_grad_clipping, sequence_mask, \
    explained_variance

torch, nn = try_import_torch()

class InvalidActionSpace(Exception):
    """Raised when the action space is invalid"""
    pass

def compute_gae_for_sample_batch(
        policy: Policy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
        episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:
    
    if not isinstance(policy.action_space, gym.spaces.tuple.Tuple):
        raise InvalidActionSpace("Expect tuple action space")

    if not sample_batch[SampleBatch.INFOS].dtype == 'float32':
        batch_infos = SampleBatch.concat_samples([
            SampleBatch({k: [v] for k,v in sample.items()})
            for sample in sample_batch[SampleBatch.INFOS]
        ])
    
    n_agents = len(batch_infos['rewards'][0])
    batches = []
    for agent_id in range(n_agents):
        sample_agent = sample_batch.copy()
        sample_agent['rewards'] = batch_infos['rewards'][:, agent_id]
        sample_agent['actions'] = sample_batch['actions'][:, agent_id]
        sample_agent[SampleBatch.VF_PREDS] = sample_batch[SampleBatch.VF_PREDS][:, agent_id]

        if sample_batch[SampleBatch.TERMINATEDS][-1]:
            last_r = 0.0
        else:
            input_dict = sample_batch.get_single_step_input_dict(
                policy.model.view_requirements, index="last"
            )  

            if policy.config["_enable_rl_module_api"]:
                input_dict = policy._lazy_tensor_dict(input_dict)
                fwd_out = policy.model.forward_exploration(input_dict)
                last_r = fwd_out[SampleBatch.VF_PREDS][-1, agent_id]
            else:
                all_values = policy._value(**input_dict)
                last_r = all_values[agent_id]

        batches.append(compute_advantages(
            sample_agent,
            last_r,
            policy.config['gamma'],
            policy.config['lambda'],
            use_gae=policy.config["use_gae"],
            use_critic=policy.config.get("use_critic", True),
        ))
    
    for k in [
        SampleBatch.REWARDS,
        SampleBatch.VF_PREDS,
        Postprocessing.ADVANTAGES,
        Postprocessing.VALUE_TARGETS,
    ]:
        sample_batch[k] = np.stack([b[k] for b in batches], axis=-1)

    return sample_batch


def ppo_surrogate_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective.
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    logits, state = model(train_batch)
    
    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch["seq_lens"])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch["seq_lens"],
            max_seq_len,
            time_major=model.is_time_major())
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    loss_data = []

    curr_action_dist = dist_class(logits, model)
    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS],
                                  model)
    logps = curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    entropies = curr_action_dist.entropy()

    action_kl = prev_action_dist.kl(curr_action_dist)
    mean_kl = reduce_mean_valid(torch.sum(action_kl, axis=1))

    n_agents = len(train_batch[SampleBatch.VF_PREDS][0])
    for i in range(n_agents):
        logp_ratio = torch.exp(
            logps[:, i] -
            train_batch[SampleBatch.ACTION_LOGP][:, i])

        mean_entropy = reduce_mean_valid(entropies[:, i])

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES][..., i] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES][..., i] * torch.clamp(
                logp_ratio, 1 - policy.config["clip_param"],
                1 + policy.config["clip_param"]))
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if policy.config["use_gae"]:
            prev_value_fn_out = train_batch[SampleBatch.VF_PREDS][..., i]
            value_fn_out = model.value_function()[..., i]
            vf_loss1 = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS][..., i], 2.0)
            vf_clipped = prev_value_fn_out + torch.clamp(
                value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"],
                policy.config["vf_clip_param"])
            vf_loss2 = torch.pow(
                vf_clipped - train_batch[Postprocessing.VALUE_TARGETS][..., i], 2.0)
            vf_loss = torch.max(vf_loss1, vf_loss2)
            mean_vf_loss = reduce_mean_valid(vf_loss)
            total_loss = reduce_mean_valid(
                -surrogate_loss + policy.kl_coeff * action_kl[:, i] +
                policy.config["vf_loss_coeff"] * vf_loss -
                policy.entropy_coeff * entropies[:, i])
        else:
            mean_vf_loss = 0.0
            total_loss = reduce_mean_valid(-surrogate_loss +
                                           policy.kl_coeff * action_kl[:, i] -
                                           policy.entropy_coeff * entropies[:, i])

        # Store stats in policy for stats_fn.
        loss_data.append(
            {
                "total_loss": total_loss,
                "mean_policy_loss": mean_policy_loss,
                "mean_vf_loss": mean_vf_loss,
                "mean_entropy": mean_entropy,
            }
        )

    policy._total_loss = (torch.sum(torch.stack([o["total_loss"] for o in loss_data])),)
    policy._mean_policy_loss = torch.mean(
        torch.stack([o["mean_policy_loss"] for o in loss_data])
    )
    policy._mean_vf_loss = torch.mean(
        torch.stack([o["mean_vf_loss"] for o in loss_data])
    )
    policy._mean_entropy = torch.mean(
        torch.stack([o["mean_entropy"] for o in loss_data])
    )
    policy._vf_explained_var = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS],
        policy.model.value_function())
    policy._mean_kl = mean_kl

    return policy._total_loss


class ValueNetworkMixin:
    """This is exactly the same mixin class as in ppo_torch_policy,
    but that one calls .item() on self.model.value_function()[0],
    which will not work for us since our value function returns
    multiple values. Instead, we call .item() in
    compute_gae_for_sample_batch above.
    """

    def __init__(self, obs_space, action_space, config):
        if config["use_gae"]:

            def value(**input_dict):
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.value_function()[0]

        else:

            def value(*args, **kwargs):
                return 0.0

        self._value = value


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])

def kl_and_loss_stats(policy, train_batch):
    policy.explained_variance = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], policy.model.value_function())

    stats_fetches = {
        "cur_kl_coeff": policy.kl_coeff,
        "cur_lr": torch.tensor(policy.cur_lr, dtype=torch.float64),
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "vf_explained_var": policy.explained_variance,
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
    }

    return stats_fetches

def vf_preds_fetches(policy):
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
    }

MultiPPOTorchPolicy = build_policy_class(
    name="MultiPPOTorchPolicy",
    framework="torch",
    get_default_config=lambda: ray.rllib.algorithms.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=compute_gae_for_sample_batch,
    extra_grad_process_fn=apply_grad_clipping,
    # before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ],
)

# def get_policy_class(config):
#     return MultiPPOTorchPolicy

# MultiPPOTrainer = build_trainer(
#     name="MultiPPO",
#     default_config=ray.rllib.algorithms.ppo.ppo.DEFAULT_CONFIG,
#     validate_config=ray.rllib.algorithms.ppo.ppo.validate_config,
#     default_policy=MultiPPOTorchPolicy,
#     get_policy_class=get_policy_class
# )
