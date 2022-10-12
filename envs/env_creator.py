import torch

from envs.IBGym_mod_envs import IBGymQ, IBGymModded
from state_quantization.transforms import NormalizeTransform


def env_creator(device, steps_per_episode=1000):
    model_path = 'state_quantization/model'

    model = torch.load(model_path).to(device)
    model.eval()
    model.look_ahead = 0
    # model.share_memory()
    normalize_dataset = NormalizeTransform.load('state_quantization/NormalizeInputConfigs.pkl')
    normalize_dataset.to(device)
    return IBGymQ(q_model=model, device=device, setpoint=70, reward_type='classic', action_type='discrete',
                  observation_type='include_past',
                  reset_after_timesteps=steps_per_episode, n_past_timesteps=model.seq_len,
                  normalize_transformer=normalize_dataset)


def ibgym_env_creator(steps_per_episode=1000):
    return IBGymModded(setpoint=70, reward_type='classic', action_type='discrete', observation_type='include_past',
                       reset_after_timesteps=steps_per_episode, n_past_timesteps=30)
