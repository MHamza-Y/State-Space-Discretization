import torch

from envs.IBGym_mod_envs import IBGymQ, IBGymModded, IBGymModelQ
from state_quantization.transforms import NormalizeTransform, LSTMQuantize


def env_creator(model_path, device='cpu', steps_per_episode=1000):
    reshape = (1, -1, 6)
    model = torch.load(model_path).to(device)
    model.eval()
    model.set_look_ahead(0)
    normalize_dataset = NormalizeTransform.load('state_quantization/NormalizeInputConfigs.pkl')
    normalize_dataset.to(device)
    lstm_quantize = LSTMQuantize(model=model, normalize_transformer=normalize_dataset, reshape=reshape)
    return IBGymQ(device=device, setpoint=70, reward_type='classic', action_type='discrete',
                  observation_type='include_past',
                  reset_after_timesteps=steps_per_episode, n_past_timesteps=model.get_seq_len(),
                  lstm_quantize=lstm_quantize)


def IBGymModelQ_creator(model_path, device='cpu', steps_per_episode=1000):
    reshape = (1, -1, 6)
    model = torch.load(model_path).to(device)
    model.eval()
    model.set_look_ahead(0)
    normalize_dataset = NormalizeTransform.load('state_quantization/NormalizeInputConfigs.pkl')
    normalize_dataset.to(device)
    lstm_quantize = LSTMQuantize(model=model, normalize_transformer=normalize_dataset, reshape=reshape)
    return IBGymModelQ(device=device, setpoint=70, reward_type='classic', action_type='discrete',
                       observation_type='include_past',
                       reset_after_timesteps=steps_per_episode, n_past_timesteps=model.get_seq_len(),
                       lstm_quantize=lstm_quantize)


def ibgym_env_creator(config=None, steps_per_episode=1000):
    return IBGymModded(setpoint=70, reward_type='classic', action_type='discrete', observation_type='include_past',
                       reset_after_timesteps=steps_per_episode, n_past_timesteps=30)


def ibgym_env_creator_rllib(config=None, steps_per_episode=1000):
    return IBGymModded(setpoint=70, reward_type='classic', action_type='discrete', observation_type='classic',
                       reset_after_timesteps=steps_per_episode)
