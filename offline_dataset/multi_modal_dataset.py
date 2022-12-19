import time
from shutil import rmtree
from os import makedirs
from os.path import join, exists, isdir, dirname

from envs.env_creator import ibgym_env_creator
from offline_dataset.convert_datset import save_numpy
from offline_dataset.dataset_creater import GymParallelSampler
from state_quantization.transforms import multi_model_quantize_transforms_creator


def generate_multimodal_dataset(model_names, model_paths, episodes, pool=None, steps_per_episode=1000, workers=8,
                                root_path='tmp'):
    writer_path = join(root_path, "dataset_creator_tmp")
    q_transform_kwargs = {'device': 'cpu', 'keys': ['obs', 'new_obs'], 'reshape': (steps_per_episode, -1, 6),
                          'model_paths': model_paths}
    env_kwargs = {'steps_per_episode': steps_per_episode}

    if exists(writer_path) and isdir(writer_path):
        rmtree(writer_path)

    start = time.time()
    parallel_sampler = GymParallelSampler(env_creator=ibgym_env_creator, path=writer_path, episodes=episodes,
                                          workers=workers, env_kwargs=env_kwargs, reward_threshold=None,
                                          buffer_transform=multi_model_quantize_transforms_creator,
                                          buffer_transform_kwargs=q_transform_kwargs,
                                          policy=None, pool=pool)
    parallel_sampler.sample()
    end = time.time()
    print(end - start)
    merged_datasets = parallel_sampler.create_merged_dataset()
    for model in model_names:
        save_path = join(root_path, "offline_rl_trajectories", model, "rl_dataset.npy")
        makedirs(dirname(save_path), exist_ok=True)
        merged_dataset = {
            'actions': merged_datasets['actions'],
            'rewards': merged_datasets['rewards'],
            'dones': merged_datasets['rewards'],
            'obs': merged_datasets[f'{model}_obs'],
            'new_obs': merged_datasets[f'{model}_new_obs']
        }
        save_numpy(save_path, merged_dataset)

    rmtree(writer_path)
