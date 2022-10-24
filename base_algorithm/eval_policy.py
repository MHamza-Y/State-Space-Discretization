import numpy as np
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt


class EvalDiscreteStatePolicy:

    def __init__(self, env_creator, env_kwargs, policy):
        self.policy = policy
        self.eval_rewards_per_epoch = []
        self.eval_new_state_found = []
        self.eval_trajectories = []
        self.env_creator = env_creator
        self.env_kwargs = env_kwargs

    def evaluate(self, epochs, render=False, show_reward_type='mean'):
        self.eval_trajectories = []
        env = self.env_creator(**self.env_kwargs)
        self.eval_rewards_per_epoch = []
        for i in range(epochs):
            episode = {'obs': [], 'next_obs': [], 'rewards': [], 'actions': [], 'info': []}
            state = env.reset()

            new_state_found = False
            done = False
            total_reward = 0
            total_steps = 0
            while not done:
                episode['obs'].append(state)
                if self.policy.state_exist(state):
                    action = self.policy.get_action(state)
                else:
                    action = env.action_space.sample()
                    new_state_found = True

                episode['actions'].append(action)
                state, reward, done, info = env.step(action)
                episode['next_obs'].append(state)
                episode['rewards'].append(reward)
                episode['info'].append(info)

                if render:
                    env.render()

                epochs += 1

                total_reward += reward
                total_steps += 1

            if show_reward_type == 'mean':
                self.eval_rewards_per_epoch.append(total_reward / total_steps)
            else:
                self.eval_rewards_per_epoch.append(total_reward)
            self.eval_trajectories.append(episode)
            self.eval_new_state_found.append(new_state_found)
            clear_output(wait=False)
            print(f'Episode {i} Reward: {self.eval_rewards_per_epoch[-1]} || New State Found: {new_state_found}')
            plt.plot(self.eval_rewards_per_epoch)
            plt.show()
        print(f'Reward Mean: {np.mean(self.eval_rewards_per_epoch)}')
        print(f'Reward std : {np.std(self.eval_rewards_per_epoch)}')
