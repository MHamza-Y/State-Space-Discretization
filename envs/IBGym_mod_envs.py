from collections import OrderedDict

import gym
import numpy as np
import torch
from industrial_benchmark_python.IDS import IDS

from state_quantization.transforms import Bin2Dec


class IBGymModded(gym.Env):
    """
    OpenAI Gym Wrapper for the industrial benchmark
    """

    def __init__(self, setpoint, reward_type, action_type, observation_type="classic", reset_after_timesteps=1000,
                 init_seed=None, n_past_timesteps=30):
        """
        Initializes the underlying environment, seeds numpy and initializes action / observation spaces
        as well as other necessary variables
        :param setpoint: determines behavior of industrial benchmark
        :param reward_type: classic / delta - determines whether absolute or change in reward is returned
        :param action_type: discrete / continuous
        :param observation_type: classic / include_past - determines wether single or N state frames used as observation
        :param reset_after_timesteps: how many timesteps can the environment run without resetting
        :param init_seed: seed for numpy to make environment behavior reproducible
        :param n_past_timesteps: if observation type is include_past, this determines how many state frames are used
        """

        # IB environment parameter
        self.setpoint = setpoint

        # initial seeding
        self.init_seed = init_seed
        np.random.seed(self.init_seed)

        # Used to determine whether to return the absolute value or the relative change in the cost function
        self.reward_function = reward_type

        # Used to set an arbitrary limit of how many time steps the environment can take before resetting
        self.reset_after_timesteps = reset_after_timesteps

        # Define what actions and observations can look like
        self.action_type = action_type  # discrete or continuous
        self.observation_type = observation_type  # classic or include_past
        self.n_past_timesteps = n_past_timesteps  # if past should be included - how many steps?

        # variables that will change over the course of a trajectory - only initialized here
        self.IB = None  # the actual IDS Object -> real environment
        self.info = None  # entire markov state, including invisible part
        self.reward = None  # reward value of the current step
        self.delta_reward = None  # alternative reward, showing how much the actual reward changed
        self.smoothed_reward = None  # smoothed reward function following a convex combination of old and new reward
        self.env_steps = None  # how many steps have already been performed in the environment
        self.done = None  # is the trajectory finished
        self.last_action = None  # contains the action taken in the last step
        self.observation = None  # contains the current observation

        # Defining the action space
        if self.action_type == 'discrete':  # Discrete action space with three values per steering (3^3 = 27)
            self.action_space = gym.spaces.Discrete(27)

            # A list of all possible discretized actions
            self.env_action = []
            for v in [-1, 0, 1]:
                for g in [-1, 0, 1]:
                    for s in [-1, 0, 1]:
                        self.env_action.append([v, g, s])

        elif self.action_type == 'continuous':  # Continuous action space for each steering [-1,1]
            self.action_space = gym.spaces.Box(np.array([-1, -1, -1]), np.array([+1, +1, +1]))

        else:
            raise ValueError('Invalid action_type. action_space can either be "discrete" or "continuous"')

        # Defining the observation space -> single frame: [setpoint, velocity, gain, shift, fatigue, consumption]
        single_low = np.array([0, 0, 0, 0, 0, 0])
        single_high = np.array([100, 100, 100, 100, 3000, 3000])

        if self.observation_type == "classic":  # classic only has the current state frame
            self.observation_space = gym.spaces.Box(low=single_low, high=single_high)

        elif self.observation_type == "include_past":  # time embedding: state contains also past N state frames
            low = np.hstack([single_low] * self.n_past_timesteps)
            high = np.hstack([single_high] * self.n_past_timesteps)
            self.observation_space = gym.spaces.Box(low=low, high=high)

        else:
            raise ValueError('Invalid observation_type. observation_type can either be "classic" or "include_past"')

        self.reset()

    def step(self, action):
        """
        performs one step in the environment by taking the specified action and returning the resulting observation
        :param action: the action to be taken
        :return: the new observation
        """

        # when the done flag has been set and the user still calls step, we want to at least reset the environment
        if self.done:
            self.reset()

        # keep the current action around for potential rendering
        self.last_action = action

        # Executing the action and saving the observation
        if self.action_type == 'discrete':
            self.IB.step(self.env_action[action])  # for discrete actions, we expect the action's index
        elif self.action_type == 'continuous':
            self.IB.step(action)  # in the continuous case, we expect the entire three dimensional action

        # update observation representation
        return_observation = self._update_observation()

        # Calculating both the relative reward (improvement or decrease) and updating the absolute reward
        new_reward = -self.IB.state['cost']
        self.delta_reward = new_reward - self.reward  # positive when improved
        self.reward = new_reward

        # Due to the very high stochasticity a smoothed reward function can be easier to follow visually
        self.smoothed_reward = 0.9 * self.smoothed_reward + 0.1 * self.reward

        # Stopping condition
        self.env_steps += 1
        if self.env_steps >= self.reset_after_timesteps:
            self.done = True

        # Two reward functions are available:
        # 'classic' which returns the original cost and
        # 'delta' which returns the change in the cost function w.r.t. the previous cost
        if self.reward_function == 'classic':
            return_reward = self.reward
        elif self.reward_function == 'delta':
            return_reward = self.delta_reward
        else:
            raise ValueError('Invalid reward function specification. "classic" for the original cost function'
                             ' or "delta" for the change in the cost fucntion between steps.')

        self.info = self._markovian_state()  # entire markov state - not all info is visible in observations
        return return_observation, return_reward, self.done, self.info

    def reset(self):
        """
        resets environment
        :return: first observation of fresh environment
        """

        # ensure reproducibility, but still use different env / seed on every reset
        self.IB = IDS(self.setpoint, inital_seed=self.init_seed)
        self.init_seed = np.random.randint(0, 100000)

        # if multiple timesteps in a single observation (time embedding), need list
        if self.observation_type == "include_past":
            self.observation = []

        return_observation = self._update_observation()

        self.info = self._markovian_state()
        self.reward = -self.IB.state['cost']

        # Alternative reward that returns the improvement or decrease in the cost function
        # If the cost function improves/decreases, the reward is positive
        # If the cost function deteriorates/increases, the reward is negative
        # e.g.: -400 -> -450 = delta_reward of -50
        self.delta_reward = 0

        # smoother reward function for monitoring the agent & environment with lower variance
        # Updates with a convex combination of old and new cost
        self.smoothed_reward = self.reward

        # used to set the self.done variable - If larger than self.reset_after_timesteps, the environment resets
        self.env_steps = 0

        # whether or not the trajectory has ended
        self.done = False

        return return_observation

    def render(self, mode='human'):
        """
        prints the current reward, state, and last action taken
        :param mode: not used, needed to overwrite the abstract method though
        """

        print('Reward:', self.reward, 'State (v,g,s):', self.IB.visibleState()[1:4], 'Action: ', self.last_action)

    def _update_observation(self):
        """
        gets the new observation from the IDS environment and updates own representation as part of the step method
        :return: the new observation representation
        """

        # when the observation type is classic, an observation consists of a single state frame
        if self.observation_type == "classic":
            self.observation = self.IB.visibleState()[:-2]
            return_observation = self.observation

        # when the observation type is include_past, an observation consists of self.n_past_timesteps state frames
        elif self.observation_type == "include_past":
            single_state = self.IB.visibleState()[:-2]

            # when the env has just been reset, observation is an empty list. Otherwise it containes
            # self.n_past_timesteps state frames, and we remove the oldest so that we have room for a new one
            if len(self.observation) > 0:
                self.observation.pop()

            self.observation.insert(0, single_state)  # insert new observation at the beginning

            # when the env has just been created, there aren't self.n_past_timesteps state frames available yet
            # thus, we repeat the oldes (only) state frame self.n_past_timesteps times
            if len(self.observation) < self.n_past_timesteps:
                oldest_obs = self.observation[-1]
                oldest_repeated = [oldest_obs for _ in range(self.n_past_timesteps - len(self.observation))]
                self.observation = self.observation + oldest_repeated

            return_observation = np.hstack(self.observation)  # return observation is a single flattened numpy.ndarray

        else:
            raise ValueError('Invalid observation_type. observation_type can either be "classic" or "include_past"')

        return return_observation

    def _markovian_state(self):
        """
        get the entire markovian state for debugging purposes
        :return: markov state as a dctionary
        """
        markovian_states_variables = ['setpoint', 'velocity', 'gain', 'shift', 'fatigue', 'consumption',
                                      'op_cost_t0', 'op_cost_t1', 'op_cost_t2', 'op_cost_t3', 'op_cost_t4',
                                      'op_cost_t5', 'op_cost_t6', 'op_cost_t7', 'op_cost_t8', 'op_cost_t9',
                                      'ml1', 'ml2', 'ml3', 'hv', 'hg']

        markovian_states_values = [self.IB.state['p'],
                                   self.IB.state['v'],
                                   self.IB.state['g'],
                                   self.IB.state['h'],
                                   self.IB.state['f'],
                                   self.IB.state['c'],
                                   self.IB.state['o'][0],
                                   self.IB.state['o'][1],
                                   self.IB.state['o'][2],
                                   self.IB.state['o'][3],
                                   self.IB.state['o'][4],
                                   self.IB.state['o'][5],
                                   self.IB.state['o'][6],
                                   self.IB.state['o'][7],
                                   self.IB.state['o'][8],
                                   self.IB.state['o'][9],
                                   self.IB.state['gs_domain'],
                                   self.IB.state['gs_sys_response'],
                                   self.IB.state['gs_phi_idx'],
                                   self.IB.state['hv'],
                                   self.IB.state['hg']]

        info = OrderedDict(zip(markovian_states_variables, markovian_states_values))
        return info


class IBGymModelQ(IBGymModded):
    def __init__(self, lstm_quantize, device='cpu', **kwargs):
        self.device = device
        self.lstm_quantize = lstm_quantize
        self.bin2dec = Bin2Dec()
        self.observation = None
        self.model_out = None
        self.last_observation = None
        self.v = 0
        self.g = 0
        self.h = 0
        super().__init__(**kwargs)

    def reset(self):
        """
                resets environment
                :return: first observation of fresh environment
                """

        # ensure reproducibility, but still use different env / seed on every reset
        self.IB = IDS(self.setpoint, inital_seed=self.init_seed)
        self.init_seed = np.random.randint(0, 100000)

        # if multiple timesteps in a single observation (time embedding), need list
        if self.observation_type == "include_past":
            self.observation = []

        return_observation = self._update_observation()

        self.info = self._markovian_state()
        self.reward = -self.IB.state['cost']

        # Alternative reward that returns the improvement or decrease in the cost function
        # If the cost function improves/decreases, the reward is positive
        # If the cost function deteriorates/increases, the reward is negative
        # e.g.: -400 -> -450 = delta_reward of -50
        self.delta_reward = 0

        # smoother reward function for monitoring the agent & environment with lower variance
        # Updates with a convex combination of old and new cost
        self.smoothed_reward = self.reward

        # used to set the self.done variable - If larger than self.reset_after_timesteps, the environment resets
        self.env_steps = 0

        # whether or not the trajectory has ended
        self.done = False
        self.last_observation = return_observation
        discrete_obs = self.lstm_quantize(return_observation)[0]
        self.model_out = self.lstm_quantize.get_continuous_output()
        self.v = self.last_observation[1]
        self.g = self.last_observation[2]
        self.h = self.last_observation[3]
        return discrete_obs

    def get_fatigue_consumption(self, model_output):
        zero_padding = torch.zeros((model_output.shape[0], model_output.shape[1], 4)).to(self.device)
        model_output = torch.cat((zero_padding, model_output), dim=-1)
        transformed_output = self.lstm_quantize.normalize_transformer.inverse_transform(model_output)[0][
            0].detach().cpu().numpy()
        return transformed_output[-2], transformed_output[-1]

    def step(self, action):
        if self.done:
            self.reset()

        delta = self.env_action[action]
        self.last_observation = np.roll(self.last_observation, 6)

        self.v = np.clip(self.v + delta[0], 0., 100.)
        self.g = np.clip(self.g + 10 * delta[1], 0., 100.)
        self.h = np.clip(self.h + 5.75153434 * delta[2], 0., 100.)
        fatigue, consumption = self.get_fatigue_consumption(self.model_out)
        self.last_observation[0] = self.setpoint
        self.last_observation[1] = self.v
        self.last_observation[2] = self.g
        self.last_observation[3] = self.h
        self.last_observation[4] = fatigue
        self.last_observation[5] = consumption

        discrete_obs = self.lstm_quantize(self.last_observation)[0]
        self.model_out = self.lstm_quantize.get_continuous_output()
        reward = - consumption - 3 * fatigue

        # Stopping condition
        self.env_steps += 1
        if self.env_steps >= self.reset_after_timesteps:
            self.done = True
        return discrete_obs, reward, self.done, None


class IBGymQ(IBGymModded):

    def __init__(self, lstm_quantize, device='cpu', **kwargs):
        self.device = device
        self.lstm_quantize = lstm_quantize
        self.bin2dec = Bin2Dec()
        super().__init__(**kwargs)

    def reset(self):
        """
                resets environment
                :return: first observation of fresh environment
                """

        # ensure reproducibility, but still use different env / seed on every reset
        self.IB = IDS(self.setpoint, inital_seed=self.init_seed)
        self.init_seed = np.random.randint(0, 100000)

        # if multiple timesteps in a single observation (time embedding), need list
        if self.observation_type == "include_past":
            self.observation = []

        return_observation = self._update_observation()

        self.info = self._markovian_state()
        self.reward = -self.IB.state['cost']

        # Alternative reward that returns the improvement or decrease in the cost function
        # If the cost function improves/decreases, the reward is positive
        # If the cost function deteriorates/increases, the reward is negative
        # e.g.: -400 -> -450 = delta_reward of -50
        self.delta_reward = 0

        # smoother reward function for monitoring the agent & environment with lower variance
        # Updates with a convex combination of old and new cost
        self.smoothed_reward = self.reward

        # used to set the self.done variable - If larger than self.reset_after_timesteps, the environment resets
        self.env_steps = 0

        # whether or not the trajectory has ended
        self.done = False

        return self.lstm_quantize(return_observation)[0]

    def step(self, action):
        """
             performs one step in the environment by taking the specified action and returning the resulting observation
             :param action: the action to be taken
             :return: the new observation
             """

        # when the done flag has been set and the user still calls step, we want to at least reset the environment
        if self.done:
            self.reset()

        # keep the current action around for potential rendering
        self.last_action = action

        # Executing the action and saving the observation
        if self.action_type == 'discrete':
            self.IB.step(self.env_action[action])  # for discrete actions, we expect the action's index
        elif self.action_type == 'continuous':
            self.IB.step(action)  # in the continuous case, we expect the entire three dimensional action

        # update observation representation
        return_observation = self._update_observation()

        # Calculating both the relative reward (improvement or decrease) and updating the absolute reward
        new_reward = -self.IB.state['cost']
        self.delta_reward = new_reward - self.reward  # positive when improved
        self.reward = new_reward

        # Due to the very high stochasticity a smoothed reward function can be easier to follow visually
        self.smoothed_reward = 0.9 * self.smoothed_reward + 0.1 * self.reward

        # Stopping condition
        self.env_steps += 1
        if self.env_steps >= self.reset_after_timesteps:
            self.done = True

        # Two reward functions are available:
        # 'classic' which returns the original cost and
        # 'delta' which returns the change in the cost function w.r.t. the previous cost
        if self.reward_function == 'classic':
            return_reward = self.reward
        elif self.reward_function == 'delta':
            return_reward = self.delta_reward
        else:
            raise ValueError('Invalid reward function specification. "classic" for the original cost function'
                             ' or "delta" for the change in the cost fucntion between steps.')

        self.info = self._markovian_state()  # entire markov state - not all info is visible in observations
        q_state = self.lstm_quantize(return_observation)[0]
        return q_state, return_reward, self.done, self.info
