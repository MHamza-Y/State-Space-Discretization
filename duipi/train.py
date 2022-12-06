import numpy as np


def calculate_uncertainty_from_variance(variance, count):
    np.nan_to_num(variance, nan=1e20)
    return np.nan_to_num(variance / (count - 1), nan=1e20)


def calculate_alpha_prior(count_state_action_state):
    num_states = count_state_action_state.shape[0]
    mask = count_state_action_state > 0
    total_next_state_visited_per_state_action = np.sum(mask, 2)
    total_next_state_visited_per_state_action = np.expand_dims(total_next_state_visited_per_state_action, axis=2)
    alpha_prior = total_next_state_visited_per_state_action / num_states
    print(alpha_prior.min())
    print(alpha_prior.max())
    return alpha_prior


class DUIPITrainer:
    """
    Diagonal Approximation of Uncertainty Incorporating Policy Iteration (DUIPI)
    https://www.tu-ilmenau.de/fileadmin/Bereiche/IA/neurob/Publikationen/conferences_int/2009/Hans-ICANN-2009.pdf
    """

    def __init__(self, reward_function, transition_model, count_state_action, count_state_action_state, variance_R, xi,
                 gamma, bayesian=False, alpha_prior=0.5):
        """

        :param reward_function: R(s, a, s_next) expected reward of performing action 'a' on state 's' and reaching state 's_next'
        :param transition_model: P(s_next|s, a) conditional probability of reaching state 's_next' while performing action 'a' on state 's'
        :param count_state_action: number of time the state action pair (s,a) was visited in the dataset
        :param xi: parameter ξ that controls the influence of uncertainty on the policy
        """

        self.reward_function = reward_function
        self.transition_model = transition_model
        self.count_state_action = count_state_action
        self.xi = xi
        self.gamma = gamma
        self.num_states = transition_model.shape[0]
        self.num_actions = transition_model.shape[1]
        self.mask = self.count_state_action > 0
        self.bayesian = bayesian
        self.count_state_action_state = count_state_action_state
        self.alpha_prior = alpha_prior  # calculate_alpha_prior(count_state_action_state)
        self.pi = np.ones([self.num_states, self.num_actions]) / self.num_actions
        self.Q = np.zeros([self.num_states, self.num_actions])
        self.uncertainty_Q = np.zeros([self.num_states, self.num_actions])

        self.uncertainty_R = calculate_uncertainty_from_variance(variance_R, count_state_action_state)
        self.uncertainty_P = self.calculate_transition_uncertainty()


    def get_policy(self):
        """
        Returns the deterministic policy
        :return: deterministic policy
        """
        print(self.pi)
        return np.argmax(self.pi, axis=1)

    def calculate_transition_uncertainty(self):
        """
        This method calculates the initial transition variance

        (σP(s_next|s, a))² = P(s_next|s, a)(1 − P(s_next|s, a))/(n_sa − 1)
        where n_sa denotes the observed transitions from (s,a)

        max_variance = 1 / 4 according to Popoviciu's inequality on variances
        """
        max_variance = 1 / 4
        if self.bayesian:
            alpha_d = (self.count_state_action_state + self.alpha_prior)
            alpha_d_0 = np.sum(alpha_d, 2)[:, :, np.newaxis]
            self.transition_model = alpha_d / alpha_d_0
            uncertainty_p = alpha_d * (alpha_d_0 - alpha_d) / alpha_d_0 ** 2 / (alpha_d_0 + 1)
        else:

            uncertainty_p = np.zeros([self.num_states, self.num_actions, self.num_states])
            for state in range(self.num_states):
                uncertainty_p[:, :, state] = self.transition_model[:, :, state] * (
                        1 - self.transition_model[:, :, state]) / (self.count_state_action - 1)

            uncertainty_p = np.nan_to_num(uncertainty_p, nan=max_variance, posinf=max_variance)
            uncertainty_p[self.count_state_action == 0] = max_variance
        return uncertainty_p

    def policy_evaluation(self):
        """
        Evaluates the current policy pi and calculates its variance
        """
        V = np.einsum('ij,ij->i', self.pi, self.Q)
        variance_v = np.einsum('ij,ij->i', self.pi ** 2, self.uncertainty_Q)
        self.Q = np.einsum('ijk,ijk->ij', self.transition_model, self.reward_function + self.gamma * V)
        self.uncertainty_Q = np.dot(self.gamma ** 2 * self.transition_model ** 2, variance_v) + \
                             np.einsum('ijk,ijk->ij', (self.reward_function + self.gamma * V) ** 2,
                                       self.uncertainty_P) + \
                             np.einsum('ijk,ijk->ij', self.transition_model ** 2, self.uncertainty_R)
        self.uncertainty_Q = np.nan_to_num(self.uncertainty_Q, nan=np.inf, posinf=np.inf)

    def policy_improvement(self, epoch):
        """
        Updates the current policy pi
        """
        q_uncertainty_and_mask_corrected = self.Q - self.xi * np.sqrt(self.uncertainty_Q)
        q_uncertainty_and_mask_corrected[~self.mask] = - np.inf

        best_action = np.argmax(q_uncertainty_and_mask_corrected, axis=1)
        for state in range(self.num_states):
            d_s = np.minimum(1 / epoch, 1 - self.pi[state, best_action[state]])
            self.pi[state, best_action[state]] += d_s
            for action in range(self.num_actions):
                if action == best_action[state]:
                    continue
                elif self.pi[state, best_action[state]] == 1:
                    self.pi[state, action] = 0
                else:
                    self.pi[state, action] = self.pi[state, action] * (1 - self.pi[state, best_action[state]]) / (
                            1 - self.pi[state, best_action[state]] + d_s)

    def train(self, epochs=500):
        old_Q = np.zeros([self.num_states, self.num_actions])
        for epoch in range(1, epochs):
            print(f'Epoch: {epoch}')

            self.policy_evaluation()
            self.policy_improvement(epoch)
            print(f'Old vs New policy difference: {np.linalg.norm(self.Q - old_Q)}')

            old_Q = self.Q.copy()
