from base_algorithm.base_policies import DiscretePolicy


class DPPolicy(DiscretePolicy):

    def state_exist(self, state):
        return state in self.state_to_index

    def __init__(self, policy_table, state_to_index, index_to_action):
        self.policy_table = policy_table
        self.state_to_index = state_to_index
        self.index_to_action = index_to_action

    def get_action(self, state):
        action_index = self.policy_table[self.state_to_index[state]]
        action = self.index_to_action[action_index]
        return action
