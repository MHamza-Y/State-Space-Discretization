from dynamic_programming.policy import DPPolicy
from experiments.base_configs import RLExperimentConfig


class OfflineDiscreteRLExperimentConfig(RLExperimentConfig):

    def __init__(self, dataset_path, mdp_path, dataset_size, model_name, model_path, policy_path):
        self.dataset_path = dataset_path
        self.mdp_path = mdp_path
        self.dataset_size = dataset_size
        self.model_name = model_name
        self.model_path = model_path
        self.policy_path = policy_path

    def get_saved_policy(self):
        return DPPolicy.load(self.policy_path)


class RMinExperimentConfig(OfflineDiscreteRLExperimentConfig):

    def __init__(self, r_min, **kwargs):
        super().__init__(**kwargs)
        self.r_min = r_min
