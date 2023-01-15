import os
import dill
from dataclasses import dataclass


@dataclass
class ExperimentsEvaluationConfigs:
    experiment_id: str
    model_name: str
    model_path: str
    policy_path: str

    def get_policy(self):
        with open(self.policy_path, 'rb') as f:
            return dill.load(f)
