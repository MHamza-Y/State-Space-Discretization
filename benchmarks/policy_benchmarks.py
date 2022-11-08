from typing import List

import numpy as np

from base_rl.eval_policy import EvalDiscreteStatePolicy


class PolicyBenchmarks:

    def __init__(self, evaluators: List[EvalDiscreteStatePolicy], epochs):
        self.epochs = epochs
        self.evaluators = evaluators
        self.benchmark_metrics = {}

    def benchmark(self):
        for evaluator in self.evaluators:
            evaluator.evaluate(self.epochs)
            self.benchmark_metrics.update({evaluator.tag: {
                'reward': np.mean(evaluator.eval_rewards_per_epoch), 'std': np.std(evaluator.eval_rewards_per_epoch)}})
