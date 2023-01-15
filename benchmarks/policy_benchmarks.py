from multiprocessing.pool import Pool
from typing import List

import numpy as np

from base_rl.eval_policy import EvalDiscreteStatePolicy


class PolicyBenchmarkResults:

    def __init__(self, evaluator: EvalDiscreteStatePolicy):
        pass


class PolicyBenchmarks:
    def __init__(self, evaluators: List[EvalDiscreteStatePolicy], epochs):
        self.epochs = epochs
        self.evaluators = evaluators
        self.benchmark_metrics = {}

    def update_metrics(self, evaluator):
        obs = [traj['obs'] for traj in evaluator.eval_trajectories]
        un = np.unique(obs)
        self.benchmark_metrics.update({evaluator.tag: {
            'reward': np.mean(evaluator.eval_rewards_per_epoch), 'std': np.std(evaluator.eval_rewards_per_epoch),
            'unique_obs': un.size}})

    def benchmark(self):
        for evaluator in self.evaluators:
            evaluator.evaluate(self.epochs)
            self.update_metrics(evaluator=evaluator)


class PolicyBenchmarksParallel(PolicyBenchmarks):

    def __init__(self, pool_size=4, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.evaluated_evaluators = []

    def benchmark(self):
        with Pool(self.pool_size) as pool:
            results = []
            for evaluator in self.evaluators:
                result = pool.apply_async(evaluator.evaluate, kwds={'epochs': self.epochs, 'render': False})
                results.append(result)

            for result in results:
                evaluated_benchmark = result.get()
                self.evaluated_evaluators.append(evaluated_benchmark)
                self.update_metrics(evaluated_benchmark)
