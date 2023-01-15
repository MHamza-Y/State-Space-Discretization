from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np
from tqdm import tqdm

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
        total_process = len(self.evaluators)
        pbar = tqdm(total=total_process)
        with ProcessPoolExecutor(max_workers=self.pool_size) as executor:
            tasks = [executor.submit(evaluator.evaluate, epochs=self.epochs, render=False) for evaluator
                              in self.evaluators]
            for task in futures.as_completed(tasks):
                self.evaluated_evaluators.append(task.result())
                pbar.update(1)
