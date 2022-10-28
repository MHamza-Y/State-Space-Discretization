from multiprocessing import Process, set_start_method

from base_rl.algorithm import RLAlgorithm
from base_rl.callbacks import OnTrainingStartCallback, OnTrainingEndCallback, OnEpisodeStartCallback, \
    OnEpisodeEndCallback

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class OnlineTrainer:

    def __init__(self, algo: RLAlgorithm, callbacks):
        self.on_training_start_cb = [cb for cb in callbacks if isinstance(cb, OnTrainingStartCallback)]
        self.on_training_end_cb = [cb for cb in callbacks if isinstance(cb, OnTrainingEndCallback)]
        self.on_episode_start_cb = [cb for cb in callbacks if isinstance(cb, OnEpisodeStartCallback)]
        self.on_episode_end_cb = [cb for cb in callbacks if isinstance(cb, OnEpisodeEndCallback)]
        self.algo = algo

    def execute_callback(self, callbacks):
        for cb in callbacks:
            cb(algo=self.algo, trainer=self)

    def fit(self):
        self.execute_callback(self.on_training_start_cb)
        self.algo.setup()
        while self.algo.keep_training():
            self.execute_callback(self.on_episode_start_cb)
            self.algo.train_episode()
            self.execute_callback(self.on_episode_end_cb)
        self.execute_callback(self.on_training_end_cb)


class TrainerProcess(Process):

    def __init__(self, trainer_class, trainer_kwargs):
        super().__init__()
        self.trainer_class = trainer_class
        self.trainer_kwargs = trainer_kwargs
        self.trainer = None

    def run(self) -> None:
        self.trainer = self.trainer_class(**self.trainer_kwargs)
        self.trainer.fit()


class ParallelTrainer:

    def __init__(self, trainer_class, workers_kwargs):
        self.trainer_class = trainer_class
        self.workers_kwargs = workers_kwargs
        self.trainer_processes = [TrainerProcess(trainer_class=trainer_class, trainer_kwargs=configs) for configs in
                                  workers_kwargs]

    def train(self):
        for p in self.trainer_processes:
            p.start()
        for p in self.trainer_processes:
            p.join()
