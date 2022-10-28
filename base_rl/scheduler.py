from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt


class HyperParamScheduler:

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def step(self):
        pass


class ConstantScheduler(HyperParamScheduler):
    def __init__(self, const_value):
        self.const_value = const_value

    def get(self):
        return self.const_value

    def step(self):
        pass


class LinearScheduler(HyperParamScheduler):

    def __init__(self, start, end, total_steps):
        self.current_step = 0
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.hyper_param = np.linspace(start=start, stop=end, num=total_steps)
        plt.plot(self.hyper_param)
        plt.show()

    def get(self):
        if self.current_step < self.hyper_param.size:
            return self.hyper_param[self.current_step]
        else:
            return self.hyper_param[-1]

    def step(self):
        self.current_step += 1


class DecayingExpContinuousScheduler(HyperParamScheduler):
    def __init__(self, start, decay, plot_time_steps=10000):
        self.start = start
        self.decay = decay
        self.current_step = 0

        values = [start * pow(decay, i) for i in range(plot_time_steps)]
        plt.plot(values)
        plt.show()

    def get(self):
        return self.start * pow(self.decay, self.current_step)

    def step(self):
        self.current_step += 1


class DecayingExpScheduler(HyperParamScheduler):

    def __init__(self, start, end, total_steps, base=1):
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.current_step = 0
        self.base = base
        self.hyper_param = np.logspace(np.log(self.start), np.log(self.end), self.total_steps, base=np.exp(self.base))
        plt.plot(self.hyper_param)
        plt.show()

    def get(self):
        if self.current_step < self.hyper_param.size:
            return self.hyper_param[self.current_step]
        else:
            return self.hyper_param[-1]

    def step(self):
        self.current_step += 1
