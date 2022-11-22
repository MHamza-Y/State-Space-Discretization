from base_rl.scheduler import DecayingExpContinuousScheduler

start = 1
decay = 0.994
epsilon = DecayingExpContinuousScheduler(start=1, decay=0.999)

print(start * pow(decay, 1000))
