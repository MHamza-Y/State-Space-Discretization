import pandas as pd


def plot_industrial_benchmark_trajectories(episode_data):
    df = pd.DataFrame(episode_data)
    axes = df[['velocity', 'gain', 'shift', 'fatigue', 'consumption']].plot(subplots=True, figsize=(62, 48),
                                                                            fontsize=32)
    for a in axes:
        a.legend(loc='best', prop={'size': 32})
