from typing import List

import numpy as np
import matplotlib.pyplot as plt
import sinter
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit

from src.util import stats_from_csv
#from src.visualization import plot_threshold
from src.analysis import process_raw_stats, collate_collected_stats

path_pair_correlated = "data/saved/experiment4/pair_all_poly_correlated.csv"
path_pair_independent = "data/saved/experiment4/pair_all_poly_independent.csv"
path_streak_correlated = "data/saved/experiment4/streak_all_poly_correlated.csv"
path_streak_independent = "data/saved/experiment4/streak_all_poly_independent.csv"

stats_pair_correlated = process_raw_stats(stats_from_csv(path_pair_correlated))
stats_pair_independent = collate_collected_stats(sinter.read_stats_from_csv_files(path_pair_independent))
stats_streak_correlated = process_raw_stats(stats_from_csv(path_streak_correlated))
stats_streak_independent = collate_collected_stats(sinter.read_stats_from_csv_files(path_streak_independent))

def plot_threshold(collected_stats, xlim=None, ylim=None):
    fig, ax = plt.subplots()
    
    if isinstance(collected_stats[0], sinter._task_stats.TaskStats):
        stats = collate_collected_stats(collected_stats)
    elif isinstance(list(collected_stats[1].values())[0][0], List):
        stats = process_raw_stats(collected_stats)
    else: stats = collected_stats
    
    for d in list(stats[0].keys())[:-1]:
        X = stats[0][d]
        try:
            Y = [y[0] for y in stats[1][d]]
        except:
            Y = stats[1][d]
        XY = [xy for xy in sorted(zip(X,Y))]
        X = [xy[0] for xy in XY]
        Y = [xy[1] for xy in XY]
        # print(XY)
        ax.plot(X, Y, label=f'd={d}', marker='.')
        
    ax.loglog()
    ax.grid()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate per Round")
    ax.legend()
    
    fig.tight_layout()
    fig.patch.set_alpha(0)
        
    return fig

xlim = [1e-4, 1e-1]
ylim = [1e-8, 5e-1]

fig1 = plot_threshold(stats_pair_correlated, xlim, ylim)
fig1.savefig("figures/output/experiment4a.pdf")

fig2 = plot_threshold(stats_pair_independent, xlim, ylim)
fig2.savefig("figures/output/experiment4b.pdf")

fig3 = plot_threshold(stats_streak_correlated, xlim, ylim)
fig3.savefig("figures/output/experiment4c.pdf")

fig4 = plot_threshold(stats_streak_independent, xlim, ylim)
fig4.savefig("figures/output/experiment4d.pdf")

stats_list = [stats_pair_correlated, stats_pair_independent, stats_streak_correlated, stats_streak_independent]
title_list = ["Pairwise correlated", "Pairwise independent", "Streaky correlated", "Streaky independent"]

fig, axs = plt.subplots(2, 2, figsize=(8.5, 7))

for i, stats in enumerate(stats_list):
    ax = axs[i//2, i%2]
    
    for d in list(stats[0].keys())[:-1]:
        #print(d)
        X = stats[0][d]
        try:
            Y = [y[0] for y in stats[1][d]]
        except:
            Y = stats[1][d]
        XY = [xy for xy in sorted(zip(X,Y))]
        X = [xy[0] for xy in XY]
        Y = [xy[1] for xy in XY]
        
        ax.plot(X, Y, label=f'd={d}', marker='.')
    
    ax.loglog()
    ax.legend(loc='lower right')
    ax.grid()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.set_title(title_list[i])

fig.supxlabel('Physical error rate')
fig.supylabel('Logical error rate per round')
fig.patch.set_alpha(0)
fig.tight_layout()

    
fig.savefig("figures/output/experiment4.pdf")
fig.savefig("figures/output/experiment4.svg")
