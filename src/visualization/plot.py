# Standard imports
from typing import List

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import sinter

# Local imports
from ..analysis import process_raw_stats, collate_collected_stats, fit_stats_for_projection


def plot_threshold(collected_stats, xlim=None, ylim=None):
    fig, ax = plt.subplots()

    if isinstance(collected_stats[0], sinter._task_stats.TaskStats):
        stats = collate_collected_stats(collected_stats=collected_stats)
    elif isinstance(list(collected_stats[1].values())[0][0], List):
        stats = process_raw_stats(collected_stats)
    else: stats = collected_stats
    for d in stats[0].keys():
        #ax.scatter(stats[1][d], stats[0][d], label=f'd={d}', linestyle='-')
        ax.plot(stats[0][d], stats[1][d], label=f'd={d}', marker='.')
        
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

def plot_projection_multi(stats_list, labels, probabilities, shape, max_dist=30, xlim=None, ylim=[10e-12, 10e-2]):
    
    r, c = shape
    fig, axs = plt.subplots(r, c, figsize=(c*4, r*3.5))

    markers = ['o', '^', 's', '*']
    axs_coords = [(i, j) for i in range(r) for j in range(c)]

    xs_multi = []
    ys_multi = []
    fits_multi = []

    for stats in stats_list:
        xs, ys, fits = fit_stats_for_projection(stats)
        xs_multi.append(xs)
        ys_multi.append(ys)
        fits_multi.append(fits)
        
    # Get colour cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i, p in enumerate(probabilities):
        for j, (xs, ys, fits) in enumerate(zip(xs_multi, ys_multi, fits_multi)):
            x = xs[p]
            y = ys[p]
            fit = fits[p]
        
            axs[axs_coords[i]].scatter(x, y, marker=markers[j])
            axs[axs_coords[i]].plot([0, max_dist], [np.exp(fit.intercept), np.exp(fit.intercept + fit.slope * max_dist)], linestyle='--')
            axs[axs_coords[i]].set_title(f"p = {p}")

        axs[axs_coords[i]].semilogy()
        axs[axs_coords[i]].grid()
        axs[axs_coords[i]].set_ylim(ylim)
        
        if axs_coords[i] == (0, c-1):
            for k, label in enumerate(labels):
                axs[axs_coords[i]].plot([], [], marker=markers[k], linestyle='--', color=colors[k], label=labels[k])
            axs[(0, c-1)].legend()
            
    fig.supxlabel('Distance')
    fig.supylabel('Logical error rate per round')
    
    fig.tight_layout()
    fig.patch.set_alpha(0)
    
    return fig