#!/usr/bin/env python3

import argparse
import pathlib
from typing import List

import matplotlib
import sinter
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, nargs='+', type=str)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save', default=None, type=str)
    args = parser.parse_args()
    if args.save is None and not args.show:
        raise ValueError("Specify --save or --show")

    fig = plot_stability_stats_error_rate(
        sinter.stats_from_csv_files(*args.csv),
    )
    fig.set_size_inches(16, 9)
    if args.save is not None:
        pathlib.Path(args.save).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(args.save, bbox_inches='tight', dpi=256)
        print(f"wrote {args.save}")
    if args.show:
        plt.show()


def plot_stability_stats_error_rate(
        stats: List[sinter.TaskStats],
) -> plt.Figure:
    stats = [
        stat
        for stat in stats
        if stat.json_metadata['pm'] == stat.json_metadata['pd']
    ]
    memory_stats = [stat for stat in stats if stat.json_metadata['type'] == 'memory']
    stability_stats = [stat for stat in stats if stat.json_metadata['type'] == 'stability']

    MARKERS: str = "ov*sp^<>8PhH+xXDd|" * 100
    COLORS: List[str] = list(matplotlib.colors.TABLEAU_COLORS) * 3

    axs: List[plt.Axes]
    fig, axs = plt.subplots(1, 2)

    mem_seen_sizes = sorted({stat.json_metadata['d'] for stat in memory_stats})
    mem_seen_rounds = sorted({stat.json_metadata['r'] for stat in memory_stats})
    mem_size_to_color = {mem_seen_sizes[k]: COLORS[k] for k in range(len(mem_seen_sizes))}
    mem_size_to_marker = {mem_seen_sizes[k]: MARKERS[k] for k in range(len(mem_seen_sizes))}
    mem_rounds_to_color = {mem_seen_rounds[k]: COLORS[k] for k in range(len(mem_seen_rounds))}
    mem_rounds_to_marker = {mem_seen_rounds[k]: MARKERS[k] for k in range(len(mem_seen_rounds))}
    sinter.plot_error_rate(
        ax=axs[0],
        stats=memory_stats,
        x_func=lambda stat: stat.json_metadata['pm'],
        group_func=lambda stat: (-stat.json_metadata['r'], stat.json_metadata['d']),
        plot_args_func=lambda _, c: {'color': mem_rounds_to_color[-c[0]], 'marker': mem_size_to_marker[c[1]], 'label': f'diam={c[1]} rounds={-c[0]}'},
    )

    stab_seen_sizes = sorted({stat.json_metadata['d'] for stat in stability_stats})
    stab_seen_rounds = sorted({stat.json_metadata['r'] for stat in stability_stats})
    stab_size_to_color = {stab_seen_sizes[k]: COLORS[k] for k in range(len(stab_seen_sizes))}
    stab_size_to_marker = {stab_seen_sizes[k]: MARKERS[k] for k in range(len(stab_seen_sizes))}
    stab_rounds_to_color = {stab_seen_rounds[k]: COLORS[k] for k in range(len(stab_seen_rounds))}
    stab_rounds_to_marker = {stab_seen_rounds[k]: MARKERS[k] for k in range(len(stab_seen_rounds))}
    sinter.plot_error_rate(
        ax=axs[1],
        stats=stability_stats,
        x_func=lambda stat: stat.json_metadata['pm'],
        group_func=lambda stat: (-stat.json_metadata['d'], stat.json_metadata['r']),
        plot_args_func=lambda _, c: {'color': stab_size_to_color[-c[0]], 'marker': stab_rounds_to_marker[c[1]], 'label': f'diam={-c[0]} rounds={c[1]}'},
    )

    axs[0].set_title("Memory Experiment")
    axs[1].set_title("Stability Experiment")
    for k in range(2):
        axs[k].set_xlim(1e-3, 1e-1)
        axs[k].set_ylim(1e-4, 1e-0)
        axs[k].set_xlabel('Physical Error Rate')
        axs[k].set_ylabel('Logical Error Rate (per shot)')
        axs[k].loglog()
        axs[k].grid(which='major', color='k')
        axs[k].grid(which='minor')
        axs[k].legend()

    return fig


if __name__ == '__main__':
    main()
