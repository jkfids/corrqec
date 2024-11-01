#!/usr/bin/env python3

import argparse
import math
import pathlib
import sys
from typing import List, Sequence, Any

import matplotlib
import numpy as np
import scipy.spatial
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

    fig = plot_stability_stats_heatmap(
        sinter.stats_from_csv_files(*args.csv),
    )
    fig.set_size_inches(16, 9)
    if args.save is not None:
        pathlib.Path(args.save).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(args.save, bbox_inches='tight', dpi=256)
        print(f"wrote {args.save}")
    if args.show:
        plt.show()


def plot_stability_stats_heatmap(
        stats: List[sinter.TaskStats],
) -> plt.Figure:
    def key_func(stat: sinter.TaskStats):
        t = stat.json_metadata['type']
        if t == 'memory':
            k = 'r'
        else:
            k = 'd'
        return t, stat.json_metadata[k]

    pms = sorted({stat.json_metadata['pm'] for stat in stats})
    pds = sorted({stat.json_metadata['pd'] for stat in stats})

    memory_stats = [stat for stat in stats if stat.json_metadata['type'] == 'memory']
    stability_stats = [stat for stat in stats if stat.json_metadata['type'] == 'stability']
    memory_groups = [e[1] for e in sorted(sinter.group_by(memory_stats, key=key_func).items())]
    stability_groups = [e[1] for e in sorted(sinter.group_by(stability_stats, key=key_func).items())]

    n = max(len(memory_groups), len(stability_groups))
    fig, axs = plt.subplots(2, max(n, 2))
    color_map = matplotlib.cm.get_cmap('plasma')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
    for k in range(n):
        if k < len(memory_groups):
            axs[0, k].set_ylim(0, max(pms))
            axs[0, k].set_xlim(0, max(pds))
            d = sorted({stat.json_metadata['d'] for stat in memory_groups[k]})
            r = sorted({stat.json_metadata['r'] for stat in memory_groups[k]})
            sizes = ','.join(f'{e}x{e}' for e in d)
            rounds = ','.join(f'{e}' for e in r)
            axs[0, k].set_title(f"Memory sizes={sizes} rounds={rounds}")
            plot_lambda_heat_map(stats=memory_groups[k], ax=axs[0, k], color_map=sm)

        if k < len(stability_groups):
            axs[1, k].set_ylim(0, max(pms))
            axs[1, k].set_xlim(0, max(pds))
            d = sorted({stat.json_metadata['d'] for stat in stability_groups[k]})
            r = sorted({stat.json_metadata['r'] for stat in stability_groups[k]})
            sizes = ','.join(f'{e}x{e}' for e in d)
            rounds = ','.join(f'{e}' for e in r)
            axs[1, k].set_title(f"Stability sizes={sizes} rounds={rounds}")
            plot_lambda_heat_map(stats=stability_groups[k], ax=axs[1, k], color_map=sm)

        axs[1, k].set_xlabel('Unitary Operation Noise Strength')

    for k in range(2):
        cbar = fig.colorbar(sm, ax=axs[k, -1])
        cbar.ax.set_yticks([0, 1, 2, 3])
        if k == 0:
            cbar.ax.set_ylabel('Logical Error Suppression per Diameter')
        else:
            cbar.ax.set_ylabel('Logical Error Suppression per Round')
        cbar.ax.set_yticklabels(['≤0dB', '1dB', '2dB', '≥3dB'])
        axs[k, 0].set_ylabel('MeasureReset Noise Strength')

    return fig


def plot_lambda_heat_map(
        *,
        ax: plt.Axes,
        stats: List[sinter.TaskStats],
        color_map: Any,
) -> None:

    groups = sinter.group_by(stats, key=lambda stat: (stat.json_metadata['pm'], stat.json_metadata['pd']))

    xs = []
    ys = []
    colors = []
    for group_key in sorted(groups.keys()):
        group_stats = sorted(groups[group_key], key=lambda stat: (stat.json_metadata['d'], stat.json_metadata['r']))
        distances = sorted({stat.json_metadata['d'] for stat in group_stats})
        rounds = sorted({stat.json_metadata['r'] for stat in group_stats})
        if len(distances) > 1:
            k = 'd'
        elif len(rounds) > 1:
            k = 'r'
        else:
            print("SKIPPING", group_key, file=sys.stderr)
            continue
        pts = []
        for stat in sorted(group_stats, key=lambda e: e.json_metadata[k]):
            if stat.errors == 0 and len(pts) > 1:
                # If no errors were seen, don't include in the fit.
                # This avoids extremely low error rates from being perceived as shallow.
                break
            x = stat.json_metadata[k]
            y = -math.log((stat.errors + 1) / (stat.shots + 2)) / math.log(10) * 10
            pts.append((x, y))
        if len(pts) < 2:
            continue
        lamb = sinter.fit_line_slope(xs=[p[0] for p in pts], ys=[p[1] for p in pts], max_extra_squared_error=1).best
        xs.append(group_key[1])
        ys.append(group_key[0])
        colors.append(color_map.to_rgba(lamb))
    voronoi_heat_map(xs=xs, ys=ys, ax=ax, colors=colors)


def voronoi_heat_map(
    *,
    ax: plt.Axes,
    xs: Sequence[float],
    ys: Sequence[float],
    colors: Sequence[Any],
) -> None:
    pts = [
        (x, y) for x, y in zip(xs, ys, strict=True)
    ]
    pts.append((-1000, -1000))
    pts.append((+1000, -1000))
    pts.append((-1000, +1000))
    pts.append((+1000, +1000))

    vor = scipy.spatial.Voronoi(np.array(pts))
    for k in range(len(xs)):
        region = vor.regions[vor.point_region[k]]
        if len(region) == 0 or -1 in region:
            continue
        region_pts = [vor.vertices[k] for k in region]
        region_xs = [p[0] for p in region_pts]
        region_ys = [p[1] for p in region_pts]
        ax.fill(region_xs, region_ys, c=colors[k], edgecolor='k')
    ax.scatter(xs, ys, marker='+', color='k')


if __name__ == '__main__':
    main()
