# Standard library imports
from typing import List

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import stim
import sinter

# Local imports

input_path = 'data/output/sinterstats/rotated_memory_z_scl.csv'
output_path = 'figures/output/threshold/threshold_rotated_memory_z_scl.pdf'

collected_stats = sinter.read_stats_from_csv_files(input_path)

# fig, ax = plt.subplots()

# sinter.plot_error_rate(
#     ax=ax,
#     stats=collected_stats,
#     x_func=lambda stat: stat.json_metadata['p'],
#     group_func=lambda stat: stat.json_metadata['d'],
#     failure_units_per_shot_func=lambda stat: stat.json_metadata['r'],
# )

# # ax.set_ylim(5e-3, 5e-2)
# # ax.set_xlim(0.008, 0.012)
# ax.loglog()
# ax.set_title("Surface Code Error Rates per Round under Circuit Noise")
# ax.set_xlabel("Phyical Error Rate")
# ax.set_ylabel("Logical Error Rate per Round")
# ax.grid(which='major')
# ax.grid(which='minor')
# ax.legend()
# fig.tight_layout()

# fig.savefig(savepath, dpi=600)