# Standard library imports

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import sinter

# Local imports
from src.util import stats_from_csv
from src.visualization import plot_projection_multi

input_path_corr = 'data/output/stats/rotated_memory_z_streak_m_poly_scl_2.csv'
input_path_marg = 'data/output/sinterstats/rotated_memory_z_streak_m_poly_scl_marginal_2.csv'
output_path = 'figures/output/projection_rotated_memory_z_streak_m_poly_scl_2.pdf'

stats = stats_from_csv(input_path_corr)
collected_stats = sinter.read_stats_from_csv_files(input_path_marg)

probabilities = np.concatenate([[1e-3, 2e-3, 5e-3], np.linspace(1e-2, 5e-2, 5)])
probabilities = probabilities[:4]
fig = plot_projection_multi([stats, collected_stats], labels=["Correlated", "Independent"], probabilities=probabilities, shape=(2, 2), max_dist=30, xlim=None, ylim=[10e-12, 10e-2])
fig.savefig(output_path, dpi=600)