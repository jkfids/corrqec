import matplotlib.pyplot as plt
import numpy as np
import sinter

from src.util import stats_from_csv
from src.visualization import plot_projection_multi

probabilities = [.01]

corr_path = 'data/output/streak_data_poly/streak_data_poly_correlated.csv'
ind_path = 'data/output/streak_data_poly/streak_data_poly_independent.csv'
output_path = 'figures/output/projection_streak_data_poly.pdf'

stats = stats_from_csv(corr_path)
collected_stats = sinter.read_stats_from_csv_files(ind_path)

fig = plot_projection_multi([stats, collected_stats], labels=["Correlated", "Independent"], probabilities=probabilities[0:4], shape=(2, 2), max_dist=30, xlim=None, ylim=[10e-12, 10e-2])
fig.savefig(output_path, dpi=600)

corr_path = 'data/output/pair_data_poly/pair_data_poly_correlated.csv'
ind_path = 'data/output/pair_data_poly/pair_data_poly_independent.csv'
output_path = 'figures/output/projection_pair_data_poly.pdf'

stats = stats_from_csv(corr_path)
collected_stats = sinter.read_stats_from_csv_files(ind_path)

fig = plot_projection_multi([stats, collected_stats], labels=["Correlated", "Independent"], probabilities=probabilities[0:4], shape=(2, 2), max_dist=30, xlim=None, ylim=[10e-12, 10e-2])
fig.savefig(output_path, dpi=600)