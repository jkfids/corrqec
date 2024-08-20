import matplotlib.pyplot as plt
import numpy as np
import sinter

from src.util import stats_from_csv
from src.analysis import collate_stats_for_projection, convert_stats_for_projection

# corr_path = 'data/output/class0/streak_data_poly_correlated.csv'
# indep_path = 'data/output/class0/streak_data_poly_independent.csv'
# corr_path = 'data/output/class1/streak_m_poly_correlated.csv'
# indep_path = 'data/output/class1/streak_m_poly_independent.csv'
corr_path = 'data/output/class2/pair_c_poly_correlated.csv'
indep_path = 'data/output/class2/pair_c_poly_independent.csv'

stats = stats_from_csv(corr_path)
collected_stats = sinter.read_stats_from_csv_files(indep_path)
corr_data = convert_stats_for_projection(stats)
indep_data = collate_stats_for_projection(collected_stats)

fig, ax = plt.subplots()

xs = []
ys = []
yerrs = []

for data in (corr_data, indep_data):
    x = next(iter(data[0].values()))
    per_rounds = np.array(next(iter(data[1].values())))
    y = per_rounds[:, 0]
    yerr = per_rounds[:, 1:].T
    
    xs.append(x)
    ys.append(y)
    yerrs.append(yerr)
    
ax.errorbar(xs[0], ys[0], yerr=yerrs[0], label='Correlated', fmt='o', markersize=3)
ax.errorbar(xs[1], ys[1], yerr=yerrs[1], label='Independent', fmt='v', markersize=3)

ax.semilogy()
ax.grid()
ax.legend()
ax.set_xlim([0, 30])
ax.set_ylim([10e-12, 10e-3])

fig.tight_layout()
fig.savefig('figures/output/test.pdf', dpi=600)
