import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.analysis import filepaths_to_data_correlated, filepaths_to_data_independent, format_data_for_plotting

def fit_func1(x, a, b):
    return a + b * np.log(x)

def fit_func2(x, a, b):
    return a + b * x

def fit_func3(x, a, b):
    return a + b * np.log(x) ** .1

path = './data/saved/experiment3/'

names_corr = ["pair_all_poly_correlated",
              "streak_all_poly_correlated"]
names_indep = ["pair_all_poly_independent",
               "streak_all_poly_independent"]

paths_correlated = [path + name + ".csv" for name in names_corr]
paths_independent = [path + name + ".csv" for name in names_indep]

data_correlated = list(filepaths_to_data_correlated(paths_correlated))
data_independent = list(filepaths_to_data_independent(paths_independent))
x_fit = np.linspace(3, 30, 100)

titles = ["Pairwise correlated",
          "Streaky correlated"]

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color'][:2]

fig, axs = plt.subplots(2, 1, figsize=(5, 8))

for i in range(len(data_correlated)):
    ax = axs[i]
    x_corr, y_corr, yerr_corr = format_data_for_plotting(data_correlated[i])
    x_indep, y_indep, yerr_indep = format_data_for_plotting(data_independent[i])
    
    ax.errorbar(x_corr, y_corr, yerr=yerr_corr, fmt='o', label="Correlated noise model", color=colors[0])
    ax.errorbar(x_indep, y_indep, yerr=yerr_indep, fmt='v', label="Independent noise model", color=colors[1])
    
    if i == 1:
        fit_func = fit_func3
    else:
        fit_func = fit_func2
    popt_corr, pcov = curve_fit(fit_func, x_corr, np.log10(y_corr))
    popt_indep, pcov = curve_fit(fit_func2, x_indep, np.log10(y_indep))
    ax.plot(x_fit, 10**fit_func(x_fit, *popt_corr), color=colors[0], label="Correlated fit")
    ax.plot(x_fit, 10**fit_func2(x_fit, *popt_indep), linestyle='--', color=colors[1], label="Independent fit")
    
    ax.semilogy()
    ax.set_xlim([0, 30])
    ax.set_ylim([10e-13, 10e-3])
    ax.grid()
    ax.set_title(titles[i])
                
# axs[0].legend()
# axs[0].set_xticklabels([])
fig.supxlabel('Surface code distance')
fig.supylabel('Logical error rate per round')

# Reorder legend
handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 3, 0, 1]
axs[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right')

fig.tight_layout()
fig.savefig('figures/output/experiment3.pdf', dpi=600)
fig.savefig('figures/output/experiment3.svg', dpi=600)