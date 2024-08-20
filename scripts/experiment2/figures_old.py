import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.analysis import filepaths_to_data_correlated, filepaths_to_data_independent, format_data_for_plotting

def fit_func(x, a, b, c):
    return a + b * x + c * np.log(x)

# def fit_func(x, a, b):
#     return a + b * np.log(x)

path = './data/saved/experiment2/class1/'
names_corr = ["streak_m_poly_a1n3_correlated",
               "streak_m_poly_a1n4_correlated",
               "streak_m_poly_a1n5_correlated",
               "streak_m_exp_a1n2_correlated"]

names_indep = ["streak_m_poly_a1n3_independent",
               "streak_m_poly_a1n4_independent",
               "streak_m_poly_a1n5_independent",
               "streak_m_exp_a1n2_independent"]

paths_correlated = [path + name + ".csv" for name in names_corr]
paths_independent = [path + name + ".csv" for name in names_indep]

titles = ["Class 1 (polynomial decay, n=3)",
          "Class 1 (polynomial decay, n=4)",
          "Class 1 (polynomial decay, n=5)",
          "Class 1 (exponential decay, n=2)",
          "Class 2 (polynomial decay, n=3)",
          "Class 2 (polynomial decay, n=4)",
          "Class 2 (polynomial decay, n=5)",
          "Class 2 (exponential decay, n=2)"]


fig, axs = plt.subplots(2, 4, figsize=(12, 6))
coords = [(0, 0), (0, 1), (0, 2), (0, 3),
          (1, 0), (1, 1), (1, 2), (1, 3)]

data_correlated = list(filepaths_to_data_correlated(paths_correlated))
data_independent = list(filepaths_to_data_independent(paths_independent))

prop_cycle = plt.rcParams['axes.prop_cycle']
color1, color2 = prop_cycle.by_key()['color'][:2]

for i in range(len(data_correlated)):
    ax = axs[coords[i]]
    # print(format_data_for_plotting(data))
    x_corr, y_corr, yerr_corr = format_data_for_plotting(data_correlated[i])
    x_indep, y_indep, yerr_indep = format_data_for_plotting(data_independent[i])
    popt_corr, pcov = curve_fit(fit_func, x_corr, np.log10(y_corr))
    popt_indep, pcov = curve_fit(fit_func, x_indep, np.log10(y_indep))
    
    ax.errorbar(x_corr, y_corr, yerr=yerr_corr, fmt='o', label="Correlated noise model", color=color1)
    ax.errorbar(x_indep, y_indep, yerr=yerr_indep, fmt='v', label="Independent noise model", color=color2)
    
    x_fit = np.linspace(3, 30, 100)
    ax.plot(x_fit, 10**fit_func(x_fit, *popt_corr), color=color1, label="Correlated fit")
    ax.plot(x_fit, 10**fit_func(x_fit, *popt_indep), linestyle='--', color=color2, label="Independent fit")

    ax.semilogy()
    ax.set_xlim([0, 30])
    ax.set_ylim([10e-13, 10e-3])
    ax.set_title(titles[i])
    ax.grid()
    
    if i not in [0, 4]:
        ax.set_yticklabels([])

fig.tight_layout()
fig.savefig('figures/output/experiment2.pdf', dpi=600)
fig.savefig('figures/output/experiment2.svg', dpi=600)