import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.analysis import filepaths_to_data_correlated, filepaths_to_data_independent, format_data_for_plotting

def fit_func(x, a, b, c):
    return a + b * x + c * np.log(x)

path = './data/saved/experiment2/'
names_class1_poly = ["streak_m_poly_a1n2",
                "streak_m_poly_a1n3",
               "streak_m_poly_a1n4",
               "streak_m_poly_a1n5",
               "streak_m_poly_a1ninf"]
names_class1_exp = ["streak_m_poly_a1n3",]

paths_list = []
for names in [names_class1_poly, names_class1_exp, [], []]:
    paths = [path + name + ".csv" for name in names]
    paths_list.append(paths)
    
titles = ["Class 1 polynomially decaying",
          "Class 1 exponentially decaying",
          "Class 2 polynomially decaying",
          "Class 2 exponentially decaying"]

labels = ['A=1, n=2', 'A=1, n=3', 'A=1, n=4', 'A=1, n=5', r'A=1, n$\rightarrow \infty$']
shapes = ['o', 'v', '^', 'D', 's']

# fig, axs = plt.subplots(4, 1, figsize=(4.5, 13.5))
fig, axs = plt.subplots(2, 2, figsize=(8, 7))
coords = [(0, 0), (0, 1),
          (1, 0), (1, 1)]

data_list = []
for paths in paths_list:
    data_correlated = list(filepaths_to_data_correlated(paths))
    data_list.append(data_correlated)
x_fit = np.linspace(3, 30, 100)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color'][:8]

for i in range(len(titles)):
    ax = axs[coords[i]]
    for j in range(len(data_list[i])):
        
        x_corr, y_corr, yerr_corr = format_data_for_plotting(data_list[i][j])
        
        ax.errorbar(x_corr, y_corr, yerr=yerr_corr, fmt=shapes[j], color=colors[j], label=labels[j], markersize=4)
        
        popt_corr, pcov = curve_fit(fit_func, x_corr, np.log10(y_corr))
        ax.plot(x_fit, 10**fit_func(x_fit, *popt_corr), color=colors[j])
        
    ax.semilogy()
    ax.set_xlim([0, 30])
    ax.set_ylim([10e-13, 10e-3])
    ax.grid()
    ax.set_title(titles[i])
    ax.legend()
    if i in [1, 3]:
        ax.set_yticklabels([])

fig.supylabel('Logical error rate per round')
fig.supxlabel('Surface code distance')

fig.tight_layout()

fig.savefig('figures/output/experiment2.pdf', dpi=600)
fig.savefig('figures/output/experiment2.svg', dpi=600)