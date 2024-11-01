import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from real2tex import real2tex

from src.analysis import filepaths_to_data_correlated, filepaths_to_data_independent, format_data_for_plotting

def fit_func1(x, a, b, c):
    return a + b * x + c * np.log(x)

def fit_func2(x, a, b):
    return a + b * x

def fit_func3(x, a, b):
    return a + b * np.log(x)

def fit_func4(x, a, b, c):
    return np.log10(a*x**b + c)

e = np.exp(1)

path = './data/saved/experiment2/'
names_class1_poly = ["class1/streak_m_poly_a1n2",
                     "class1/streak_m_poly_a1n3",
                     "class1/streak_m_poly_a1n4",
                     "class1/streak_m_poly_a1n5",
                     "class1/streak_m_poly_a1ninf"]

names_class1_exp = ["class1/streak_m_exp_a1n2",
                    "class1/streak_m_exp_a1n3",
                    "class1/streak_m_exp_a1n4",
                    "class1/streak_m_exp_a1n5",
                    "class1/streak_m_exp_a1ninf"
                    ]
names_class2_poly = ["class2/streak_c_poly_a1n2",
                     "class2/streak_c_poly_a1n3",
                     "class2/streak_c_poly_a1n4",
                     "class2/streak_c_poly_a1n5",
                     "class2/streak_c_poly_a1ninf"]

names_class2_exp = ["class2/streak_c_exp_a1n2",
                    "class2/streak_c_exp_a1n3",
                    "class2/streak_c_exp_a1n4",
                    "class2/streak_c_exp_a1n5",
                    "class2/streak_c_exp_a1ninf"]

paths_list = []
for names in [names_class1_poly, names_class1_exp, names_class2_poly, names_class2_exp]:
    paths = [path + name + ".csv" for name in names]
    paths_list.append(paths)
    
titles = ["Class 1 polynomial decay",
          "Class 1 exponential decay",
          "Class 2 polynomial decay",
          "Class 2 exponential decay"]

labels1 = [f'A=1, n=2', 'A=1, n=3', 'A=1, n=4', 'A=1, n=5', r'A=1, n$\rightarrow \infty$']
labels2 = [f'A=0.25, n=2', 'A=0.25, n=3', 'A=0.25, n=4', 'A=0.25, n=5', r'A=0.25, n$\rightarrow \infty$']
shapes = ['o', 'v', '^', 'D', 's']

# fig, axs = plt.subplots(4, 1, figsize=(4.5, 13.5))
fig, axs = plt.subplots(2, 2, figsize=(8, 7))
coords = [(0, 0), (0, 1),
          (1, 0), (1, 1)]

data_list = []
for paths in paths_list:
    data_correlated = list(filepaths_to_data_correlated(paths))
    data_list.append(data_correlated)
x_fit = np.linspace(3, 60, 200)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color'][:8]

for i in range(len(titles)):
    print(titles[i])
    ax = axs[coords[i]]
    if i in [0, 1]:
        labels = labels1
    else:
        labels = labels2
    for j in range(len(data_list[i])):
        
        x_corr, y_corr, yerr_corr = format_data_for_plotting(data_list[i][j])
        ax.errorbar(x_corr, y_corr, yerr=yerr_corr, fmt=shapes[j], color=colors[j], label=labels[j], markersize=4)
        if j == 0 and i in [0, 2]:
            fit_func = fit_func3
        else:
            fit_func = fit_func2
        popt_corr, pcov = curve_fit(fit_func, x_corr, np.log(y_corr))
        y_fit = np.exp(1)**fit_func(x_fit, *popt_corr)
        ax.plot(x_fit, y_fit, color=colors[j])
        A = real2tex(e**popt_corr[0])
        B = real2tex(popt_corr[1])
        d = np.ceil((np.log(1e-12) - popt_corr[0]) / popt_corr[1]).astype(int)
        print(labels[j], '|', '&', '$'+A, 'e^{'+B+'d}$', '&', d)
        # print(labels[j], popt_corr)
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