import numpy as np
import matplotlib.pyplot as plt
import sinter
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit

from src.util import stats_from_csv 
from src.analysis import convert_stats_for_projection, collate_stats_for_projection

def filepaths_to_data_correlated(filepaths):
    for filepath in filepaths:
        stats = stats_from_csv(filepath)
        data = convert_stats_for_projection(stats)
        yield data
        
def filepaths_to_data_independent(filepaths):
    for filepath in filepaths:
        stats = sinter.read_stats_from_csv_files(filepath)
        data = collate_stats_for_projection(stats)
        yield data
        
def format_data_for_plotting(data):
    
    x = next(iter(data[0].values()))
    per_rounds = np.array(next(iter(data[1].values())))
    y = per_rounds[:, 0]
    yerr = per_rounds[:, 1:].T
    
    return x, y, yerr

def calc_line_fit(x, y, yerr):
    weights = 1 / yerr
    coeffs = np.polyfit(x, y, 1)
    # coeffs = Polynomial.fit(x, y, 1, domain=[3, 30]).coef
    line_fit = np.poly1d(coeffs)
    
    return line_fit

def fit_func1(x, a, b, c):
    return a + b * x + c * np.log(x)

def fit_func2(x, a, b):
    return a + b * np.log(x)

fit_func = fit_func1

paths_correlated = ["data/saved/experiment1/class0/pair_data_poly_correlated.csv",
                    "data/saved/experiment1/class0/streak_data_poly_correlated.csv",
                    "data/saved/experiment1/class1/pair_m_poly_correlated.csv",
                    "data/saved/experiment1/class1/streak_m_poly_correlated.csv",
                    "data/saved/experiment1/class2/pair_c_poly_correlated.csv",
                    "data/saved/experiment1/class2/streak_c_poly_correlated.csv"]

paths_independent = ["data/saved/experiment1/class0/pair_data_poly_independent.csv",
                    "data/saved/experiment1/class0/streak_data_poly_independent.csv",
                    "data/saved/experiment1/class1/pair_m_poly_independent.csv",
                    "data/saved/experiment1/class1/streak_m_poly_independent.csv",
                    "data/saved/experiment1/class2/pair_c_poly_independent.csv",
                    "data/saved/experiment1/class2/streak_c_poly_independent.csv"]

coords = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
titles = ["Class 0 (Pairwise)", 
          "Class 0 (Streaky)", 
          "Class 1 (Pairwise)", 
          "Class 1 (Streaky)", 
          "Class 2 (Pairwise)", 
          "Class 2 (Streaky)"]

data_correlated = list(filepaths_to_data_correlated(paths_correlated))
data_independent = list(filepaths_to_data_independent(paths_independent))

fig, axs = plt.subplots(2, 3, figsize=(10, 6.5))

for i in range(6):
    ax = axs[coords[i]]
    x_corr, y_corr, yerr_corr = format_data_for_plotting(data_correlated[i])
    x_indep, y_indep, yerr_indep = format_data_for_plotting(data_independent[i])
    if i == 3:
        print(titles[i])
        print(y_corr, y_indep)
    
    line_fit_corr = calc_line_fit(x_corr, np.log10(y_corr), yerr_corr)
    line_fit_indep = calc_line_fit(x_indep, np.log10(y_indep), yerr_indep)
    popt_corr, pcov = curve_fit(fit_func, x_corr, np.log10(y_corr))
    popt_indep, pcov = curve_fit(fit_func, x_indep, np.log10(y_indep))
    

    # Get right colours.
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color1, color2 = prop_cycle.by_key()['color'][:2]
    
    # ax.plot(x_corr, 10**line_fit(x_corr))
    
    ax.errorbar(x_corr, y_corr, yerr=yerr_corr, label='Correlated noise model', fmt='o', markersize=3, color=color1)
    ax.errorbar(x_indep, y_indep, yerr=yerr_indep, label='Independent noise model', fmt='v', markersize=3, color=color2)
    
    x_fit = np.linspace(3, 30, 100)
    if i not in [3, 5]:
        ax.plot(x_fit, 10**line_fit_corr(x_fit), color=color1, label='Correlated fit')
        # ax.plot(x_fit, 10**fit_func(x_fit, *popt_corr), color=color1, label='Correlated fit')
    else:
        ax.plot(x_fit, 10**fit_func(x_fit, *popt_corr), color=color1, label='Correlated fit')
    ax.plot(x_fit, 10**line_fit_indep(x_fit), linestyle='--', color=color2, label='Independent fit')
    # ax.plot(x_fit, 10**fit_func(x_fit, *popt_indep), linestyle='--', color=color2, label='Correlated fit')
    
    ax.semilogy()
    ax.grid()
    ax.set_xlim([0, 30])
    ax.set_ylim([10e-13, 10e-3])
    if i not in [0, 1]:
        ax.set_yticklabels([])
    
    ax.set_title(titles[i])
plt.subplots_adjust(wspace=0, hspace=0)
axs[coords[5]].legend(loc='lower right')

fig.supxlabel('Surface code distance')
fig.supylabel('Logical error rate per round')

np.polynomial.polynomial.Polynomial.fit

# Reorder legend
handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 3, 0, 1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right')

fig.tight_layout()
fig.savefig('figures/output/experiment1.pdf', dpi=600)
fig.savefig('figures/output/experiment1.svg', dpi=600)