import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

def plot_temporal_correlation_matrix(corr_matrix, norm_max=0.5, figsize=(6.5, 6.5)):
    fig, ax = plt.subplots(figsize=figsize)
    
    np.fill_diagonal(corr_matrix.values, norm_max)
    
    cmap = colormaps['viridis']
    norm = plt.Normalize(0, norm_max)
    
    rgba = cmap(norm(corr_matrix))
    
    # white out main diagonal (self-correlations)
    l = rgba.shape[0]
    rgba[range(l), range(l), :3] = 1, 1, 1
    
    # draw minor and major ticks
    n_half = int(0.5*corr_matrix.shape[0])
    major_loc = np.arange(corr_matrix.shape[0])
    # major_labels = [i for i in range(n_half)] + [i for i in range(n_half)]
    ax.set_xticks(major_loc)
    ax.set_yticks(major_loc)
    
    # color bar
    im = ax.imshow(corr_matrix, visible=False, cmap=cmap, origin='lower')
    fig.colorbar(im, ax=ax, fraction=0.0457, pad=0.04)
    
    # Axis label
    ax.set_xlabel("Round, $t$")
    ax.set_ylabel("Round, $t'$")
    
    # plot
    ax.imshow(rgba, origin='lower')
    
    fig.tight_layout()
    fig.patch.set_alpha(0)
    
    return fig