import matplotlib.pyplot as plt
import numpy as np

from src.noisemodel import LongTimePairPoly, LongTimeStreakPoly, LongTimePairMPoly, LongTimeStreakMPoly, LongTimePairCPoly, LongTimePairCPoly, LongTimeStreakCPoly
from src.sample import sample_tcorr_matrix
from src.visualization import plot_temporal_correlation_matrix

# Universal parameters
shots = 1_000_000
d = 3
r = 10
A = 1
n = 2
p = 2e-3

# Models
model1a_kwargs = {"NoiseModel": LongTimePairPoly,
                 "model_params": {"p": p, "A": A, "n": n, "noisy_qubits": "data"},
                 "scl_noise": ["after_clifford_depolarization", "before_measure_flip_probability", "after_reset_flip_probability"],
                 }

model1b_kwargs = {"NoiseModel": LongTimeStreakPoly,
                 "model_params": {"p": p, "A": A, "n": n, "noisy_qubits": "data"},
                 "scl_noise": ["after_clifford_depolarization", "before_measure_flip_probability", "after_reset_flip_probability"],
                 }

model2a_kwargs = {"NoiseModel": LongTimePairMPoly,
                 "model_params": {"p": p, "A": A, "n": n},
                 "scl_noise": ["before_round_data_depolarization", "after_clifford_depolarization", "after_reset_flip_probability"],
                 }

model2b_kwargs = {"NoiseModel": LongTimeStreakMPoly,
                  "model_params": {"p": p, "A": A, "n": n},
                  "scl_noise": ["before_round_data_depolarization", "after_clifford_depolarization", "after_reset_flip_probability"],
                  }

model3a_kwargs = {"NoiseModel": LongTimePairCPoly,
                  "model_params": {"p": p, "A": .5, "n": n},
                  "scl_noise": ["before_round_data_depolarization", "before_measure_flip_probability", "after_reset_flip_probability"],
                  }

model3b_kwargs = {"NoiseModel": LongTimeStreakCPoly,
                  "model_params": {"p": p, "A": .5, "n": n},
                  "scl_noise": ["before_round_data_depolarization", "before_measure_flip_probability", "after_reset_flip_probability"],
                  }

model_kwargs_list = [model1a_kwargs, model1b_kwargs, model2a_kwargs, model2b_kwargs, model3a_kwargs, model3b_kwargs]
labels = ["Class 0 (Pairwise)", 
          "Class 0 (Streaky)", 
          "Class 1 (Pairwise)", 
          "Class 1 (Streaky)", 
          "Class 2 (Pairwise)", 
          "Class 2 (Streaky)"]

if __name__ == "__main__":
    corr_matrix_list = []
    for i in range(len(model_kwargs_list)):
        corr_matrix = sample_tcorr_matrix(**model_kwargs_list[i], distance=d, rounds=r, shots=shots)
        corr_matrix_list.append(corr_matrix)
    
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [color for color in prop_cycle.by_key()['color'][:3] for _ in range(2)]
    markers = ['o', 'v'] * 3
    linestyles = ['-', '--'] * 3
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    
    for i in range(len(corr_matrix_list)):
        matrix = corr_matrix_list[i]
        X = list(range(2, matrix.shape[0]))
        Y = [np.mean(np.diagonal(matrix, offset=i)) for i in X]
        ax.plot(X, Y, marker=markers[i], linestyle=linestyles[i], color=colors[i], label=labels[i])
            
    ax.semilogy()
    ax.legend()
    
    # Axis label
    ax.set_xlabel("Round distance, $t-t'$")
    ax.set_ylabel("Average correlation")
    ax.grid()
    ax.grid(True, which='both', axis='y')
    
    fig.tight_layout()
    fig.patch.set_alpha(0)
    
    fig.savefig("figures/output/experiment1_mean_tcorr.pdf", dpi=600)
    fig.savefig("figures/output/experiment1_mean_tcorr.svg", dpi=600)