import numpy as np
from time import time
from src.util import print_sampling_time

from .correlated import sample_correlated
from .independent import sample_independent


num_workers = 16  # Number of threads
print_progress = False

# Experiment parameters.
A = 1
n = 2
model_params = {"A": A, "n": n, "noisy_qubits": "data"}
scl_noise = ["after_clifford_depolarization", "before_measure_flip_probability", "after_reset_flip_probability"]

distances = [3, 5, 7, 9, 11, 13, 15]
# probabilities = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 3e-2]
probabilities = [5e-4, 1e-3, 2e-3, 5e-3]

max_shots = int(10_000)

# Run experiments.
if __name__ == '__main__':
    time0 = time()
    sample_correlated(num_workers, model_params, distances, probabilities, scl_noise, max_shots, print_progress)
    time1 = time()
    print_sampling_time(time0, time1, "Correlated")

    sample_independent(num_workers, model_params, distances, probabilities, scl_noise, max_shots, print_progress)
    time2 = time()
    print_sampling_time(time1, time2, "Independent")