# Standard library imports
from time import time

# Third-party library imports
import numpy as np

# Local imports
from src.noisemodel import LongTimeStreakMPoly
from src.sample import sample_threshold
from src.util import gen_filepath, stats_to_csv

filename = 'rotated_memory_z_streak_m_poly_scl.csv'
output_path = './data/output/stats/'

num_workers = 16  # Number of threads to use for parallel sampling.

# Experiment parameters.
model = LongTimeStreakMPoly
A = 1
n = 2
rotated = True
scl_noise = ["before_round_data_depolarization", "after_clifford_depolarization"]

distances = [3, 5, 7, 9, 11, 13, 15]
probabilities = np.concatenate([[1e-3, 2e-3, 5e-3], np.linspace(1e-2, 5e-2, 5)])

max_shots = int(1_000_000)
max_errors = max_shots
batch_size = 1000

filepath = gen_filepath(output_path, filename, override_counter=None)

if __name__ == '__main__':
    start = time()
    stats = sample_threshold(
        num_workers=num_workers,
        NoiseModel=model,
        distances=distances,
        probabilities=probabilities,
        scl_noise=scl_noise,
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        A=A,
        n=n
    )
    end = time()
    print(f"Time taken: {(end-start)/60} minutes")

    stats_to_csv(stats, filepath)