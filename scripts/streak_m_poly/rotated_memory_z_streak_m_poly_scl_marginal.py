# Standard library imports
from time import time
from typing import List

# Third-party library imports
import numpy as np
import sinter

# Local imports
from src.noisemodel import LongTimeStreakMPoly
from src.util import gen_filepath, collected_stats_to_csv

filename = 'rotated_memory_z_streak_m_poly_scl_marginal.csv'
output_path = './data/output/sinterstats/'

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
    tasks = [
        sinter.Task(
            circuit=model(A=A, p=p, n=n).gen_circuit_marginalised(distance=d, rounds=d*2, scl_noise=scl_noise, rotated=rotated),
            json_metadata={'d': d, 'r': d*2, 'p': p},
        )
        for d in distances
        for p in probabilities
    ]

    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        save_resume_filepath=filepath,
        decoders=['pymatching'],
        max_shots=max_shots,
        max_errors=max_errors,
        # print_progress=True,
    )
    end = time()
    print(f"Time taken: {(end-start)/60} minutes")

    collected_stats_to_csv(collected_stats, filepath)