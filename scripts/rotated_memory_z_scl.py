# Standard library imports
from typing import List

# Third-party library imports
import numpy as np
import stim
import sinter

# Local imports
from src.util import gen_filepath, collected_stats_to_csv


filename = 'rotated_memory_z_scl.csv'
output_path = './data/output/sinterstats/'

num_workers = 16  # Number of threads to use for parallel sampling.

# Experiment parameters.
distances = [3, 5, 7]
probabilities = np.concatenate([[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3], np.linspace(7e-3, 1.5e-2, 5), [2e-2, 5e-2]])

max_shots = 10_000
max_errors = max_shots

filepath = gen_filepath(output_path, filename, override_counter=None)

if __name__ == '__main__':
    tasks = [
        sinter.Task(
            circuit=stim.Circuit.generated("surface_code:rotated_memory_z",
                                            rounds=d * 2,
                                            distance=d,
                                            after_clifford_depolarization=p,
                                            after_reset_flip_probability=p,
                                            before_measure_flip_probability=p,
                                            before_round_data_depolarization=p,
                                            ),
            json_metadata={'d': d, 'r': d * 2, 'p': p},
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

    collected_stats_to_csv(collected_stats, filepath)