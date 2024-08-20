from typing import List

import numpy as np
import stim
import sinter

from src.util import gen_filepath, collected_stats_to_csv


num_workers = 16  # Number of threads to use for parallel sampling.
print_progress = False

filename = 'rotated_memory_z_scl.csv'
output_path = './data/output/sinterstats/'
filepath = gen_filepath(output_path, filename, override_counter=None)

# Experiment parameters.
distances = [3, 5, 7, 9, 11, 13, 15]
probabilities = np.concatenate([[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3], np.linspace(7e-3, 1.5e-2, 5), [2e-2, 5e-2]])
max_shots = 10_000

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
        max_errors=max_shots,
        print_progress=print_progress,
    )
    collected_stats_to_csv(collected_stats, filepath)