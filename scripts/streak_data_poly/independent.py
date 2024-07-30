import sinter
from typing import List
from src.noisemodel import LongTimeStreakPoly
from src.util import collected_stats_to_csv, gen_filepath

filename = 'streak_data_poly_independent.csv'
output_path = './data/output/streak_data_poly/'

def sample_independent(num_workers, model_params, distances, probabilities, scl_noise, max_shots, print_progress):
    filepath = gen_filepath(output_path, filename, override_counter=None)
    tasks = [
        sinter.Task(
            circuit=LongTimeStreakPoly(p=p, **model_params).gen_circuit_marginalised(distance=d, rounds=d*2, scl_noise=scl_noise, rotated=True),
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
        max_errors=max_shots,
        print_progress=print_progress,
    )
    collected_stats_to_csv(collected_stats, filepath)
    return collected_stats