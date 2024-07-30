from src.noisemodel import LongTimeStreakPoly
from src.sample import sample_threshold
from src.util import stats_to_csv, gen_filepath

filename = 'streak_data_poly_correlated.csv'
output_path = './data/output/streak_data_poly/'

def sample_correlated(num_workers, model_params, distances, probabilities, scl_noise, max_shots, print_progress):
    filepath = gen_filepath(output_path, filename, override_counter=None)
    stats = sample_threshold(
        num_workers=num_workers,
        NoiseModel=LongTimeStreakPoly,
        model_params=model_params,
        distances=distances,
        probabilities=probabilities,
        scl_noise=scl_noise,
        max_shots=max_shots,
        max_errors=max_shots,
        batch_size=1000,
        print_progress=print_progress
    )
    stats_to_csv(stats, filepath)
    return stats