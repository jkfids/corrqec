from time import time
from src.noisemodel import LongTimePairPoly, LongTimeStreakPoly
from src.util import gen_csv_filepaths, print_sampling_time
from src.sample import sample_correlated_to_file, sample_independent_to_file

# Experiment parameters
kwargs = {"num_workers": 12,
          "model_params": {"A": 1, "n": 2, "noisy_qubits": "data"},
          "distances": [3, 5, 7, 9, 11, 13, 15],
          "probabilities": [2e-3],
          "scl_noise": ["after_clifford_depolarization", "before_measure_flip_probability", "after_reset_flip_probability"],
          "max_shots": 10_000_000,
          "print_progress": True
          }

# Models
model1 = LongTimePairPoly
name1 = 'pair_data_poly'
model2 = LongTimeStreakPoly
name2 = 'streak_data_poly'

path = './data/output/experiment1/class0/'
fp1a, fp1b = gen_csv_filepaths(path, name1, counter=None)
fp2a, fp2b = gen_csv_filepaths(path, name2, counter=None)

# Run experiments.
if __name__ == '__main__':
    print("Running Class 0 experiments...")
    time0 = time()
    sample_correlated_to_file(filepath=fp1a, NoiseModel=model1, **kwargs)
    time1 = time()
    print_sampling_time(time0, time1, "Pairwise correlated")

    sample_independent_to_file(filepath=fp1b, NoiseModel=model1, **kwargs)
    time2 = time()
    print_sampling_time(time1, time2, "Pairwise independent")
    
    sample_correlated_to_file(filepath=fp2a, NoiseModel=model2, **kwargs)
    time3 = time()
    print_sampling_time(time2, time3, "Streak correlated")
    
    sample_independent_to_file(filepath=fp2b, NoiseModel=model2, **kwargs)
    time4 = time()
    print_sampling_time(time3, time4, "Streak independent")