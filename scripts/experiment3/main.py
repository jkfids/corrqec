from time import time
from src.noisemodel import LongTimePairAPoly, LongTimePairAExp, LongTimeStreakAPoly, LongTimeStreakAExp
from src.util import gen_csv_filepaths, print_sampling_time
from src.sample import sample_correlated_to_file, sample_independent_to_file

# Experiment parameters
kwargs = {"num_workers": 10,
          "model_params": {"A": 1, "n": 2},
          "distances": [3, 5, 7, 9, 11, 13, 15],
          "probabilities": [1e-3],
          "scl_noise": [],
          # "scl_noise": ["before_round_data_depolarization", "after_clifford_depolarization", "before_measure_flip_probability", "after_reset_flip_probability"],
          "max_shots": 10_000_000,
          "print_progress": True
          }

# Models

models = [#LongTimePairAPoly,
          LongTimeStreakAPoly]
names = [#'pair_all_poly',
         'streak_all_poly']

path = './data/output/experiment3/'
filepaths = []
for name in names:
    filepaths.append(gen_csv_filepaths(path, name, counter=None))

# Run experiments.
if __name__ == "__main__":
    print("Running experiment 3...")
    for i in range(len(models)):
        print(f"Model: {names[i]}")
        sample_correlated_to_file(filepath=filepaths[i][0], NoiseModel=models[i], **kwargs)
        sample_independent_to_file(filepath=filepaths[i][1], NoiseModel=models[i], **kwargs)
    print("Done.")