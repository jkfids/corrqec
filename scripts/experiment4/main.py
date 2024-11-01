from time import time
from src.noisemodel import LongTimePairAPoly, LongTimePairAExp, LongTimeStreakAPoly, LongTimeStreakAExp
from src.util import gen_csv_filepaths, print_sampling_time
from src.sample import sample_correlated_to_file, sample_independent_to_file

# Experiment parameters
# probabilities = np.concatenate([[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3], np.linspace(7e-3, 1.5e-2, 5), [2e-2, 5e-2]])
# probabilities = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 3e-2]
probabilities = [1e-4, 2e-4, 5e-4]
kwargs = {"num_workers": 12,
          "model_params": {"A": 1, "n": 2},
          "distances": [3, 5, 7, 9, 11, 13],
          "probabilities": probabilities,
          "scl_noise": [],
          # "scl_noise": ["before_round_data_depolarization", "after_clifford_depolarization", "before_measure_flip_probability", "after_reset_flip_probability"],
          "max_shots": 10_000_000,
          "print_progress": False
          }

# Models

models = [LongTimePairAPoly,
          LongTimeStreakAPoly]
names = ['pair_all_poly',
         'streak_all_poly']

path = './data/output/experiment4/'
filepaths = []
for name in names:
    filepaths.append(gen_csv_filepaths(path, name, counter=None))
    
# Run experiments.
if __name__ == "__main__":
    print("Running experiment 4...")
    for i in range(len(models)):
        print(f"Model: {names[i]}")
        sample_correlated_to_file(filepath=filepaths[i][0], NoiseModel=models[i], **kwargs)
        sample_independent_to_file(filepath=filepaths[i][1], NoiseModel=models[i], **kwargs)
    print("Done.")