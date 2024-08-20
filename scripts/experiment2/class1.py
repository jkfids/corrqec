from time import time
from src.noisemodel import LongTimeStreakMPoly, LongTimeStreakMExp
from src.util import gen_csv_filepath_list
from src.sample import sample_correlated_to_file, sample_independent_to_file

from src.noisemodel import poly_decay

# Experiment parameters
kwargs = {"num_workers": 12,
          "distances": [3, 5, 7, 9, 11, 13, 15],
          "probabilities": [2e-3],
          "scl_noise": ["before_round_data_depolarization", "after_clifford_depolarization", "after_reset_flip_probability"],
          "max_shots": 10_000_000,
          "print_progress": False
          }

# Models
models = [LongTimeStreakMPoly] * 3
params_list = [{"A": 1, "n": 2},
               {"A": 1, "n": 5},
               {"A": 1, "n": float('inf')}]

path = './data/output/experiment2/class1/'
names = ["streak_m_poly_a1n2", 
         "streak_m_poly_a1n5", 
         "streak_m_poly_a1ninf"]

filepaths = gen_csv_filepath_list(path, names)
    
# Run experiments.
if __name__ == "__main__":
    print("Running Class 1 experiments...")
    for i in range(len(models)):
        sample_correlated_to_file(filepath=filepaths[i], NoiseModel=models[i], model_params=params_list[i], **kwargs)
    print("Done.")
        
        