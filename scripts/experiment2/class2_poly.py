from time import time
from src.noisemodel import LongTimeStreakCPoly, LongTimeStreakCExp
from src.util import gen_csv_filepath_list
from src.sample import sample_correlated_to_file, sample_independent_to_file

from src.noisemodel import poly_decay

# Experiment parameters
kwargs = {"num_workers": 12,
          "distances": [3, 5, 7, 9, 11, 13, 15],
          "probabilities": [2e-3],
          "scl_noise": ["before_round_data_depolarization", "after_clifford_depolarization", "after_reset_flip_probability"],
          "max_shots": 1_000_000,
          "print_progress": False
          }

# Models
models = [LongTimeStreakCPoly] * 5
A = 0.25
params_list = [{"A": A, "n": 2},
               {"A": A, "n": 3},
               {"A": A, "n": 4},
               {"A": A, "n": 5},
               {"A": A, "n": float('inf')}]

path = './data/output/experiment2/class2/'
names = ["streak_c_poly_a1n2", 
         "streak_c_poly_a1n3", 
         "streak_c_poly_a1n4", 
         "streak_c_poly_a1n5", 
         "streak_c_poly_a1ninf"]

filepaths = gen_csv_filepath_list(path, names)
    
# Run experiments.
if __name__ == "__main__":
    print("Running class 2 poly experiments...")
    for i in range(len(models)):
        sample_correlated_to_file(filepath=filepaths[i], NoiseModel=models[i], model_params=params_list[i], **kwargs)
    print("Done.")
        
        