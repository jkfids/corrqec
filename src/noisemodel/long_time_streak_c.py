import numpy as np
from .long_time_streak import LongTimeStreak, LongTimeStreakPoly, LongTimeStreakExp
from .long_time_pair_c import LongTimePairC
from .noise_model_util import get_round_pair_distances, calc_marginals_mixing

class LongTimeStreakC(LongTimePairC, LongTimeStreak):
    def __init__(self, interaction_func=None):
        super().__init__(interaction_func=interaction_func, noisy_qubits="all", error_type='depolarizing')
        self.error_type = "depolarizing"
        
    def calc_marginals_per_round(self, rounds: int) -> np.ndarray:
        marginals = np.zeros(rounds)
        round_pair_distances = get_round_pair_distances(rounds=rounds)
        round_pair_probabilities = self.interaction_func(round_pair_distances)
        for i in range(round_pair_probabilities.shape[0]):
            probabilities = np.concatenate([round_pair_probabilities[i, :], round_pair_probabilities[:i, i:].flatten()])
            p = calc_marginals_mixing(probabilities)
            marginals[i] = p
            marginals[-i-1] = p
        return marginals * 0.9375
    
class LongTimeStreakCPoly(LongTimeStreakC, LongTimeStreakPoly):
    def __init__(self, A, p, n): 
        LongTimeStreakPoly.__init__(self, A, p, n)
        self.noisy_qubits = "all"  # Set from LongTimeStreakC
        self.error_type = "depolarizing"
        
class LongTimeStreakCExp(LongTimeStreakC, LongTimeStreakExp):
    def __init__(self, A, p, n): 
        LongTimeStreakExp.__init__(self, A, p, n)
        self.noisy_qubits = "all"  # Set from LongTimeStreakC
        self.error_type = "depolarizing"