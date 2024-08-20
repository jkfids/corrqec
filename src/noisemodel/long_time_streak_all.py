from numpy import ndarray
from .long_time_pair_all import LongTimePairA
from .long_time_streak import LongTimeStreak, LongTimeStreakPoly, LongTimeStreakExp
from .long_time_streak_c import LongTimeStreakC


class LongTimeStreakA(LongTimePairA, LongTimeStreak):
    def __init__(self, interaction_func=None):
        super().__init__(interaction_func=interaction_func, noisy_qubits="all", error_type='depolarizing')
        self.noisy_qubits = "all"
        self.error_type = "depolarizing"
        
    def _calc_c_marginals_per_round(self, rounds: int) -> ndarray:
        return LongTimeStreakC.calc_marginals_per_round(self, rounds)
        
class LongTimeStreakAPoly(LongTimeStreakA, LongTimeStreakPoly):
    def __init__(self, A, p, n): 
        LongTimeStreakPoly.__init__(self, A, p, n)
        self.noisy_qubits = "all"  # Set from LongTimeStreakA
        self.error_type = "depolarizing"
        
class LongTimeStreakAExp(LongTimeStreakA, LongTimeStreakExp):
    def __init__(self, A, p, n): 
        LongTimeStreakExp.__init__(self, A, p, n)
        self.noisy_qubits = "all"  # Set from LongTimeStreakA
        self.error_type = "depolarizing"