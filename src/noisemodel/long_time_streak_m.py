from .long_time_streak import LongTimeStreak, LongTimeStreakPoly, LongTimeStreakExp
from .long_time_pair_m import LongTimePairM

class LongTimeStreakM(LongTimePairM, LongTimeStreak):
    def __init__(self, interaction_func=None):
        super().__init__(interaction_func=interaction_func, noisy_qubits="syndrome", error_type='X')
        self.error_type = 'X'
        self.split_measurements = True
        
class LongTimeStreakMPoly(LongTimeStreakM, LongTimeStreakPoly):
    def __init__(self, A, p, n): 
        LongTimeStreakPoly.__init__(self, A, p, n)
        # Override noisy_qubits and error_type from LongTimePairM
        self.noisy_qubits = "syndrome"  # Set from LongTimePairM
        self.error_type = 'X'
        self.split_measurements = True
        
class LongTimeStreakMExp(LongTimeStreakM, LongTimeStreakExp):
    def __init__(self, A, p, n): 
        LongTimeStreakExp.__init__(self, A, p, n)
        # Override noisy_qubits and error_type from LongTimePairM
        self.noisy_qubits = "syndrome"  # Set from LongTimePairM
        self.error_type = 'X'
        self.split_measurements = True