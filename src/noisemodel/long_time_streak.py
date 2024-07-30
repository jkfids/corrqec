import numpy as np
from .long_time_pair import LongTimePair, LongTimePairExp, LongTimePairPoly

from .noise_model_util import get_round_pairs, get_round_pair_distances, calc_marginals

class LongTimeStreak(LongTimePair):
    def __init__(self, interaction_func=None, noisy_qubits=None, error_type="depolarizing"):
        super().__init__(interaction_func=interaction_func, noisy_qubits=noisy_qubits, error_type=error_type)
    
    def gen_pair_to_qubit_map(self, rounds: int):
        """
        Generates a pair-to-qubit map for a given number of rounds to be used in streaky correlated error sampling.

        Args:
            rounds (int): The number of rounds.

        Returns:
            np.ndarray: A 2D numpy array representing the pair-to-qubit map. Each row corresponds to a pair, and each column corresponds to a round.
        """
        pairs = get_round_pairs(rounds=rounds)
        pair_to_qubit_map = np.zeros((len(pairs), rounds), dtype=int)
        
        for i, (a, b) in enumerate(pairs):
            pair_to_qubit_map[i][a:b+1] = 1 # Streak or errors across qubits [a, b]
            
        return pair_to_qubit_map
    
    def calc_pair_probabilities_coefficient(self) -> float:
        """
        Calculates the coefficient multiplier for the pair probabilities vector.

        Returns:
            float: The coefficient multiplier for the pair probabilities vector.
        """
        return 1.
    
    def int_to_bool_errors(self, errors: np.ndarray) -> np.ndarray:
        """
        Converts integer errors to boolean errors at .5 probability for the streaky model.

        Args:
            errors (np.ndarray): The integer errors to be converted.

        Returns:
            np.ndarray: The boolean errors.
        """
        rng = np.random.default_rng()
        errors = errors.astype(bool)
        random_bits = rng.integers(0, 1, size=errors.shape, endpoint=True)
        return random_bits * errors
    
    def calc_marginals_per_round(self, rounds: int):
        """
        Calculate the marginal error probabilities for each round under the streaky model.

        Args:
            rounds (int): The number of rounds in the circuit.

        Returns:
            np.ndarray: Array of marginal error probabilities for each round.
        """
        marginals = np.zeros(rounds)
        round_pair_distances = get_round_pair_distances(rounds=rounds)
        round_pair_probabilities = self.interaction_func(round_pair_distances)
        for i in range(round_pair_probabilities.shape[0]):
            
            probabilities = np.concatenate([round_pair_probabilities[i, :], round_pair_probabilities[:i, i:].flatten()])  # Additional streak terms
            p = calc_marginals(probabilities)
            marginals[i] = p
            marginals[-i-1] = p
        if self.error_type  == "depolarizing":
            marginals *= .75
        else:
            marginals *= .5
        return marginals
    
class LongTimeStreakPoly(LongTimeStreak, LongTimePairPoly):
    def __init__(self, A, p, n, noisy_qubits=None, error_type="depolarizing"): 
        LongTimePairPoly.__init__(self, A, p, n, noisy_qubits, error_type)

class LongTimeStreakExp(LongTimeStreak, LongTimePairExp):
    def __init__(self, A, p, n, noisy_qubits=None, error_type="depolarizing"): 
        LongTimePairExp.__init__(self, A, p, n, noisy_qubits, error_type)