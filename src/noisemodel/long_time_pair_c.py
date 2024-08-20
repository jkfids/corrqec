import numpy as np
import stim

from .long_time_pair import LongTimePair, LongTimePairPoly, LongTimePairExp
from .noise_model_util import split_circuit, split_circuit_cx, get_control_qubits, calc_marginals_mixing, get_round_pair_distances

class LongTimePairC(LongTimePair):
    def __init__(self, interaction_func=None):
        super().__init__(interaction_func=interaction_func, noisy_qubits="all", error_type='depolarizing')
        
    def get_targets(self, circuit: stim.Circuit):
        return get_control_qubits(circuit)
        
    def _sample_circuit(self, split_circuits: tuple, targets: list, n_qubits: int, rounds: int, batch_size: int):
        circuit_init, circuit_init_round, circuit_repeat_block, circuit_final = split_circuits

        sim = stim.FlipSimulator(batch_size=batch_size)
        
        errors_list = self.sample_control_errors(targets=targets, n_qubits=n_qubits, rounds=rounds, batch_size=batch_size)
        sim.do(circuit_init)
        for i in range(4):
            sim.do(circuit_init_round[i])
            for j, pauli in enumerate(['X', 'Y', 'Z']):
                sim.broadcast_pauli_errors(pauli=pauli, mask=errors_list[i][j][0])
        sim.do(circuit_init_round[4])
        
        for k in range(1, rounds):
            for i in range(4):
                sim.do(circuit_repeat_block[i])
                for j, pauli in enumerate(['X', 'Y', 'Z']):
                    sim.broadcast_pauli_errors(pauli=pauli, mask=errors_list[i][j][k])
            sim.do(circuit_repeat_block[4])
            
        sim.do(circuit_final)
        
        detection_events = sim.get_detector_flips().transpose()
        observable_flips = sim.get_observable_flips().flatten()
        
        return detection_events, observable_flips
                
    def sample_control_errors(self, targets: list, n_qubits: int, rounds: int, batch_size: int):
        n_targets = int(len(targets[0]) * 0.5)  # Number of target pairs per CX layer
        rng = np.random.default_rng()
        
        errors_list = []
        
        for targetsi in targets:
            pair_probabilities = self.calc_pair_probabilities(rounds=rounds)
            pair_probabilities_matrix = np.tile(pair_probabilities, (n_targets * batch_size, 1))
            
            rnd = rng.random(pair_probabilities_matrix.shape)
            pair_errors = rnd < pair_probabilities_matrix
            
            map = self.gen_pair_to_qubit_map(rounds)
            c_errors = np.matmul(pair_errors, map).reshape(batch_size, n_targets, rounds).transpose(2, 1, 0)
            c_errors = c_errors.astype(bool)
            
            # Convert control (two-qubit) errors to (single-qubit) qubit errors
            qubit_errors = np.zeros((rounds, n_targets * 2, batch_size), dtype=bool)
            qubit_errors[:, ::2, :] = c_errors
            qubit_errors[:, 1::2, :] = c_errors
            
            # Roll for Paulis
            random_paulis = rng.integers(0, 3, size=qubit_errors.shape, endpoint=True)
            pauli_errors = np.zeros([rounds, n_qubits, batch_size], dtype=int)
            pauli_errors[:, targetsi, :] = random_paulis * qubit_errors
            X_errors = pauli_errors == 1
            Y_errors = pauli_errors == 2
            Z_errors = pauli_errors == 3
            
            errors_list.append((X_errors, Y_errors, Z_errors))
            
        return errors_list
    
    def split_circuit(self, circuit: stim.Circuit) -> tuple:
        split_circuits, repeat_count = super().split_circuit(circuit)
        circuit_init, circuit_init_round, circuit_repeat_block, circuit_final = split_circuits
        circuit_init_round = split_circuit_cx(circuit_init_round)
        circuit_repeat_block = split_circuit_cx(circuit_repeat_block)
        
        return (circuit_init, circuit_init_round, circuit_repeat_block, circuit_final), repeat_count
        
    def convert_circuit_marginalised(self, circuit: stim.Circuit) -> stim.Circuit:
        output_circuit = stim.Circuit()
        split_circuits, repeat_count = self.split_circuit(circuit)
        circuit_init, circuit_init_round, circuit_repeat_block, circuit_final = split_circuits
        rounds = repeat_count + 1
        
        marginals = self.calc_marginals_per_round(rounds=rounds)
        output_circuit += circuit_init
        
        targets = self.get_targets(circuit)
        for i in range(4):
            output_circuit += circuit_init_round[i]
            output_circuit.append('DEPOLARIZE2', targets[i], marginals[0])
        output_circuit += circuit_init_round[4]
        
        for j in range(1, rounds):
            for i in range(4):
                output_circuit += circuit_repeat_block[i]
                output_circuit.append('DEPOLARIZE2', targets[i], marginals[j])
            output_circuit += circuit_repeat_block[4]
            
        output_circuit += circuit_final
        
        return output_circuit
        
    # def calc_marginals_per_round(self, rounds: int) -> np.ndarray:
    #     return super().calc_marginals_per_round(rounds) * 1.17708333333  # (1/0.75) * (15/16)^2 = 1.171875
    
    def calc_marginals_per_round(self, rounds: int) -> np.ndarray:
        marginals = np.zeros(rounds)
        round_pair_distances = get_round_pair_distances(rounds=rounds)
        round_pair_probabilities = self.interaction_func(round_pair_distances)
        for i in range(round_pair_probabilities.shape[0]):
            p = calc_marginals_mixing(round_pair_probabilities[i, :])
            marginals[i] = p
            marginals[-i-1] = p
        return marginals * 0.9375
        
class LongTimePairCPoly(LongTimePairC, LongTimePairPoly):
    def __init__(self, A, p, n): 
        LongTimePairPoly.__init__(self, A, p, n, noisy_qubits="all")
        self.noisy_qubits = "all"  # Set from LongTimePairC
        self.error_type = "depolarizing"

class LongTimePairCExp(LongTimePairC, LongTimePairExp):
    def __init__(self, A, p, n): 
        LongTimePairExp.__init__(self, A, p, n, noisy_qubits="all")
        self.noisy_qubits = "all"  # Set from LongTimePairC
        self.error_type = "depolarizing"