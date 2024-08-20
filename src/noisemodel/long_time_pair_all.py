import numpy as np
import stim

from .long_time_pair import LongTimePair, LongTimePairPoly, LongTimePairExp
from.long_time_pair_c import LongTimePairC
from.noise_model_util import get_partitioned_qubit_coords, get_control_qubits, split_circuit_cx_m

class LongTimePairA(LongTimePair):
    def __init__(self, interaction_func=None):
        super().__init__(interaction_func=interaction_func, noisy_qubits="all")
        
    def _sample_circuit(self, split_circuits: tuple, targets: tuple, n_qubits: int, rounds: int, batch_size: int):
        
        circuit_init, circuit_init_round, circuit_repeat_block, circuit_final = split_circuits
        sim = stim.FlipSimulator(batch_size=batch_size)
    
        d_errors, m_errors, c_errors = self.sample_errors(targets=targets, n_qubits=n_qubits, rounds=rounds, batch_size=batch_size)
        
        sim.do(circuit_init)
        for j, pauli in enumerate(['X', 'Y', 'Z']):
            sim.broadcast_pauli_errors(pauli=pauli, mask=d_errors[j][0])
        for i in range(4):
            sim.do(circuit_init_round[i])
            for j, pauli in enumerate(['X', 'Y', 'Z']):
                sim.broadcast_pauli_errors(pauli=pauli, mask=c_errors[i][j][0])
        sim.do(circuit_init_round[4])
        sim.broadcast_pauli_errors(pauli='X', mask=m_errors[0])
        sim.do(circuit_init_round[5])
        
        for k in range(1, rounds):
            for j, pauli in enumerate(['X', 'Y', 'Z']):
                sim.broadcast_pauli_errors(pauli=pauli, mask=d_errors[j][k])
            for i in range(4):
                sim.do(circuit_repeat_block[i])
                for j, pauli in enumerate(['X', 'Y', 'Z']):
                    sim.broadcast_pauli_errors(pauli=pauli, mask=c_errors[i][j][k])
            sim.do(circuit_repeat_block[4])
            sim.broadcast_pauli_errors(pauli='X', mask=m_errors[k])
            sim.do(circuit_repeat_block[5])
        
        sim.do(circuit_final)
        
        detection_events = sim.get_detector_flips().transpose()
        observable_flips = sim.get_observable_flips().flatten()
        
        return detection_events, observable_flips
        
    def sample_errors(self, targets: tuple, n_qubits: int, rounds: int, batch_size: int):
        d_targets, m_targets, c_targets = targets
        args = (n_qubits, rounds, batch_size)
        d_errors = self._sample_errors("depolarizing", d_targets, *args)
        m_errors = self._sample_errors("X", m_targets, *args)
        
        # jank but it works
        A_orig = self.A
        self.A = 0.5
        c_errors = LongTimePairC.sample_control_errors(self, c_targets, *args)
        self.A = A_orig
        
        return d_errors, m_errors, c_errors
        
    def get_targets(self, circuit: stim.Circuit):
        data, sz, sx = get_partitioned_qubit_coords(circuit)
        data_coords = data
        syndrome_coords = {**sz, **sx}
        d_targets = list(data_coords.keys())
        m_targets = list(syndrome_coords.keys())
        c_targets = get_control_qubits(circuit)
        return d_targets, m_targets, c_targets
    
    def split_circuit(self, circuit: stim.Circuit) -> tuple:
        split_circuits, repeat_count = super().split_circuit(circuit)
        circuit_init, circuit_init_round, circuit_repeat_block, circuit_final = split_circuits
        circuit_init_round = split_circuit_cx_m(circuit_init_round)
        circuit_repeat_block = split_circuit_cx_m(circuit_repeat_block)
        
        return (circuit_init, circuit_init_round, circuit_repeat_block, circuit_final), repeat_count
    
    def convert_circuit_marginalised(self, circuit: stim.Circuit) -> stim.Circuit:
        output_circuit = stim.Circuit()
        split_circuits, repeat_count = self.split_circuit(circuit)
        circuit_init, circuit_init_round, circuit_repeat_block, circuit_final = split_circuits
        rounds = repeat_count + 1
        
        d_targets, m_targets, c_targets = self.get_targets(circuit)
        d_marginals, m_marginals, c_marginals = self.calc_marginals_per_round(rounds)
        
        # First round
        output_circuit += circuit_init
        output_circuit.append('DEPOLARIZE1', d_targets, d_marginals[0])
        for i in range(4):
            output_circuit += circuit_init_round[i]
            output_circuit.append('DEPOLARIZE2', c_targets[i], c_marginals[0])
        output_circuit += circuit_init_round[4]
        output_circuit.append('X_ERROR', m_targets, m_marginals[0])
        output_circuit += circuit_init_round[5]
        
        # Other rounds
        for j in range(1, rounds):
            output_circuit.append('DEPOLARIZE1', d_targets, d_marginals[j])
            for i in range(4):
                output_circuit += circuit_repeat_block[i]
                output_circuit.append('DEPOLARIZE2', c_targets[i], c_marginals[j])
            output_circuit += circuit_repeat_block[4]
            output_circuit.append('X_ERROR', m_targets, m_marginals[j])
            output_circuit += circuit_repeat_block[5]
            
        output_circuit += circuit_final
        
        return output_circuit
        
    def calc_marginals_per_round(self, rounds: int) -> np.ndarray:
        d_marginals = self._calc_marginals_per_round(rounds, "depolarizing")
        m_marginals = self._calc_marginals_per_round(rounds, "X")
        A_orig = self.A
        self.A = 0.5 * A_orig
        c_marginals = self._calc_c_marginals_per_round(rounds)
        self.A = A_orig
        return d_marginals, m_marginals, c_marginals
    
    def _calc_c_marginals_per_round(self, rounds: int) -> np.ndarray:
        return LongTimePairC.calc_marginals_per_round(self, rounds)
        
class LongTimePairAPoly(LongTimePairA, LongTimePairPoly):
    def __init__(self, A, p, n):
        LongTimePairPoly.__init__(self, A, p, n, noisy_qubits="all")
        
class LongTimePairAExp(LongTimePairA, LongTimePairExp):
    def __init__(self, A, p, n):
        LongTimePairExp.__init__(self, A, p, n, noisy_qubits="all")
    