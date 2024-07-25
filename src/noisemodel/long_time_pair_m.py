import stim

from .long_time_pair import LongTimePair, LongTimePairPoly, LongTimePairExp
from .noise_model_util import split_circuit

class LongTimePairM(LongTimePair):
    def __init__(self, interaction_func=None):
        super().__init__(interaction_func=interaction_func, noisy_qubits="syndrome", error_type='X')
        self.split_measurements = True
        
    def _sample_circuit(self, split_circuits, targets, n_qubits, rounds, batch_size):
        
        circuit_init, circuit_init_round, circuit_repeat_block, circuit_final = split_circuits
        
        sim = stim.FlipSimulator(batch_size=batch_size)
        
        errors = self.sample_errors(targets=targets, n_qubits=n_qubits, rounds=rounds, batch_size=batch_size)
        sim.do(circuit_init)
        sim.do(circuit_init_round[0])
        sim.broadcast_pauli_errors(pauli=self.error_type, mask=errors[0])
        sim.do(circuit_init_round[1])
        
        for j in range(1, rounds):
            
            sim.do(circuit_repeat_block[0])
            sim.broadcast_pauli_errors(pauli=self.error_type, mask=errors[j])
            sim.do(circuit_repeat_block[1])
            
        sim.do(circuit_final)
        
        detection_events = sim.get_detector_flips().transpose()
        observable_flips = sim.get_observable_flips().flatten()
        
        return detection_events, observable_flips
    
    def convert_circuit_marginalised(self, circuit: stim.Circuit):
        
        output_circuit = stim.Circuit()
        
        split_circuits, repeat_count = split_circuit(circuit=circuit, split_measurements=True)
        circuit_init, circuit_init_round, circuit_repeat_block, circuit_final = split_circuits
            
        qubit_coords = self.get_noisy_qubit_coords(circuit)
        
        qubits = qubit_coords.keys()
        
        rounds = repeat_count + 1
        
        marginals = self.calc_marginals_per_round(rounds=rounds)
        
        output_circuit += circuit_init
        
        output_circuit += circuit_init_round[0]
        output_circuit.append('X_ERROR', qubits, marginals[0])
        output_circuit += circuit_init_round[1]
        
        for i in range(1, rounds):
            output_circuit += circuit_repeat_block[0]
            output_circuit.append('X_ERROR', qubits, marginals[i])
            output_circuit += circuit_repeat_block[1]
        
        output_circuit += circuit_final
        
        return output_circuit
    
class LongTimePairMPoly(LongTimePairM, LongTimePairPoly):
    def __init__(self, A, p, n): 
        LongTimePairPoly.__init__(self, A, p, n)
        # Override noisy_qubits and error_type from LongTimePairM
        self.noisy_qubits = "syndrome"  # Set from LongTimePairM
        self.error_type = 'X'
        self.split_measurements = True

class LongTimePairMExp(LongTimePairM, LongTimePairExp):
    def __init__(self, A, p, n): 
        LongTimePairExp.__init__(self, A, p, n)
        # Override noisy_qubits and error_type from LongTimePairM
        self.noisy_qubits = "syndrome"  # Set from LongTimePairM
        self.error_type = 'X'
        self.split_measurements = True