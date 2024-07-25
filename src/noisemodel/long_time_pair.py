import numpy as np
import stim
import pymatching
from .base import NoiseModel
from .noise_model_util import get_round_pairs, get_partitioned_qubit_coords, poly_decay, exp_decay, split_circuit, get_round_pair_distances, calc_marginals

class LongTimePair(NoiseModel):
    def __init__(self, interaction_func=None, noisy_qubits=None, error_type="depolarizing"):
        super().__init__()
        self.interaction_func = interaction_func
        self.noisy_qubits = noisy_qubits
        self.error_type = error_type
        self.split_measurements = False
        
    def sample_logical_error_rate(self, circuit: stim.Circuit, repetitions: int, batch_size: int, max_errors: int=None, return_error_rate: bool=True):
        """
        Samples the logical error rate for a given Stim circuit under correlated noise.

        Args:
            circuit (stim.Circuit): The Stim circuit to sample the logical error rate for.
            repetitions (int): The number of repetitions to perform.
            batch_size (int): The number of parallel samples to generate.
            max_errors (int, optional): The maximum number of errors before stopping sampling. Defaults to None.
            return_error_rate (bool, optional): Whether to return the error rate or the total number of errors. Defaults to True.

        Returns:
            float or tuple: The sampled logical error rate, or a tuple containing the total number of shots and errors.

        """
        total_errors = 0
        sampled_rates = np.zeros(repetitions)

        split_circuits, repeat_count = split_circuit(circuit=circuit, split_measurements=self.split_measurements)

        qubit_coords = self.get_noisy_qubit_coords(circuit)
        targets = list(qubit_coords.keys())

        n_qubits = circuit.num_qubits  # Includes unused qubits
        rounds = repeat_count + 1

        detector_error_model = self.convert_circuit_marginalised(circuit).detector_error_model()
        matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

        for i in range(repetitions):
            detection_events, observable_flips = self._sample_circuit(split_circuits=split_circuits,
                                                                     targets=targets,
                                                                     n_qubits=n_qubits,
                                                                     rounds=rounds,
                                                                     batch_size=batch_size)

            predictions = matcher.decode_batch(detection_events).flatten()

            n_errors = np.sum(predictions != observable_flips)
            total_errors += n_errors
            sampled_rates[i] = n_errors / batch_size

            if max_errors is not None:
                if total_errors >= max_errors:
                    break

        if return_error_rate is True:
            output = np.mean(sampled_rates[:i + 1])
        elif return_error_rate is False:
            output = (int(batch_size * (i + 1)), int(total_errors))

        return output
        
    def sample_circuit(self, circuit: stim.Circuit, shots: int):
        """
        Samples a given Stim circuit under correlated noise.

        Args:
            circuit (stim.Circuit): The Stim circuit to sample correlated noise for.
            shots (int): The number of full circuit simulations to run.

        Returns:
            tuple: A tuple containing the detection events and observable flips.
                   - detection_events (ndarray): A 2D numpy array representing the detection events.
                   - observable_flips (ndarray): A 1D numpy array representing the observable flips.
        """
        split_circuits, repeat_count = split_circuit(circuit=circuit, split_measurements=self.split_measurements)
        qubit_coords = self.get_noisy_qubit_coords(circuit)
        targets = list(qubit_coords.keys())
        n_qubits = circuit.num_qubits  # Includes unused qubits in the Stim circuit
        rounds = repeat_count + 1
        return self._sample_circuit(split_circuits=split_circuits,
                                    targets=targets,
                                    n_qubits=n_qubits,
                                    rounds=rounds,
                                    batch_size=shots)
        
    def _sample_circuit(self, split_circuits: tuple, targets: list, n_qubits: int, rounds: int, batch_size: int):
        """
        Backend function to sample correlated noise for a stim circuit using stim.FlipSimulator.

        Args:
            split_circuits (tuple): A tuple containing the split circuit (circuit_init, circuit_init_round,
                                   circuit_repeat_block, circuit_final).
            targets (list): List of target qubits.
            n_qubits (int): Number of qubits in the Stim circuit, which may not equal len(targets).
            rounds (int): Number of error correction rounds.
            batch_size (int): Number of parallel samples to generate.

        Returns:
            tuple: A tuple containing the detection events and observable flips.
                   - detection_events (ndarray): A 2D numpy array representing the detection events.
                   - observable_flips (ndarray): A 1D numpy array representing the observable flips.
        """
        circuit_init, circuit_init_round, circuit_repeat_block, circuit_final = split_circuits
        sim = stim.FlipSimulator(batch_size=batch_size)
        
        if self.error_type == "depolarizing":
            X_errors, Y_errors, Z_errors = self.sample_errors(targets=targets, n_qubits=n_qubits, rounds=rounds, batch_size=batch_size)
            sim.do(circuit_init)
            sim.broadcast_pauli_errors(pauli='X', mask=X_errors[0])
            sim.broadcast_pauli_errors(pauli='Y', mask=Y_errors[0])
            sim.broadcast_pauli_errors(pauli='Z', mask=Z_errors[0])
            
            sim.do(circuit_init_round)
            for j in range(1, rounds):
                sim.broadcast_pauli_errors(pauli='X', mask=X_errors[j])
                sim.broadcast_pauli_errors(pauli='Y', mask=Y_errors[j])
                sim.broadcast_pauli_errors(pauli='Z', mask=Z_errors[j])
                sim.do(circuit_repeat_block)
                
        else:
            errors = self.sample_errors(targets=targets, n_qubits=n_qubits, rounds=rounds, batch_size=batch_size)
            sim.do(circuit_init)
            sim.broadcast_pauli_errors(pauli=self.error_type, mask=errors[0])
            sim.do(circuit_init_round)
            for j in range(1, rounds):
                sim.broadcast_pauli_errors(pauli=self.error_type, mask=errors[j])
                sim.do(circuit_repeat_block)
            
        sim.do(circuit_final)
        
        detection_events = sim.get_detector_flips().transpose()
        observable_flips = sim.get_observable_flips().flatten()
        
        return detection_events, observable_flips

    def sample_errors(self, targets: list, n_qubits: int, rounds: int, batch_size: int) -> tuple:
        """
        Samples temporally correlated pairwise errors for a given number of rounds over a set of target qubits. Sliced 
        and used as input in FlipSimulator.broadcast_pauli_errors.

        Args:
            targets (list): List of target qubits.
            n_qubits (int): Number of qubits in the Stim circuit, which may not equal len(targets).
            rounds (int): Number of error correction rounds.
            batch_size (int): Number of parallel samples to generate.

        Returns:
            tuple: A tuple of three boolean arrays representing sampled X, Y, and Z errors respectively. Has shape (rounds, n_qubits, batch_size).
        """
        n_targets = len(targets)
        rng = np.random.default_rng()

        if self.error_type == "depolarizing":
            c = 1.06666666667
        else:
            c = 1

        pair_probabilities = c * self.calc_pair_probabilities(rounds=rounds)
        pair_probabilities_matrix = np.tile(pair_probabilities, (n_targets * batch_size, 1))

        rnd = rng.random(pair_probabilities_matrix.shape)
        pair_errors = rnd < pair_probabilities_matrix

        map = self.gen_pair_to_qubit_map(rounds)
        qubit_errors = np.matmul(pair_errors, map).reshape(batch_size, n_targets, rounds).transpose(2, 1, 0)

        if self.error_type == "depolarizing":
            qubit_errors = qubit_errors.astype(bool)

            random_paulis = rng.integers(0, 3, size=qubit_errors.shape, endpoint=True)
            pauli_errors = np.zeros([rounds, n_qubits, batch_size], dtype=int)
            pauli_errors[:, targets, :] = random_paulis * qubit_errors

            X_errors = pauli_errors == 1
            Y_errors = pauli_errors == 2
            Z_errors = pauli_errors == 3

            errors = (X_errors, Y_errors, Z_errors)
        else:
            qubit_errors = self.int_to_bool_errors(qubit_errors)
            errors = np.zeros([rounds, n_qubits, batch_size], dtype=bool)
            errors[:, targets, :] = qubit_errors

        return errors

    def calc_pair_probabilities(self, rounds: int) -> np.ndarray:
        """
        Calculates the probabilities for each pair of rounds based on the interaction function.

        Args:
            rounds (int): The number of rounds to calculate pair probabilities for.

        Returns:
            np.ndarray: A 2D numpy array representing the pair probabilities.
        """
        round_list = list(range(rounds))
        round_pair_distances = np.array([b - a for i, b in enumerate(round_list) for a in round_list[:i]])
        pair_probabilities = self.interaction_func(round_pair_distances)
        return pair_probabilities

    def gen_pair_to_qubit_map(self, rounds: int) -> np.ndarray:
        """
        Generates a pair-to-qubit map for a given number of rounds to be used in pairwise correlated error sampling.
        Args:
            rounds (int): The number of rounds.

        Returns:
            np.ndarray: A 2D numpy array representing the pair-to-qubit map. Each row corresponds to a pair, and each column corresponds to a round.
        """
        pairs = get_round_pairs(rounds=rounds)
        pair_to_qubit_map = np.zeros((len(pairs), rounds), dtype=int)

        for i, (a, b) in enumerate(pairs):
            pair_to_qubit_map[i][a] = 1
            pair_to_qubit_map[i][b] = 1

        return pair_to_qubit_map
    
    def int_to_bool_errors(self, errors: np.ndarray) -> np.ndarray:
        """
        Converts integer errors to boolean errors by taking the modulo 2.

        Args:
            errors (np.ndarray): The integer errors to be converted.

        Returns:
            np.ndarray: The boolean errors.
        """
        return np.mod(errors, 2)
    
    def gen_circuit_marginalised(self, distance: int, rounds: int, scl_noise: list=[], rotated: bool=True) -> stim.Circuit:
        """
        Generate a Stim circuit with marginalised noise.

        Args:
            distance (int): The distance of the surface code.
            rounds (int): The number of rounds in the circuit.
            scl_noise (list, optional): A list of Stim noise to include in the circuit. Defaults to [].
            rotated (bool, optional): Whether to use rotated or unrotated memory Z surface code. Defaults to True.

        Returns:
            stim.Circuit: The generated circuit with marginalised noise.
        """
        if rotated is True:
            code = "surface_code:rotated_memory_z"
        elif rotated is False:
            code = "surface_code:unrotated_memory_z"
            
        scl_kwargs = {k: self.p for k in scl_noise}
        
        circuit = stim.Circuit.generated(
            code,
            rounds=rounds,
            distance=distance,
            **scl_kwargs
        )
        
        output_circuit = self.convert_circuit_marginalised(circuit=circuit)
        
        return output_circuit
    
    def convert_circuit_marginalised(self, circuit: stim.Circuit) -> stim.Circuit:
        """
        Converts a given circuit into a noisy circuit by applying errors given by the marginal error probabilities.

        Args:
            circuit (stim.Circuit): The input circuit to be converted.

        Returns:
            stim.Circuit: The converted circuit with marginal errors inserted.
        """
        output_circuit = stim.Circuit()
        split_circuits, repeat_count = split_circuit(circuit=circuit)
        circuit_init, circuit_init_round, circuit_repeat_block, circuit_final = split_circuits
        qubit_coords = self.get_noisy_qubit_coords(circuit)
        qubits = qubit_coords.keys()
        rounds = repeat_count + 1
        marginals = self.calc_marginals_per_round(rounds=rounds)
        output_circuit += circuit_init
        if self.error_type == "depolarizing":
            output_circuit.append('DEPOLARIZE1', qubits, marginals[0])
            output_circuit += circuit_init_round
            for i in range(1, rounds):
                output_circuit.append('DEPOLARIZE1', qubits, marginals[i])
                output_circuit += circuit_repeat_block
        else:
            output_circuit.append(self.error_type + '_ERROR', qubits, marginals[0])
            output_circuit += circuit_init_round
            for i in range(1, rounds):
                output_circuit.append(self.error_type + '_ERROR', qubits, marginals[i])
                output_circuit += circuit_repeat_block
        output_circuit += circuit_final
        return output_circuit

    def calc_marginals_per_round(self, rounds: int) -> np.ndarray:
        """
        Calculate the marginal error probabilities for each round under the pairwise model.

        Args:
            rounds (int): The number of rounds in the circuit.

        Returns:
            np.ndarray: Array of marginal error probabilities for each round.
        """
        marginals = np.zeros(rounds)
        round_pair_distances = get_round_pair_distances(rounds=rounds)
        round_pair_probabilities = self.interaction_func(round_pair_distances)
        if self.error_type == "depolarizing":
            round_pair_probabilities *= 1.06666666667
        for i in range(round_pair_probabilities.shape[0]):
            p = calc_marginals(round_pair_probabilities[i, :])
            marginals[i] = p
            marginals[-i-1] = p
        if self.error_type == "depolarizing":
            marginals *= .75
        return marginals

    def get_noisy_qubit_coords(self, circuit: stim.Circuit) -> dict:
        """
        Get the qubit coordinates of a circuit based on the `noisy_qubits` parameter.

        Args:
            circuit (stim.Circuit): The Stim input circuit.

        Returns:
            dict: A dictionary containing the qubit coordinates partitioned by type.
        """
        # Obtain qubit coordinates paritioned by data, syndrome Z, and syndrome X
        data, sz, sx = get_partitioned_qubit_coords(circuit) 
        
        if self.noisy_qubits == "all" or self.noisy_qubits is None:
            # Include all qubit coordinates
            qubit_coords = {**data, **sz, **sx}
        elif self.noisy_qubits == "data":
            # Include only data qubit coordinates
            qubit_coords = data
        elif self.noisy_qubits == "syndrome":
            # Include only syndrome qubit coordinates
            qubit_coords = {**sz, **sx}

        return qubit_coords

class LongTimePairPoly(LongTimePair):
    """
    Long-time pairwise correlated errors that decay polynomially with separation in rounds.
    """
    def __init__(self, A, p, n, noisy_qubits=None, error_type="depolarizing"):
        super().__init__(interaction_func=lambda r: poly_decay(r, self.A, self.p, self.n),
                         noisy_qubits=noisy_qubits, error_type=error_type)
        self.A = A
        self.p = p
        self.n = n
        
class LongTimePairExp(LongTimePair):
    """
    Long-time pairwise correlated errors that decay exponentially with separation in rounds.
    """
    def __init__(self, A, p, n, noisy_qubits=None, error_type="depolarizing"): 
        super().__init__(interaction_func=lambda r: exp_decay(r, self.A, self.p, self.n),
                         noisy_qubits=noisy_qubits, error_type=error_type)
        self.A = A
        self.p = p
        self.n = n