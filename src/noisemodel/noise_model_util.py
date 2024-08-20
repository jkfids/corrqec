import numpy as np
from typing import Dict, List, Tuple, Union
import stim

def calc_marginals_depolarizing(probabilities):
    p_vec = np.array([1 - probabilities[0], probabilities[0]])
    for p in probabilities[1:]:
        p_vec = np.matmul(np.array([[1 - p, p / 3], [p, 1 - (p / 3)]]), p_vec)
    return p_vec[1]

def calc_marginals(probabilities):
    t_l = np.prod(1 - (2 * np.array(probabilities)))
    return 0.5 * (1 - t_l)

def calc_marginals_mixing(probabilities):
    t = np.prod(1 - np.array(probabilities))
    return 1 - t

def split_circuit(circuit):
    i_tick = None
    i_block = None

    for i, instr in enumerate(circuit):
        if instr.name == 'TICK' and i_tick is None:
            i_tick = i
        elif isinstance(instr, stim.CircuitRepeatBlock) and i_block is None:
            i_block = i

    circuit_init = circuit[:i_tick + 1]
    circuit_init_round = circuit[i_tick + 1:i_block]
    circuit_repeat_block = circuit[i_block].body_copy()
    repeat_count = circuit[i_block].repeat_count
    circuit_final = circuit[i_block + 1:]

    # if split_measurements is True:
    #     circuit_init_round = split_circuit_measurements(circuit_init_round)
    #     circuit_repeat_block = split_circuit_measurements(circuit_repeat_block)

    return (circuit_init, circuit_init_round, circuit_repeat_block, circuit_final), repeat_count


def split_circuit_measurements(circuit):
    for i, instr in enumerate(circuit):
        if instr.name == 'MR' or instr.name == 'M':
            i_split = i
            break
    return circuit[:i_split], circuit[i_split:]

def split_circuit_cx(circuit):
    circuits = []
    i_split_prev = 0
    for i, instr in enumerate(circuit):
        if instr.name == 'CX':
            i_split = i + 1
            circuits.append(circuit[i_split_prev:i_split])
            i_split_prev = i_split
    circuits.append(circuit[i_split_prev:])
            
    return circuits

def split_circuit_cx_m(circuit):
    circuits = []
    i_split_prev = 0
    for i, instr in enumerate(circuit):
        if instr.name == 'CX':
            i_split = i + 1
            circuits.append(circuit[i_split_prev:i_split])
            i_split_prev = i_split
        elif instr.name == 'MR' or instr.name == 'M':
            i_split = i
            circuits.append(circuit[i_split_prev:i_split])
            i_split_prev = i_split
    circuits.append(circuit[i_split_prev:])
    
    return circuits

def get_control_qubits(circuit: stim.Circuit):
    control_qubits = []
    for instr in circuit:
        if instr.name == 'CX':
            control_qubits.append([q.value for q in instr.targets_copy()])
    return control_qubits

def pauli_product_prob(n):
    return 0.75 - 0.75 * np.power((-1 / 3), n)


def calc_euclidean(qubit_coords, rotated=False):
    qubit_coords = {key: np.array(value) for key, value in qubit_coords.items()}
    qubit_list = list(qubit_coords.keys())

    if rotated is False:
        factor = 1
    elif rotated is True:
        factor = 1 / np.sqrt(2)

    distances = {(a, b): factor * np.linalg.norm(qubit_coords[a] - qubit_coords[b], 2)
                 for idx, a in enumerate(qubit_list) for b in qubit_list[idx + 1:]}

    return distances


def nearest_neighbour(qubit_coords, rotated=False):
    distances = calc_euclidean(qubit_coords, rotated=False)
    minr = min(distances.values())
    return {k: 1. for k, v in distances.items() if v == minr}


def calc_manhattan(qubit_coords, rotated=False):
    qubit_coords = {key: np.array(value) for key, value in qubit_coords.items()}
    qubit_list = list(qubit_coords.keys())

    if rotated is False:
        order = 0
    elif rotated is True:
        order = np.inf

    distances = {(a, b): np.linalg.norm(qubit_coords[a] - qubit_coords[b], order)
                 for idx, a in enumerate(qubit_list) for b in qubit_list[idx + 1:]}

    return distances


def poly_decay(r, A, p, n):
    return A * p / (r ** n)


def exp_decay(r, A, p, n):
    return A * p / (n ** r)


def get_partitioned_qubit_coords(circuit: stim.Circuit) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    qubit_coords = circuit.get_final_qubit_coordinates()
    circuit = circuit.flattened()

    for instr in circuit[::-1]:
        if instr.name == "MX" or instr.name == "M":
            data_targets = instr.targets_copy()
            data_qubits = [i.value for i in data_targets]
        elif instr.name == "MR":
            syndrome_targets = instr.targets_copy()
            syndrome_qubits = [i.value for i in syndrome_targets]
        elif instr.name == "H":
            syndromex_targets = instr.targets_copy()
            syndromex_qubits = [i.value for i in syndromex_targets]
            syndromez_qubits = [q for q in syndrome_qubits if q not in syndromex_qubits]
            break

    data = {q: qubit_coords[q] for q in data_qubits}
    syndromez = {q: qubit_coords[q] for q in syndromez_qubits}
    syndromex = {q: qubit_coords[q] for q in syndromex_qubits}

    return data, syndromez, syndromex


def get_detector_coords(circuit: stim.Circuit) -> List[List[Union[int, float]]]:
    coords: List[List[Union[int, float]]] = []
    for instr in circuit[::-1]:
        if isinstance(instr, stim.CircuitRepeatBlock):
            body = instr.body_copy()
            for body_instr in body:
                if body_instr.name == "DETECTOR":
                    coords.append(body_instr.gate_args_copy()[:2])
    return coords


def get_round_pairs(rounds: int):
    round_list = list(range(rounds))
    return np.array([(a, b) for i, b in enumerate(round_list) for a in round_list[:i]])


def get_round_pair_distances(rounds: int):
    size = int(np.ceil(0.5 * rounds))
    round_pair_distances = np.zeros((size, rounds - 1))

    for i in range(rounds - 1):
        np.fill_diagonal(round_pair_distances[:, i:], i + 1)
        np.fill_diagonal(round_pair_distances[i + 1:, :], i + 1)

    return round_pair_distances
