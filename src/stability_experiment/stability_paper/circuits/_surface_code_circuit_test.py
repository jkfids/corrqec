import pytest
import stim

from stability_paper.circuits import \
    surface_code_stability_experiment_circuit, surface_code_memory_experiment_circuit
from stability_paper.tools import NoiseModel


def test_surface_code_stability_experiment_circuit():
    circuit = surface_code_stability_experiment_circuit(
        diam=3,
        rounds=5,
        basis='X',
    )
    noisy = NoiseModel.depolarizing_cz_noise(1e-3).noisy_circuit(circuit)
    assert len(noisy.shortest_graphlike_error()) == 5
    assert noisy == stim.Circuit("""
        QUBIT_COORDS(0, 1) 0
        QUBIT_COORDS(0, 2) 1
        QUBIT_COORDS(1, 0) 2
        QUBIT_COORDS(1, 1) 3
        QUBIT_COORDS(1, 2) 4
        QUBIT_COORDS(2, 0) 5
        QUBIT_COORDS(2, 1) 6
        QUBIT_COORDS(-0.5, 1.5) 7
        QUBIT_COORDS(0.5, 0.5) 8
        QUBIT_COORDS(0.5, 1.5) 9
        QUBIT_COORDS(0.5, 2.5) 10
        QUBIT_COORDS(1.5, -0.5) 11
        QUBIT_COORDS(1.5, 0.5) 12
        QUBIT_COORDS(1.5, 1.5) 13
        QUBIT_COORDS(2.5, 0.5) 14
        R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
        X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
        TICK
        H 1 3 5 7 8 9 10 11 12 13 14
        DEPOLARIZE1(0.001) 1 3 5 7 8 9 10 11 12 13 14 0 2 4 6
        TICK
        CZ 0 9 1 10 2 12 3 13 5 14
        DEPOLARIZE2(0.001) 0 9 1 10 2 12 3 13 5 14
        DEPOLARIZE1(0.001) 4 6 7 8 11
        TICK
        H 0 1 2 3 4 5 6
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
        TICK
        CZ 0 8 3 9 2 11 5 12 4 13 6 14
        DEPOLARIZE2(0.001) 0 8 3 9 2 11 5 12 4 13 6 14
        DEPOLARIZE1(0.001) 1 7 10
        TICK
        CZ 0 7 2 8 1 9 4 10 3 12 6 13
        DEPOLARIZE2(0.001) 0 7 2 8 1 9 4 10 3 12 6 13
        DEPOLARIZE1(0.001) 5 11 14
        TICK
        H 0 1 2 3 4 5 6
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
        TICK
        CZ 1 7 3 8 4 9 5 11 6 12
        DEPOLARIZE2(0.001) 1 7 3 8 4 9 5 11 6 12
        DEPOLARIZE1(0.001) 0 2 10 13 14
        TICK
        H 1 3 5 7 8 9 10 11 12 13 14
        DEPOLARIZE1(0.001) 1 3 5 7 8 9 10 11 12 13 14 0 2 4 6
        TICK
        M(0.001) 7 8 9 10 11 12 13 14
        DETECTOR(0.5, 1.5, 0) rec[-6]
        DETECTOR(1.5, 0.5, 0) rec[-3]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.001) 7 8 9 10 11 12 13 14 0 1 2 3 4 5 6
        TICK
        REPEAT 3 {
            R 7 8 9 10 11 12 13 14
            X_ERROR(0.001) 7 8 9 10 11 12 13 14
            DEPOLARIZE1(0.001) 0 1 2 3 4 5 6
            TICK
            H 1 3 5 7 8 9 10 11 12 13 14
            DEPOLARIZE1(0.001) 1 3 5 7 8 9 10 11 12 13 14 0 2 4 6
            TICK
            CZ 0 9 1 10 2 12 3 13 5 14
            DEPOLARIZE2(0.001) 0 9 1 10 2 12 3 13 5 14
            DEPOLARIZE1(0.001) 4 6 7 8 11
            TICK
            H 0 1 2 3 4 5 6
            DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
            TICK
            CZ 0 8 3 9 2 11 5 12 4 13 6 14
            DEPOLARIZE2(0.001) 0 8 3 9 2 11 5 12 4 13 6 14
            DEPOLARIZE1(0.001) 1 7 10
            TICK
            CZ 0 7 2 8 1 9 4 10 3 12 6 13
            DEPOLARIZE2(0.001) 0 7 2 8 1 9 4 10 3 12 6 13
            DEPOLARIZE1(0.001) 5 11 14
            TICK
            H 0 1 2 3 4 5 6
            DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
            TICK
            CZ 1 7 3 8 4 9 5 11 6 12
            DEPOLARIZE2(0.001) 1 7 3 8 4 9 5 11 6 12
            DEPOLARIZE1(0.001) 0 2 10 13 14
            TICK
            H 1 3 5 7 8 9 10 11 12 13 14
            DEPOLARIZE1(0.001) 1 3 5 7 8 9 10 11 12 13 14 0 2 4 6
            TICK
            M(0.001) 7 8 9 10 11 12 13 14
            DETECTOR(-0.5, 1.5, 0) rec[-16] rec[-8]
            DETECTOR(0.5, 0.5, 0) rec[-15] rec[-7]
            DETECTOR(0.5, 1.5, 0) rec[-14] rec[-6]
            DETECTOR(0.5, 2.5, 0) rec[-13] rec[-5]
            DETECTOR(1.5, -0.5, 0) rec[-12] rec[-4]
            DETECTOR(1.5, 0.5, 0) rec[-11] rec[-3]
            DETECTOR(1.5, 1.5, 0) rec[-10] rec[-2]
            DETECTOR(2.5, 0.5, 0) rec[-9] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE1(0.001) 7 8 9 10 11 12 13 14 0 1 2 3 4 5 6
            TICK
        }
        R 7 8 9 10 11 12 13 14
        X_ERROR(0.001) 7 8 9 10 11 12 13 14
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6
        TICK
        H 1 3 5 7 8 9 10 11 12 13 14
        DEPOLARIZE1(0.001) 1 3 5 7 8 9 10 11 12 13 14 0 2 4 6
        TICK
        CZ 0 9 1 10 2 12 3 13 5 14
        DEPOLARIZE2(0.001) 0 9 1 10 2 12 3 13 5 14
        DEPOLARIZE1(0.001) 4 6 7 8 11
        TICK
        H 0 1 2 3 4 5 6
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
        TICK
        CZ 0 8 3 9 2 11 5 12 4 13 6 14
        DEPOLARIZE2(0.001) 0 8 3 9 2 11 5 12 4 13 6 14
        DEPOLARIZE1(0.001) 1 7 10
        TICK
        CZ 0 7 2 8 1 9 4 10 3 12 6 13
        DEPOLARIZE2(0.001) 0 7 2 8 1 9 4 10 3 12 6 13
        DEPOLARIZE1(0.001) 5 11 14
        TICK
        H 0 1 2 3 4 5 6
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
        TICK
        CZ 1 7 3 8 4 9 5 11 6 12
        DEPOLARIZE2(0.001) 1 7 3 8 4 9 5 11 6 12
        DEPOLARIZE1(0.001) 0 2 10 13 14
        TICK
        H 1 3 5 7 8 9 10 11 12 13 14
        DEPOLARIZE1(0.001) 1 3 5 7 8 9 10 11 12 13 14 0 2 4 6
        TICK
        M(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
        DETECTOR(-0.5, 1.5, 0) rec[-23] rec[-8]
        DETECTOR(0.5, 0.5, 0) rec[-22] rec[-7]
        DETECTOR(0.5, 1.5, 0) rec[-21] rec[-6]
        DETECTOR(0.5, 2.5, 0) rec[-20] rec[-5]
        DETECTOR(1.5, -0.5, 0) rec[-19] rec[-4]
        DETECTOR(1.5, 0.5, 0) rec[-18] rec[-3]
        DETECTOR(1.5, 1.5, 0) rec[-17] rec[-2]
        DETECTOR(2.5, 0.5, 0) rec[-16] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DETECTOR(0.5, 1.5, 0) rec[-15] rec[-14] rec[-12] rec[-11] rec[-6]
        DETECTOR(1.5, 0.5, 0) rec[-13] rec[-12] rec[-10] rec[-9] rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-8] rec[-7] rec[-5] rec[-4] rec[-2] rec[-1]
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    """)


def test_surface_code_memory_experiment_circuit():
    circuit = surface_code_memory_experiment_circuit(
        diam=3,
        rounds=5,
        basis='X',
    )
    noisy = NoiseModel.depolarizing_cz_noise(1e-3).noisy_circuit(circuit)
    assert len(noisy.shortest_graphlike_error()) == 3

    assert noisy == stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 1) 1
        QUBIT_COORDS(0, 2) 2
        QUBIT_COORDS(1, 0) 3
        QUBIT_COORDS(1, 1) 4
        QUBIT_COORDS(1, 2) 5
        QUBIT_COORDS(2, 0) 6
        QUBIT_COORDS(2, 1) 7
        QUBIT_COORDS(2, 2) 8
        QUBIT_COORDS(-0.5, 1.5) 9
        QUBIT_COORDS(0.5, -0.5) 10
        QUBIT_COORDS(0.5, 0.5) 11
        QUBIT_COORDS(0.5, 1.5) 12
        QUBIT_COORDS(1.5, 0.5) 13
        QUBIT_COORDS(1.5, 1.5) 14
        QUBIT_COORDS(1.5, 2.5) 15
        QUBIT_COORDS(2.5, 0.5) 16
        R 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        TICK
        H 1 2 3 5 7 8 9 10 11 12 13 14 15 16
        DEPOLARIZE1(0.001) 1 2 3 5 7 8 9 10 11 12 13 14 15 16 0 4 6
        TICK
        CZ 0 11 1 12 3 13 4 14 5 15 6 16
        DEPOLARIZE2(0.001) 0 11 1 12 3 13 4 14 5 15 6 16
        DEPOLARIZE1(0.001) 2 7 8 9 10
        TICK
        H 0 1 4 5 6 7
        DEPOLARIZE1(0.001) 0 1 4 5 6 7 2 3 8 9 10 11 12 13 14 15 16
        TICK
        CZ 1 11 4 12 6 13 5 14 8 15 7 16
        DEPOLARIZE2(0.001) 1 11 4 12 6 13 5 14 8 15 7 16
        DEPOLARIZE1(0.001) 0 2 3 9 10
        TICK
        H 3 5
        DEPOLARIZE1(0.001) 3 5 0 1 2 4 6 7 8 9 10 11 12 13 14 15 16
        TICK
        CZ 1 9 0 10 3 11 2 12 4 13 7 14
        DEPOLARIZE2(0.001) 1 9 0 10 3 11 2 12 4 13 7 14
        DEPOLARIZE1(0.001) 5 6 8 15 16
        TICK
        H 1 2 3 4 7 8
        DEPOLARIZE1(0.001) 1 2 3 4 7 8 0 5 6 9 10 11 12 13 14 15 16
        TICK
        CZ 2 9 3 10 4 11 5 12 7 13 8 14
        DEPOLARIZE2(0.001) 2 9 3 10 4 11 5 12 7 13 8 14
        DEPOLARIZE1(0.001) 0 1 6 15 16
        TICK
        H 2 4 8 9 10 11 12 13 14 15 16
        DEPOLARIZE1(0.001) 2 4 8 9 10 11 12 13 14 15 16 0 1 3 5 6 7
        TICK
        M(0.001) 9 10 11 12 13 14 15 16
        DETECTOR(-0.5, 1.5, 0) rec[-8]
        DETECTOR(0.5, 0.5, 0) rec[-6]
        DETECTOR(1.5, 1.5, 0) rec[-3]
        DETECTOR(2.5, 0.5, 0) rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.001) 9 10 11 12 13 14 15 16 0 1 2 3 4 5 6 7 8
        TICK
        REPEAT 3 {
            R 9 10 11 12 13 14 15 16
            X_ERROR(0.001) 9 10 11 12 13 14 15 16
            DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8
            TICK
            H 0 4 6 9 10 11 12 13 14 15 16
            DEPOLARIZE1(0.001) 0 4 6 9 10 11 12 13 14 15 16 1 2 3 5 7 8
            TICK
            CZ 0 11 1 12 3 13 4 14 5 15 6 16
            DEPOLARIZE2(0.001) 0 11 1 12 3 13 4 14 5 15 6 16
            DEPOLARIZE1(0.001) 2 7 8 9 10
            TICK
            H 0 1 4 5 6 7
            DEPOLARIZE1(0.001) 0 1 4 5 6 7 2 3 8 9 10 11 12 13 14 15 16
            TICK
            CZ 1 11 4 12 6 13 5 14 8 15 7 16
            DEPOLARIZE2(0.001) 1 11 4 12 6 13 5 14 8 15 7 16
            DEPOLARIZE1(0.001) 0 2 3 9 10
            TICK
            H 3 5
            DEPOLARIZE1(0.001) 3 5 0 1 2 4 6 7 8 9 10 11 12 13 14 15 16
            TICK
            CZ 1 9 0 10 3 11 2 12 4 13 7 14
            DEPOLARIZE2(0.001) 1 9 0 10 3 11 2 12 4 13 7 14
            DEPOLARIZE1(0.001) 5 6 8 15 16
            TICK
            H 1 2 3 4 7 8
            DEPOLARIZE1(0.001) 1 2 3 4 7 8 0 5 6 9 10 11 12 13 14 15 16
            TICK
            CZ 2 9 3 10 4 11 5 12 7 13 8 14
            DEPOLARIZE2(0.001) 2 9 3 10 4 11 5 12 7 13 8 14
            DEPOLARIZE1(0.001) 0 1 6 15 16
            TICK
            H 2 4 8 9 10 11 12 13 14 15 16
            DEPOLARIZE1(0.001) 2 4 8 9 10 11 12 13 14 15 16 0 1 3 5 6 7
            TICK
            M(0.001) 9 10 11 12 13 14 15 16
            DETECTOR(-0.5, 1.5, 0) rec[-16] rec[-8]
            DETECTOR(0.5, -0.5, 0) rec[-15] rec[-7]
            DETECTOR(0.5, 0.5, 0) rec[-14] rec[-6]
            DETECTOR(0.5, 1.5, 0) rec[-13] rec[-5]
            DETECTOR(1.5, 0.5, 0) rec[-12] rec[-4]
            DETECTOR(1.5, 1.5, 0) rec[-11] rec[-3]
            DETECTOR(1.5, 2.5, 0) rec[-10] rec[-2]
            DETECTOR(2.5, 0.5, 0) rec[-9] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE1(0.001) 9 10 11 12 13 14 15 16 0 1 2 3 4 5 6 7 8
            TICK
        }
        R 9 10 11 12 13 14 15 16
        X_ERROR(0.001) 9 10 11 12 13 14 15 16
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8
        TICK
        H 0 4 6 9 10 11 12 13 14 15 16
        DEPOLARIZE1(0.001) 0 4 6 9 10 11 12 13 14 15 16 1 2 3 5 7 8
        TICK
        CZ 0 11 1 12 3 13 4 14 5 15 6 16
        DEPOLARIZE2(0.001) 0 11 1 12 3 13 4 14 5 15 6 16
        DEPOLARIZE1(0.001) 2 7 8 9 10
        TICK
        H 0 1 4 5 6 7
        DEPOLARIZE1(0.001) 0 1 4 5 6 7 2 3 8 9 10 11 12 13 14 15 16
        TICK
        CZ 1 11 4 12 6 13 5 14 8 15 7 16
        DEPOLARIZE2(0.001) 1 11 4 12 6 13 5 14 8 15 7 16
        DEPOLARIZE1(0.001) 0 2 3 9 10
        TICK
        H 3 5
        DEPOLARIZE1(0.001) 3 5 0 1 2 4 6 7 8 9 10 11 12 13 14 15 16
        TICK
        CZ 1 9 0 10 3 11 2 12 4 13 7 14
        DEPOLARIZE2(0.001) 1 9 0 10 3 11 2 12 4 13 7 14
        DEPOLARIZE1(0.001) 5 6 8 15 16
        TICK
        H 1 2 3 4 7 8
        DEPOLARIZE1(0.001) 1 2 3 4 7 8 0 5 6 9 10 11 12 13 14 15 16
        TICK
        CZ 2 9 3 10 4 11 5 12 7 13 8 14
        DEPOLARIZE2(0.001) 2 9 3 10 4 11 5 12 7 13 8 14
        DEPOLARIZE1(0.001) 0 1 6 15 16
        TICK
        H 0 1 3 5 6 7 9 10 11 12 13 14 15 16
        DEPOLARIZE1(0.001) 0 1 3 5 6 7 9 10 11 12 13 14 15 16 2 4 8
        TICK
        M(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        DETECTOR(-0.5, 1.5, 0) rec[-25] rec[-8]
        DETECTOR(0.5, -0.5, 0) rec[-24] rec[-7]
        DETECTOR(0.5, 0.5, 0) rec[-23] rec[-6]
        DETECTOR(0.5, 1.5, 0) rec[-22] rec[-5]
        DETECTOR(1.5, 0.5, 0) rec[-21] rec[-4]
        DETECTOR(1.5, 1.5, 0) rec[-20] rec[-3]
        DETECTOR(1.5, 2.5, 0) rec[-19] rec[-2]
        DETECTOR(2.5, 0.5, 0) rec[-18] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        DETECTOR(-0.5, 1.5, 0) rec[-16] rec[-15] rec[-8]
        DETECTOR(0.5, 0.5, 0) rec[-17] rec[-16] rec[-14] rec[-13] rec[-6]
        DETECTOR(1.5, 1.5, 0) rec[-13] rec[-12] rec[-10] rec[-9] rec[-3]
        DETECTOR(2.5, 0.5, 0) rec[-11] rec[-10] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-17] rec[-14] rec[-11]
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    """)


@pytest.mark.parametrize("diam,rounds,basis", [
    (d, r, b)
    for d in [3, 4, 5]
    for r in [2, 3, 4, 5]
    for b in 'XZ'
])
def test_circuit_distances(diam: int, rounds: int, basis: str):
    stability_circuit = surface_code_stability_experiment_circuit(
        diam=diam,
        rounds=rounds,
        basis=basis,
    )
    memory_circuit = surface_code_memory_experiment_circuit(
        diam=diam,
        rounds=rounds,
        basis=basis,
    )
    noise = NoiseModel.depolarizing_cz_noise(1e-3)
    assert len(noise.noisy_circuit(stability_circuit).shortest_graphlike_error()) == rounds
    assert len(noise.noisy_circuit(memory_circuit).shortest_graphlike_error()) == diam
