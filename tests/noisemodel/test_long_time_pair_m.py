import sys

import numpy as np

import stim

from src import LongTimePairMPoly, split_circuit

p = .001
r = 3
d = 3

model = LongTimePairMPoly(A=1, p=p, n=2)

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    rounds=r,
    distance=d,
    )

split_circuits, repeat_count = split_circuit(circuit=circuit, split_measurements=True)
qubit_coords = model.get_noisy_qubit_coords(circuit)
targets = list(qubit_coords.keys())
n_qubits = circuit.num_qubits  # Includes unused qubits in the Stim circuit
rounds = repeat_count + 1

errors = model.sample_errors(targets=targets, n_qubits=n_qubits, rounds=rounds, batch_size=1000000)

mean_physical_error = np.zeros(rounds)

for i in range(rounds):
    mean_physical_error[i] += np.sum(np.mean(errors[i], axis=1))/len(targets)

print(mean_physical_error)

# noisy_circuit = model.gen_circuit_marginalised(distance=d, rounds=r, rotated=True)
# diagram = noisy_circuit.diagram('timeline-svg')
# with open('diagram.svg', 'w') as f:
#     print(diagram, file=f)