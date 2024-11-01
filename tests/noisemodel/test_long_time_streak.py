import numpy as np
import stim

from src.noisemodel import LongTimeStreakPoly, LongTimePairPoly, LongTimeStreakExp, LongTimePairMPoly
from src.noisemodel import split_circuit

p = .1
d = 3
r = 6


# error_type = 'depolarizing'
error_type = 'X'

NoiseModel = LongTimePairPoly
model = NoiseModel(A=1, p=p, n=2, error_type=error_type, noisy_qubits="data")

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    rounds=r,
    distance=d,
    )

split_circuits, repeat_count = split_circuit(circuit=circuit)
qubit_coords = model.get_noisy_qubit_coords(circuit)
targets = list(qubit_coords.keys())
n_qubits = circuit.num_qubits  # Includes unused qubits in the Stim circuit
rounds = repeat_count + 1

if error_type == 'depolarizing':
    X_errors, Y_errors, Z_errors = model.sample_errors(targets=targets, n_qubits=n_qubits, rounds=rounds, batch_size=1000000)
    errors = X_errors + Y_errors + Z_errors
else:
    errors = model.sample_errors(targets=targets, n_qubits=n_qubits, rounds=rounds, batch_size=1000000)

mean_physical_error = np.zeros(rounds)

for i in range(rounds):
    mean_physical_error[i] += np.sum(np.mean(errors[i], axis=1))/len(targets)

print(mean_physical_error)

marginals = model.calc_marginals_per_round(rounds=r)

print(marginals)

div = []
for i in range(len(mean_physical_error)):
    div.append((mean_physical_error[i]/marginals[i]))
    
print(np.mean(div))
    

# noisy_circuit = model.convert_circuit_marginalised(circuit=circuit)
# diagram = noisy_circuit.diagram('timeline-svg')
# with open('diagram.svg', 'w') as f:
#     print(diagram, file=f)