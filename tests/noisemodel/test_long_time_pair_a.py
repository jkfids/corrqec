import numpy as np
import stim

from src.noisemodel import LongTimePairAPoly, LongTimeStreakAPoly
from src.noisemodel import split_circuit_cx_m, split_circuit

A = 1
p = .01
n = 2

d = 3
r = 6
batch_size = 100000

scl_noise = []
scl_kwargs = {k: p for k in scl_noise}

circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=r, distance=d, **scl_kwargs)

NoiseModel = LongTimePairAPoly
model = NoiseModel(A=A, p=p, n=n)

split_circuits, repeat_count = split_circuit(circuit=circuit)
rounds = r
targets = model.get_targets(circuit=circuit)
d_targets, m_targets, c_targets = targets

d_errors, m_errors, c_errors = model.sample_errors(targets=targets, n_qubits=circuit.num_qubits, rounds=rounds, batch_size=batch_size)
d_marginals, m_marginals, c_marginals = model.calc_marginals_per_round(rounds=r)

# errors = m_errors
# marginals = m_marginals
# targets = m_targets

X_errors, Y_errors, Z_errors = c_errors[0]
errors = X_errors + Y_errors + Z_errors
marginals = c_marginals * (12/15)
targets = c_targets[0]

mean_physical_error = np.zeros(rounds)
for i in range(rounds):
    mean_physical_error[i] += np.sum(np.mean(errors[i], axis=1))/len(targets)
    
div = []
for i in range(len(mean_physical_error)):
    div.append((mean_physical_error[i]/marginals[i]))
    
print(mean_physical_error)
print(d_marginals)
print(np.mean(div))

# output = model.sample_circuit(circuit, shots=10000)
output = model.sample_logical_error_rate(circuit=circuit,
                                         repetitions=1,
                                         batch_size=batch_size,
                                         max_errors=None)
print(output)