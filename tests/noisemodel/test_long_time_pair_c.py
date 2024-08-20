import numpy as np
import stim

from src.noisemodel import LongTimePairCPoly
from src.noisemodel import split_circuit, split_circuit_cx

p = .1
d = 3
r = 9
batch_size = 1000000

model = LongTimePairCPoly(A=1, p=p, n=2)

scl_noise = []
scl_kwargs = {k: p for k in scl_noise}

circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=r, distance=d, **scl_kwargs)

marginals = model.calc_marginals_per_round(rounds=r) * (12/15)

targets = model.get_targets(circuit)
X_errors, Y_errors, Z_errors = model.sample_control_errors(targets=targets, n_qubits=circuit.num_qubits, rounds=r, batch_size=batch_size)[0]
errors = X_errors + Y_errors + Z_errors

mean_physical_error = np.zeros(r)
for i in range(r):
    mean_physical_error[i] += np.sum(np.mean(errors[i], axis=1))/len(targets[0])
    
print(marginals)
print(mean_physical_error)

div = []
for i in range(len(marginals)):
    div.append(mean_physical_error[i]/marginals[i])
print(np.mean(div))



# margin_circuit = model.convert_circuit_marginalised(circuit=circuit)
# margin_circuit = model.gen_circuit_marginalised(distance=d, rounds=r, scl_noise=["after_clifford_depolarization"])
# print(margin_circuit)
