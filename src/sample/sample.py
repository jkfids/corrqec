# Standard library imports
import multiprocessing

# Third-party library imports
import pymatching
import numpy as np
import stim

# Local imports


def sample_threshold(num_workers, NoiseModel, distances, probabilities, scl_noise=[], max_shots=100000, max_errors=None, batch_size=1000, **kwargs):
    physical = {d: [] for d in distances}
    logical = {d: [] for d in distances}
    
    args_list = [(d, p, NoiseModel, scl_noise, max_shots, max_errors, batch_size, kwargs) for d in distances for p in probabilities]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(task_sample, args_list)
    
    for result in results:
        d, r, p, shots, errors = result['d'], result['r'], result['p'], result['shots'], result['errors']
        if errors == 0:
            print(f"No errors for d = {d}, p = {p}")
            continue
        physical[d].append(p)
        logical[d].append([shots, errors, r])
    
    raw_stats = physical, logical
    return raw_stats

def task_sample(args):
    d, p, NoiseModel, scl_noise, max_shots, max_errors, batch_size, kwargs = args
    r = d * 2
    scl_kwargs = {k: p for k in scl_noise}
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=r, distance=d, **scl_kwargs)
    model = NoiseModel(p=p, **kwargs)
    repetitions = int(max_shots / batch_size)
    shots, errors = model.sample_logical_error_rate(circuit, repetitions=repetitions, batch_size=batch_size, max_errors=max_errors, return_error_rate=False)
    return {'d': d, 'r': r, 'p': p, 'shots': shots, 'errors': errors}

# def task_sample_circuit(args):
#     d, p, NoiseModel, scl_noise, shots, kwargs = args
#     r = d * 2
#     model = NoiseModel(p=p, **kwargs)
#     circuit = model.gen_circuit_marginalised(distance=d, rounds=r, scl_noise=scl_noise, rotated=True)
#     sampler = circuit.compile_detector_sampler()
#     detection_events, observable_flips = sampler.sample(shots, separate_observables=True)
#     detector_error_model = circuit.detector_error_model(decompose_errors=True)
#     matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
#     predictions = matcher.decode_batch(detection_events)
#     errors = np.sum(predictions != observable_flips)
#     return {'d': d, 'r': r, 'p': p, 'shots': shots, 'errors': errors}