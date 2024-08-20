import multiprocessing
from typing import List
from time import time

import numpy as np
import pandas as pd
import sinter
import stim

from ..analysis import calc_tcorr_matrix
from ..util import stats_to_csv, collected_stats_to_csv

# def sample_correlated(num_workers, NoiseModel, model_params, distances, probabilities, scl_noise=None, max_shots=100000, batch_size=1000, print_progress=False):
#     physical = {d: [] for d in distances}
#     logical = {d: [] for d in distances}
    
#     if scl_noise is None:
#         scl_noise = []
#     args_list = [(d, d*2, p, NoiseModel, model_params, scl_noise, max_shots, max_errors, batch_size, print_progress)
#         for d in distances for p in probabilities]
    
#     with multiprocessing.Pool(processes=num_workers) as pool:
#         results = pool.map(task_sample, args_list)
    
#     for result in results:
#         d, r, p, shots, errors = result['d'], result['r'], result['p'], result['shots'], result['errors']
#         if errors == 0:
#             # print(f"No errors for d = {d}, p = {p}")
#             continue
#         physical[d].append(p)
#         logical[d].append([shots, errors, r])
    
#     raw_stats = physical, logical
#     return raw_stats

# def task_sample(args):
#     time0 = time()
#     d, r, p, NoiseModel, model_params, scl_noise, max_shots, max_errors, batch_size, print_progress = args
#     scl_kwargs = {k: p for k in scl_noise}
#     circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=r, distance=d, **scl_kwargs)
#     model = NoiseModel(p=p, **model_params)
#     repetitions = int(max_shots / batch_size)
#     shots, errors = model.sample_logical_error_rate(circuit, repetitions=repetitions, batch_size=batch_size, max_errors=max_errors, return_error_rate=False)
#     time1 = time()
#     if print_progress:
#         print(f"Sampling time (d = {d}, p = {p}): {time1 - time0:.2f}s")
#     return {'d': d, 'r': r, 'p': p, 'shots': shots, 'errors': errors}

def sample_independent_to_file(filepath, num_workers, NoiseModel, model_params, distances, probabilities, scl_noise, max_shots, print_progress):
    tasks = [
        sinter.Task(
            circuit=NoiseModel(p=p, **model_params).gen_circuit_marginalised(distance=d, rounds=d*2, scl_noise=scl_noise, rotated=True),
            json_metadata={'d': d, 'r': d*2, 'p': p},
        )
        for d in distances
        for p in probabilities
    ]
    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        save_resume_filepath=filepath,
        decoders=['pymatching'],
        max_shots=max_shots,
        max_errors=max_shots,
        print_progress=print_progress,
    )
    collected_stats_to_csv(collected_stats, filepath)
    return collected_stats

def sample_correlated_to_file(filepath, num_workers, NoiseModel, model_params, distances, probabilities, scl_noise, max_shots, print_progress):
    stats = sample_correlated(
        num_workers=num_workers,
        NoiseModel=NoiseModel,
        model_params=model_params,
        distances=distances,
        probabilities=probabilities,
        scl_noise=scl_noise,
        max_shots=max_shots,
        batch_size=1000,
        print_progress=print_progress
    )
    stats_to_csv(stats, filepath)
    return stats

def sample_correlated(num_workers, NoiseModel, model_params, distances, probabilities, scl_noise=None, max_shots=100000, batch_size=1000, print_progress=False):
    physical = {d: [] for d in distances}
    logical = {d: [] for d in distances}
    
    if scl_noise is None:
        scl_noise = []
    repetitions = int(np.ceil(max_shots / batch_size))
    args_list = [(d, d*2, p, NoiseModel, model_params, scl_noise, batch_size, print_progress) 
                 for i in range(repetitions) for d in distances for p in probabilities]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(task_sample, args_list)
    
    df = pd.DataFrame(results, columns=['d', 'r', 'p', 'shots', 'errors'])
    df = df.groupby(['d', 'r', 'p']).agg({'shots': 'sum', 'errors': 'sum'}).reset_index()
    results = df.to_numpy()
    
    for row in results:
        d, r, shots, errors = row[[0, 1, 3, 4]].astype(int)
        p = row[2]
        if errors == 0:
            # print(f"No errors for d = {d}, r = {r}, p = {p}, shots = {shots}")
            continue
        physical[d].append(p)
        logical[d].append([shots, errors, r])
        
    raw_stats = physical, logical
    print(raw_stats)
    return raw_stats

def task_sample(args):
    time0 = time()
    d, r, p, NoiseModel, model_params, scl_noise, batch_size, print_progress = args
    scl_kwargs = {k: p for k in scl_noise}
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=r, distance=d, **scl_kwargs)
    detector_error_model_circuit = stim.Circuit.generated("surface_code:rotated_memory_z", 
                                                          rounds=r, 
                                                          distance=d, 
                                                          after_clifford_depolarization=p,
                                                          before_round_data_depolarization=p,
                                                          before_measure_flip_probability=p,
                                                          after_reset_flip_probability=p)
    detector_error_model = detector_error_model_circuit.detector_error_model()
    model = NoiseModel(p=p, **model_params)
    # shots, errors = model.sample_logical_error_rate(circuit, repetitions=1, batch_size=batch_size, return_error_rate=False, detector_error_model=detector_error_model)
    shots, errors = model.sample_logical_error_rate(circuit, repetitions=1, batch_size=batch_size, return_error_rate=False)
    time1 = time()
    # if print_progress:
    #     print(f"Sampling time (d = {d}, r = {r}, p = {p}, shots = {shots}): {time1 - time0:.2f}s")
    return [d, r, p, shots, errors]

def sample_tcorr_matrix(NoiseModel, model_params, distance, rounds, scl_noise=None, shots=1000):
    if scl_noise is None:
        scl_noise = []
    scl_kwargs = {k: model_params["p"] for k in scl_noise}
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=rounds, distance=distance, **scl_kwargs)
    model = NoiseModel(**model_params)
    detection_events, observable_flips = model.sample_circuit(circuit=circuit, shots=shots)
    tcorr_matrix = calc_tcorr_matrix(detection_events, circuit)
    return tcorr_matrix

# def sample_raw_correlated(NoiseModel, model_params, distance, rounds, scl_noise=None, shots=1000):
#     if scl_noise is None:
#         scl_noise = []
#     scl_kwargs = {k: model_params["p"] for k in scl_noise}
#     circuit = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=rounds, distance=distance, **scl_kwargs)
#     model = NoiseModel(**model_params)
#     detection_events, observable_flips = model.sample_circuit(circuit=circuit, shots=shots)
#     return detection_events, observable_flips
