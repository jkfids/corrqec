# Standard imports
from typing import List

# Third-library imports
import numpy as np
import scipy
import sinter

# Local imports


def calc_per_round(per_shot: float, rounds: int):
    
    if per_shot >= 0.5:
         per_shot = 1 - per_shot
        
    return 0.5 * (1 - (1 - 2 * per_shot) ** (1/rounds))

def process_raw_stats(raw_stats):
    
    physical, sample = raw_stats
    
    logical = {d: [calc_per_round(sample[1]/sample[0], sample[2]) for sample in samples] for d, samples in sample.items()}
    
    stats = physical, logical
    
    return stats

def collate_collected_stats(collected_stats):
    distances = set()
    probabilities = set()
    for stats in collected_stats:
        distances.add(stats.json_metadata['d'])
        probabilities.add(stats.json_metadata['p'])
        
    distances = list(distances)
    probabilities = list(probabilities)
    distances.sort()
    probabilities.sort()
        
    logical = {d: [0] * len(probabilities) for d in distances}
    physical = {d: probabilities.copy() for d in distances}

    for stats in collected_stats:
        d = stats.json_metadata['d']
        p = stats.json_metadata['p']
        r = stats.json_metadata['r']
        i = physical[d].index(p)
        if not stats.errors:
            #print(f"Didn't see any errors for d={d}, p={p}")
            del logical[d][i]
            del physical[d][i]
            continue
        
        per_shot = stats.errors / stats.shots
        per_round = calc_per_round(per_shot, r)
        logical[d][i] = per_round
        
    return physical, logical

def fit_stats_for_projection(stats):
    
    if isinstance(stats[0], sinter._task_stats.TaskStats):
        distance_dict, logical_dict = collate_stats_for_projection(stats)
    else: 
        distance_dict, logical_dict = convert_stats_for_projection(stats)
        
    probabilities = distance_dict.keys()
    fit_dict = {p: scipy.stats.linregress(distance_dict[p], np.log(logical_dict[p])) for p in probabilities}
    
    return distance_dict, logical_dict, fit_dict

def collate_stats_for_projection(collected_stats):
    
    distances = set()
    probabilities = set()
    for stats in collected_stats:
        distances.add(stats.json_metadata['d'])
        probabilities.add(stats.json_metadata['p'])
        
    distances = list(distances)
    probabilities = list(probabilities)
    distances.sort()
    probabilities.sort()
    
    logical = {p: [0] * len(distances) for p in probabilities}
    distances = {p: distances.copy() for p in probabilities}
    
    for stats in collected_stats:
        d = stats.json_metadata['d']
        p = stats.json_metadata['p']
        r = stats.json_metadata['r']
        i = distances[p].index(d)
        if not stats.errors:
            del logical[p][i]
            del distances[p][i]
            continue
        
        per_shot = stats.errors / stats.shots
        per_round = calc_per_round(per_shot, r)
        logical[p][i] = per_round
    
    return distances, logical

def convert_stats_for_projection(stats):
    
    if isinstance(list(stats[1].values())[0][0], List):
        stats = process_raw_stats(stats)

    distance_dict = {}
    logical_dict = {}

    physical, logical = stats
    distances = physical.keys()
    for d in distances:
        for p, l in zip(physical[d], logical[d]):
            try:
                distance_dict[p].append(d)
                logical_dict[p].append(l)
            except KeyError:
                distance_dict[p] = [d]
                logical_dict[p] = [l]
                
    return distance_dict, logical_dict