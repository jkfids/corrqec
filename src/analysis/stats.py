# Standard imports
from typing import List

# Third-library imports
import numpy as np
import scipy
import sinter
from scipy.stats import norm

# Local imports
from ..util import stats_from_csv

def filepaths_to_data_correlated(filepaths):
    for filepath in filepaths:
        stats = stats_from_csv(filepath)
        data = convert_stats_for_projection(stats)
        yield data
        
def filepaths_to_data_independent(filepaths):
    for filepath in filepaths:
        stats = sinter.read_stats_from_csv_files(filepath)
        data = collate_stats_for_projection(stats)
        yield data
        
def format_data_for_plotting(data):
    
    x = next(iter(data[0].values()))
    per_rounds = np.array(next(iter(data[1].values())))
    y = per_rounds[:, 0]
    yerr = per_rounds[:, 1:].T
    
    return x, y, yerr

def calc_per_round_ci(errors: int, shots: int, rounds: int, confidence: float = 0.95):
    per_shot = errors / shots
    per_round = calc_per_round(per_shot, rounds)
    
    # Standard error of the mean
    se_per_shot = np.sqrt(per_shot * (1 - per_shot) / shots)
    
    # Z-score for 95% confidence interval
    z_score = norm.ppf(0.975)  # 1.96
    
    # Calculate the upper and lower bounds of the per-shot error rate
    me_per_shot = z_score * se_per_shot
    per_shot_lower = per_shot - me_per_shot
    per_shot_upper = per_shot + me_per_shot
        
    # Calculate the confidence interval
    lower = per_round - calc_per_round(per_shot_lower, rounds)
    upper = calc_per_round(per_shot_upper, rounds) - per_round
    
    return per_round, lower, upper
def calc_per_round(per_shot: float, rounds: int):
    
    if per_shot >= 0.5:
         per_shot = 1 - per_shot
        
    return 0.5 * (1 - (1 - 2 * per_shot) ** (1/rounds))

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
        
        # per_shot = stats.errors / stats.shots
        per_round = calc_per_round_ci(stats.errors, stats.shots, r)
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

def process_raw_stats(raw_stats):
    
    physical, sample = raw_stats
    
    logical = {d: [calc_per_round_ci(sample[1], sample[0], sample[2]) for sample in samples] for d, samples in sample.items()}
    
    stats = physical, logical
    
    return stats