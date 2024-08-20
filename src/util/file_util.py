# Standard library imports
import csv
import os
import re
from typing import List

# Third-party library imports
import sinter

def gen_csv_filepaths(path: str, name: str, counter: int = None) -> tuple:
    
    correlated = f"{path}{name}_correlated"
    independent = f"{path}{name}_independent"
    
    if counter is not None:
        correlated_csv = f"{correlated}_{counter}.csv"
        independent_csv = f"{independent}_{counter}.csv"
    elif counter is None:
        i = 1
        while os.path.exists(correlated + '.csv') or os.path.exists(independent + '.csv'):
            correlated = f"{path}{name}_correlated_{i}"
            independent = f"{path}{name}_independent_{i}"
            i += 1
        correlated_csv = f"{correlated}.csv"
        independent_csv = f"{independent}.csv"
    return correlated_csv, independent_csv
        
def gen_csv_filepath_list(path: str, name_list: List, counter: int = None) -> str:
    
    filepaths = [f"{path}{name}" for name in name_list]
    
    if counter is not None:
        filepaths = [f"{path}{name}_{counter}.csv" for name in name_list]
    elif counter is None:
        i = 1
        while any([os.path.exists(filepath + '.csv') for filepath in filepaths]):
            filepaths = [f"{path}{name}_{i}" for name in name_list]
            i += 1
        filepaths = [f"{filepath}.csv" for filepath in filepaths]
    return filepaths
    

def gen_filepath(path: str, name: str, counter: int = None) -> str:
    """Generates a unique filepath by appending a counter if a collision occurs.

    Args:
        path: The directory path where the file will be saved.
        name: The desired filename including the extension (e.g., "report.csv").

    Returns:
        A unique filepath with the same extension as provided in the name parameter.
    """
    filepath = os.path.join(path, name)  # Use os.path.join for platform-independent paths
    
    if counter is not None:
        base, ext = os.path.splitext(name)
        filepath = f"{path}{base}_{counter}{ext}"
    else:
        i = 1
        while os.path.exists(filepath):
            base, ext = os.path.splitext(name)  # Separate base name and extension
            filepath = f"{path}{base}_{i}{ext}"  # Combine with counter and extension
            i += 1

    return filepath

def collected_stats_to_csv(stats: sinter.TaskStats, filepath: str):
    with open(filepath, 'w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(sinter.CSV_HEADER.split(','))
        for stat in stats:
            row = re.split(',\s*(?![^{}]*\})', stat.to_csv_line())
            row[6] = row[6][1:-1].replace('""', '"')  # Remove outer quotes and unescape inner quotes for json_metadata
            writer.writerow(row)

def stats_to_csv(stats, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["shots", "errors", "distance", "rounds", "p"])

        physical, logical = stats
        
        distances = physical.keys()
        for d in distances:
            for p, (shots, errors, r) in zip(physical[d], logical[d]):
                writer.writerow([shots, errors, d, r, p])
                
def stats_from_csv(filepath):
    physical = {}
    logical = {}
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip header
        for row in reader:
            shots = int(row[0])
            errors = int(row[1])
            d = int(row[2])
            r = int(row[3])
            p = float(row[4])
            try:
                physical[d].append(p)
                logical[d].append([shots, errors, r])
            # When dict is empty before first entry
            except KeyError:
                physical[d] = [p]
                logical[d] = [[shots, errors, r]]
                
    return physical, logical