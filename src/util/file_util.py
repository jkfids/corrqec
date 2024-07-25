# Standard library imports
import csv
import os
import re

# Third-party library imports
import sinter


def gen_filepath(path: str, name: str, override_counter: int=None) -> str:
    """Generates a unique filepath by appending a counter if a collision occurs.

    Args:
        path: The directory path where the file will be saved.
        name: The desired filename including the extension (e.g., "report.csv").

    Returns:
        A unique filepath with the same extension as provided in the name parameter.
    """
    filepath = os.path.join(path, name)  # Use os.path.join for platform-independent paths
    
    if override_counter is not None:
        base, ext = os.path.splitext(name)
        filepath = f"{path}{base}_{override_counter}{ext}"
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
        writer.writerow(["shots", "errors", "distance", "p", "rounds"])

        physical, logical = stats
        
        distances = physical.keys()
        for d in distances:
            for p, (shots, errors, r) in zip(physical[d], logical[d]):
                writer.writerow([shots, errors, d, p, r])
                
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
            p = float(row[3])
            r = int(row[4])
            # l = float(row[2])
            try:
                physical[d].append(p)
                logical[d].append([shots, errors, r])
            # When dict is empty before first entry
            except KeyError:
                physical[d] = [p]
                logical[d] = [[shots, errors, r]]
                
    return physical, logical