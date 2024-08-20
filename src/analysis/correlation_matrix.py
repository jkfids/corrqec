import numpy as np
import pandas as pd

from src.noisemodel import get_partitioned_qubit_coords, get_detector_coords

def calc_tcorr_matrix(detection_events, circuit, rotated=True):
    formatted_detection_events = format_detection_events(detection_events, circuit, rotated)
    data, sz, sx = get_partitioned_qubit_coords(circuit)
    n_syndrome = len(sz) + len(sx)
    nt = int(detection_events.shape[1]/n_syndrome) - 1
    tcorr_matrix = pd.DataFrame(np.zeros((nt, nt)))
    for i in range(n_syndrome):
        df = pd.DataFrame(formatted_detection_events[:, i*nt:(i+1)*nt])
        tcorr_matrix += df.corr()
    tcorr_matrix /= n_syndrome
    return tcorr_matrix

def format_detection_events(detection_events, circuit, rotated=True):
    
    data, sz, sx = get_partitioned_qubit_coords(circuit)
    n_syndrome = len(sz) + len(sx)
    margin = int(0.5 * n_syndrome)
    
    detection_events = detection_events[:, margin:-margin]
    rounds = int(detection_events.shape[1]/n_syndrome) + 1
    
    shots = detection_events.shape[0]
    
    idx = get_detector_idx_for_sorting(circuit, rotated=rotated)
    
    formatted_detection_events = detection_events.reshape(shots, rounds-1, n_syndrome)[:, :, idx].transpose(0, 2, 1).reshape(shots, (rounds-1)*n_syndrome)
    
    return formatted_detection_events

def get_detector_idx_for_sorting(circuit, rotated=True):
    data, sz, sx = get_partitioned_qubit_coords(circuit)
    detector_coords = get_detector_coords(circuit)
    
    if rotated is True:
        data, sz, sx = ({q: rotate_coords45(coord).tolist() for q, coord in partition.items()} for partition in (data, sz, sx))
        detector_coords = [rotate_coords45(coord).tolist() for coord in detector_coords]

    sz_enum = []
    sx_enum = []

    for i, coord in enumerate(detector_coords):
        if coord in sz.values():
            sz_enum.append([i, *coord])
        elif coord in sx.values():
            sx_enum.append([i, *coord])
        
    sz_enum_sorted = sorted(sz_enum, key= lambda x: [x[1], -x[2]])
    sx_enum_sorted = sorted(sx_enum, key= lambda x: [-x[2], x[1]])

    idx = [i[0] for i in sz_enum_sorted] + [i[0] for i in sx_enum_sorted]
    
    return idx

def rotate_coords45(coords):
    theta = np.radians(-45)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    rotated_coords = np.dot(np.array(coords), rotation_matrix.T)
    C = 1/np.sqrt(2)
    normalised = C * rotated_coords
    
    return np.rint(normalised).astype(int)