#!/usr/bin/env python3

import argparse
import pathlib
import sys

from stability_paper.circuits import surface_code_stability_experiment_circuit, \
    surface_code_memory_experiment_circuit
from stability_paper.tools import NoiseModel
from stability_paper.tools._noise import NoiseRule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--bases", nargs='+', required=True, type=str)
    parser.add_argument("--measure_noise", nargs='+', required=True, type=float)
    parser.add_argument("--data_noise", nargs='+', required=True, type=float)
    parser.add_argument("--type", choices=['stability', 'memory'], required=True)
    parser.add_argument("--rounds", nargs='+', required=True, type=int)
    parser.add_argument("--diams", nargs='+', required=True, type=int)
    args = parser.parse_args()

    if args.type == 'stability':
        method = surface_code_stability_experiment_circuit
    elif args.type == 'memory':
        method = surface_code_memory_experiment_circuit
    else:
        raise NotImplementedError(f'{args.type=}')

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    for measure_noise in args.measure_noise:
        for data_noise in args.data_noise:
            noise = NoiseModel(
                idle_depolarization=data_noise,
                any_clifford_1q_rule=NoiseRule(after={'DEPOLARIZE1': data_noise}),
                measure_rules={
                    'Z': NoiseRule(after={'DEPOLARIZE1': measure_noise}, flip_result=measure_noise),
                },
                gate_rules={
                    'R': NoiseRule(after={'X_ERROR': measure_noise}),
                    'CZ': NoiseRule(after={'DEPOLARIZE2': data_noise}),
                }
            )

            for basis in args.bases:
                for diam in args.diams:
                    for rounds in args.rounds:
                        circuit = method(
                            basis=basis,
                            rounds=rounds,
                            diam=diam)
                        noisy_circuit = noise.noisy_circuit(circuit)
                        json_metadata = {
                            'type': args.type,
                            'b': basis,
                            'd': diam,
                            'r': rounds,
                            'pm': measure_noise,
                            'pd': data_noise,
                        }
                        name = ','.join(f'{k}={json_metadata[k]}' for k in sorted(json_metadata.keys()))
                        path = out_dir / f'{name}.stim'
                        with open(path, 'w') as f:
                            print(noisy_circuit, file=f)
                        print(f'wrote {path}', file=sys.stderr)


if __name__ == '__main__':
    main()
