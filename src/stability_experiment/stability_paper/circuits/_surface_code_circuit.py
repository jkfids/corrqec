from typing import List, Iterable, Dict

import stim

from stability_paper.tools import Builder, AtLayer, Tile, surface_code_tiles


def measure_surface_code_tiles(
        tiles: Iterable[Tile],
        *,
        data_init: Dict[complex, str],
        data_measure: Dict[complex, str],
        out: Builder,
        layer: int):
    out.gate("R", [tile.measure_qubit for tile in tiles] + list(data_init.keys()))
    out.tick()

    # Figure out what basis each qubit needs to be rotated into in each layer.
    x_basis_layers = [set(d for d, b in data_init.items() if b == 'X')]
    for k in range(4):
        layer_xs = set()
        for tile in tiles:
            d = tile.ordered_data[k]
            if tile.basis == 'X' and d is not None:
                layer_xs.add(d)
            layer_xs.add(tile.measure_qubit)
        x_basis_layers.append(layer_xs)
    x_basis_layers.append(set(d for d, b in data_measure.items() if b == 'X'))

    for k in range(4):
        switch = x_basis_layers[k + 1] ^ x_basis_layers[k]
        if switch:
            out.gate("H", switch)
            out.tick()

        for tile in tiles:
            if tile.ordered_data[k] is not None:
                out.gate("CZ", [tile.ordered_data[k], tile.measure_qubit])
        out.tick()
    
    out.gate("H", x_basis_layers[-1] ^ x_basis_layers[-2])
    out.tick()

    out.measure([tile.measure_qubit for tile in tiles] + list(data_measure.keys()), layer=layer)


def build_surface_code_circuit(*, tiles: List[Tile], rounds: int, time_basis: str, builder: Builder) -> None:
    assert rounds >= 2
    data_set = {q for tile in tiles for q in tile.data_set}

    # Initialize.
    cur_layer = 0
    measure_surface_code_tiles(
        tiles,
        out=builder,
        layer=cur_layer,
        data_init={d: time_basis for d in data_set},
        data_measure={},
    )
    for tile in tiles:
        if tile.basis == time_basis:
            builder.detector([AtLayer(tile.measure_qubit, cur_layer)], pos=tile.measure_qubit)
    builder.shift_coords(dt=1)
    builder.tick()
    cur_layer += 1

    # Loop body.
    if rounds > 2:
        loop_builder = Builder(q2i=builder.q2i, circuit=stim.Circuit(), tracker=builder.tracker)
        measure_surface_code_tiles(tiles, out=loop_builder, layer=cur_layer, data_measure={}, data_init={})
        for tile in tiles:
            loop_builder.detector([
                AtLayer(tile.measure_qubit, cur_layer),
                AtLayer(tile.measure_qubit, cur_layer - 1),
            ], pos=tile.measure_qubit)
        loop_builder.shift_coords(dt=1)
        loop_builder.tick()
        builder.circuit += loop_builder.circuit * (rounds - 2)
        cur_layer += 1

    # Measure.
    measure_surface_code_tiles(
        tiles,
        out=builder,
        layer=cur_layer,
        data_init={},
        data_measure={d: time_basis for d in data_set},
    )
    for tile in tiles:
        builder.detector([
            AtLayer(tile.measure_qubit, cur_layer),
            AtLayer(tile.measure_qubit, cur_layer - 1),
        ], pos=tile.measure_qubit)
    builder.shift_coords(dt=1)
    for tile in tiles:
        if tile.basis == time_basis:
            builder.detector([AtLayer(q, cur_layer) for q in tile.used_set], pos=tile.measure_qubit)


def surface_code_stability_experiment_circuit(*, diam: int, rounds: int, basis: str) -> stim.Circuit:
    tiles = surface_code_tiles(diam=diam, top_bot_basis=basis, side_basis=basis, x_order=[0, 1j, 1, 1 + 1j], z_order=[0, 1, 1j, 1 + 1j])
    other_basis = 'X' if basis == 'Z' else 'Z'
    builder = Builder.for_qubits({q for tile in tiles for q in tile.used_set})
    build_surface_code_circuit(tiles=tiles, rounds=rounds, time_basis=other_basis, builder=builder)
    last_layer = 2 if rounds > 2 else 1
    builder.obs_include([AtLayer(tile.measure_qubit, last_layer) for tile in tiles if tile.basis == basis], obs_index=0)
    return builder.circuit


def surface_code_memory_experiment_circuit(*, diam: int, rounds: int, basis: str) -> stim.Circuit:
    tiles = surface_code_tiles(diam=diam, top_bot_basis='Z', side_basis='X', x_order=[0, 1j, 1, 1 + 1j], z_order=[0, 1, 1j, 1 + 1j])
    builder = Builder.for_qubits({q for tile in tiles for q in tile.used_set})
    build_surface_code_circuit(tiles=tiles, rounds=rounds, time_basis=basis, builder=builder)
    data_set = {d for tile in tiles for d in tile.data_set}
    last_layer = 2 if rounds > 2 else 1
    if basis == 'X':
        builder.obs_include([AtLayer(d, last_layer) for d in data_set if d.imag == 0], obs_index=0)
    else:
        builder.obs_include([AtLayer(d, last_layer) for d in data_set if d.real == 0], obs_index=0)
    return builder.circuit
