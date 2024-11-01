from stability_paper.tools import surface_code_tiles, Tile


def test_surface_code_tiles():
    tiles = surface_code_tiles(
        diam=2,
        side_basis='X',
        top_bot_basis='Z',
        z_order=[0, 1, 1j, 1 + 1j],
        x_order=[0, 1j, 1, 1 + 1j],
    )
    assert tiles == [
        Tile(ordered_data=(None, None, 0j, 1), measure_qubit=(0.5-0.5j), basis='Z'),
        Tile(ordered_data=(0j, 1j, 1, 1+1j), measure_qubit=(0.5+0.5j), basis='X'),
        Tile(ordered_data=(1j, 1+1j, None, None), measure_qubit=(0.5+1.5j), basis='Z'),
    ]
