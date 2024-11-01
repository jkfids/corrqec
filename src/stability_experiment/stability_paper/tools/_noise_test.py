import stim

from stability_paper.tools._noise import _measure_basis, _iter_split_op_moments, _occurs_in_classical_control_system


def test_measure_basis():
    f = lambda e: _measure_basis(split_op=stim.Circuit(e)[0])
    assert f('H') is None
    assert f('H 0') is None
    assert f('R 0 1 2') is None

    assert f('MX') == 'X'
    assert f('MX(0.01) 1') == 'X'
    assert f('MY 0 1') == 'Y'
    assert f('MZ 0 1 2') == 'Z'
    assert f('M 0 1 2') == 'Z'

    assert f('MRX') is None

    assert f('MPP X5') == 'X'
    assert f('MPP X0*X2') == 'XX'
    assert f('MPP Y0*Z2*X3') == 'YZX'


def test_iter_split_op_moments():
    assert list(_iter_split_op_moments(stim.Circuit("""
    """))) == []

    assert list(_iter_split_op_moments(stim.Circuit("""
        H 0
    """))) == [
        [stim.CircuitInstruction('H', [0])]
    ]

    assert list(_iter_split_op_moments(stim.Circuit("""
        H 0
        TICK
    """))) == [
        [stim.CircuitInstruction('H', [0])]
    ]

    assert list(_iter_split_op_moments(stim.Circuit("""
        H 0
        TICK
        H 1
    """))) == [
        [stim.CircuitInstruction('H', [0])],
        [stim.CircuitInstruction('H', [1])],
    ]

    assert list(_iter_split_op_moments(stim.Circuit("""
        CX rec[-1] 0 1 2 3 4
        MPP X5*X6 Y5
        CX 8 9 10 11
        TICK
        H 0
    """))) == [
        [
            stim.CircuitInstruction('CX', [stim.target_rec(-1), 0]),
            stim.CircuitInstruction('CX', [1, 2]),
            stim.CircuitInstruction('CX', [3, 4]),
            stim.CircuitInstruction('MPP', [stim.target_x(5), stim.target_combiner(), stim.target_x(6)]),
            stim.CircuitInstruction('MPP', [stim.target_y(5)]),
            stim.CircuitInstruction('CX', [8, 9, 10, 11]),
        ],
        [
            stim.CircuitInstruction('H', [0]),
        ],
    ]


def test_occurs_in_classical_control_system():
    assert not _occurs_in_classical_control_system(split_op=stim.CircuitInstruction('H', [0]))
    assert not _occurs_in_classical_control_system(split_op=stim.CircuitInstruction('CX', [0, 1, 2, 3]))
    assert not _occurs_in_classical_control_system(split_op=stim.CircuitInstruction('M', [0, 1, 2, 3]))

    assert _occurs_in_classical_control_system(split_op=stim.CircuitInstruction('CX', [stim.target_rec(-1), 0]))
    assert _occurs_in_classical_control_system(split_op=stim.CircuitInstruction('DETECTOR', [stim.target_rec(-1)]))
    assert _occurs_in_classical_control_system(split_op=stim.CircuitInstruction('TICK', []))
    assert _occurs_in_classical_control_system(split_op=stim.CircuitInstruction('SHIFT_COORDS', []))
