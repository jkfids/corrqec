import sinter
import stim

from stability_paper.tools._util import circuit_has_unsigned_stabilizers, \
    score_binomial_line


def test_circuit_has_unsigned_stabilizers():
    assert circuit_has_unsigned_stabilizers(
        stim.Circuit("H 0"),
        [
            ({"X": [0]}, {"Z": [0]}, []),
            ({"Z": [0]}, {"X": [0]}, []),
            ({"Y": [0]}, {"Y": [0]}, []),
        ],
    )
    assert circuit_has_unsigned_stabilizers(
        stim.Circuit("CX 1 3"),
        [
            ({"X": [0]}, {"X": [0]}, []),
            ({"X": [1]}, {"X": [1, 3]}, []),
        ],
    )
    assert not circuit_has_unsigned_stabilizers(
        stim.Circuit("CX 1 3"),
        [
            ({"X": [3]}, {"X": [1, 3]}, []),
        ],
    )
    assert circuit_has_unsigned_stabilizers(
        stim.Circuit("RY 1"),
        [
            ({}, {"Y": [1]}, []),
        ],
    )
    assert not circuit_has_unsigned_stabilizers(
        stim.Circuit("RX 1"),
        [
            ({}, {"Y": [1]}, []),
        ],
    )
    assert not circuit_has_unsigned_stabilizers(
        stim.Circuit("MY 3"),
        [
            ({"Y": [3]}, {}, []),
        ],
    )
    assert circuit_has_unsigned_stabilizers(
        stim.Circuit("MY 3"),
        [
            ({"Y": [3]}, {}, [stim.target_rec(-1)]),
        ],
    )


def test_score_binomial_line():
    sb_xs_fit = [14.696938456699069, 17.146428199482248, 19.595917942265423, 22.045407685048602, 24.49489742783178, 7.3484692283495345, 9.797958971132712, 12.24744871391589]
    sb_shots_fit = [100000000, 100000000, 100000000, 100000000, 100000000, 3573, 1193318, 71428572]
    sb_errors_fit = [0, 0, 0, 0, 0, 10, 10, 10]
    a, b, c, d = score_binomial_line(
        min_x_lim=0,
        max_x_lim=1000,
        min_p_lim=1e-12,
        max_p_lim=1,
        xs=sb_xs_fit,
        shots=sb_shots_fit,
        errors=sb_errors_fit,
        max_likelihood_factor=1000,
        y_distortion=lambda y: sinter.shot_error_rate_to_piece_error_rate(shot_error_rate=y, pieces=3) if 0 < y < 1 else y
    )
    assert 0 < b[0] < 10**10
    assert 0 < b[1] < 10**10
