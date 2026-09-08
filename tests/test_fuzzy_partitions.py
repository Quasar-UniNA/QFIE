import numpy as np
import pytest

from QFIE.fuzzy_partitions import fuzzy_partition
from QFIE.FuzzyEngines import QuantumFuzzyEngine


@pytest.mark.parametrize(
    ("encoding", "minimize_hamming", "expected"),
    [
        (
            "logaritmic",
            False,
            {"a": "00", "b": "10", "c": "01"},
        ),
        (
            "logaritmic",
            True,
            {"a": "00", "b": "01", "c": "11"},
        ),
        (
            "linear",
            False,
            {"a": "001", "b": "010", "c": "100"},
        ),
    ],
)
def test_quantum_state_association(encoding, minimize_hamming, expected):
    partition = fuzzy_partition(
        "input",
        ["a", "b", "c"],
        encoding=encoding,
        minimize_hamming=minimize_hamming,
    )

    assert partition.associate_quantum_states() == expected


@pytest.mark.parametrize(
    ("exact_partition", "expected_register_size"),
    [
        (False, 3),
        (True, 2),
    ],
)
def test_register_size_drops_garbage_qubit_only_when_exact(
    exact_partition, expected_register_size
):
    # 4 sets is a power of two: the garbage qubit is only needed when the
    # partition is not exact (membership values don't sum to 1 everywhere).
    partition = fuzzy_partition(
        "input",
        ["a", "b", "c", "d"],
        minimize_hamming=True,
        exact_partition=exact_partition,
    )

    assert partition.register_size() == expected_register_size
    assert all(
        len(code) == expected_register_size
        for code in partition.associate_quantum_states().values()
    )


def test_register_size_unaffected_when_set_count_is_not_a_power_of_two():
    # 3 is not a power of two: ceil(log2(3)) == ceil(log2(3 + 1)), so exactness
    # cannot save a qubit here regardless of the flag.
    exact = fuzzy_partition(
        "input", ["a", "b", "c"], minimize_hamming=True, exact_partition=True
    )
    inexact = fuzzy_partition(
        "input", ["a", "b", "c"], minimize_hamming=True, exact_partition=False
    )

    assert exact.register_size() == inexact.register_size() == 2


def test_add_input_fuzzysets_detects_exact_partition():
    universe = np.array([0, 1, 2, 3])
    engine = QuantumFuzzyEngine(verbose=False, encoding="logaritmic")
    engine.input_variable("var", universe)

    engine.add_input_fuzzysets(
        "var",
        ["a", "b", "c", "d"],
        [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ],
    )

    partition = engine.input_partitions["var"]
    assert partition.exact_partition is True
    assert partition.register_size() == 2


def test_add_input_fuzzysets_detects_non_exact_partition():
    universe = np.array([0, 1, 2, 3])
    engine = QuantumFuzzyEngine(verbose=False, encoding="logaritmic")
    engine.input_variable("var", universe)

    engine.add_input_fuzzysets(
        "var",
        ["a", "b", "c", "d"],
        [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.5]),  # gap at x=3: membership sums to 0.5
        ],
    )

    partition = engine.input_partitions["var"]
    assert partition.exact_partition is False
    assert partition.register_size() == 3


def test_build_inference_qc_rejects_memberships_summing_above_one():
    universe = np.array([0.0, 1.0])
    engine = QuantumFuzzyEngine(verbose=False, encoding="logaritmic")
    engine.input_variable("x", universe)
    engine.add_input_fuzzysets(
        "x", ["a", "b"], [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
    )

    with pytest.raises(
        Exception, match="Sum of memberships must be less than or equal to 1"
    ):
        engine.build_inference_qc({"x": 0.0})
