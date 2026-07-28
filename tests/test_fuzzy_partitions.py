import pytest

from QFIE.fuzzy_partitions import fuzzy_partition


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
