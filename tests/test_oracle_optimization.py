import warnings

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from QFIE.FuzzyEngines import QuantumFuzzyEngine


CONFIGURATIONS = [
    pytest.param(False, False, "one-hot", 2, False, id="standard-one-hot"),
    pytest.param(False, False, "gray", 1, False, id="standard-gray"),
    pytest.param(True, False, "one-hot", 2, False, id="optimized-one-hot"),
    pytest.param(True, True, "one-hot", 2, True, id="optimized-one-hot-ancilla"),
    pytest.param(True, False, "gray", 1, False, id="optimized-gray"),
    pytest.param(True, True, "gray", 1, True, id="optimized-gray-ancilla"),
]

VALID_INPUTS = [
    pytest.param({"x": 0.0, "y": 0.0}, id="low-low"),
    pytest.param({"x": 0.0, "y": 1.0}, id="low-high"),
    pytest.param({"x": 1.0, "y": 0.0}, id="high-low"),
    pytest.param({"x": 1.0, "y": 1.0}, id="high-high"),
]


def make_or_engine():
    """Build a two-input fuzzy system whose ON output minimizes to x OR y."""
    universe = np.array([0.0, 1.0])
    low = np.array([1.0, 0.0])
    high = np.array([0.0, 1.0])

    engine = QuantumFuzzyEngine(verbose=False, encoding="logaritmic")
    engine.input_variable("x", universe)
    engine.input_variable("y", universe)
    engine.output_variable("out", universe)
    engine.add_input_fuzzysets("x", ["low", "high"], [low, high])
    engine.add_input_fuzzysets("y", ["low", "high"], [low, high])
    engine.add_output_fuzzysets("out", ["off", "on"], [low, high])
    engine.set_rules(
        [
            "if x is low and y is low then out is off",
            "if x is low and y is high then out is on",
            "if x is high and y is low then out is on",
            "if x is high and y is high then out is on",
        ]
    )
    return engine


def build_circuit(input_values, optimize, ancilla, output_encoding):
    engine = make_or_engine()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        engine.build_inference_qc(
            input_values,
            optimize=optimize,
            ancilla=ancilla,
            output_encoding=output_encoding,
        )
    return engine.qc["full_circuit"]


def register_by_name(circuit, name):
    return next(register for register in circuit.qregs if register.name == name)


def register_indices(circuit, register):
    return [circuit.find_bit(qubit).index for qubit in register]


def statevector_without_measurements(circuit):
    unitary_circuit = circuit.remove_final_measurements(inplace=False)
    return unitary_circuit, Statevector.from_instruction(unitary_circuit)


def output_probabilities(circuit):
    unitary_circuit, statevector = statevector_without_measurements(circuit)
    output_register = register_by_name(unitary_circuit, "out")
    return statevector.probabilities(
        qargs=register_indices(unitary_circuit, output_register)
    )


@pytest.mark.parametrize(
    (
        "optimize",
        "ancilla",
        "output_encoding",
        "expected_output_size",
        "expects_ancilla",
    ),
    CONFIGURATIONS,
)
def test_all_oracle_configurations_build(
    optimize,
    ancilla,
    output_encoding,
    expected_output_size,
    expects_ancilla,
):
    circuit = build_circuit(
        {"x": 1.0, "y": 1.0},
        optimize=optimize,
        ancilla=ancilla,
        output_encoding=output_encoding,
    )

    assert register_by_name(circuit, "out").size == expected_output_size
    assert any(register.name == "anc" for register in circuit.qregs) is expects_ancilla


@pytest.mark.parametrize("input_values", VALID_INPUTS)
@pytest.mark.parametrize("output_encoding", ["one-hot", "gray"])
def test_optimized_oracles_match_standard_on_valid_inputs(
    input_values,
    output_encoding,
):
    standard = build_circuit(
        input_values,
        optimize=False,
        ancilla=False,
        output_encoding=output_encoding,
    )
    optimized_without_ancillas = build_circuit(
        input_values,
        optimize=True,
        ancilla=False,
        output_encoding=output_encoding,
    )
    optimized_with_ancillas = build_circuit(
        input_values,
        optimize=True,
        ancilla=True,
        output_encoding=output_encoding,
    )

    expected = output_probabilities(standard)
    np.testing.assert_allclose(
        output_probabilities(optimized_without_ancillas),
        expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        output_probabilities(optimized_with_ancillas),
        expected,
        atol=1e-12,
    )


@pytest.mark.parametrize("output_encoding", ["one-hot", "gray"])
def test_ancillas_are_uncomputed(output_encoding):
    circuit = build_circuit(
        {"x": 1.0, "y": 1.0},
        optimize=True,
        ancilla=True,
        output_encoding=output_encoding,
    )
    unitary_circuit, statevector = statevector_without_measurements(circuit)
    ancilla_register = register_by_name(unitary_circuit, "anc")

    ancilla_probabilities = statevector.probabilities(
        qargs=register_indices(unitary_circuit, ancilla_register)
    )

    np.testing.assert_allclose(
        ancilla_probabilities,
        np.array([1.0, 0.0, 0.0, 0.0]),
        atol=1e-12,
    )


def test_overlapping_sop_without_ancillas_emits_fallback_warning():
    engine = make_or_engine()

    with pytest.warns(RuntimeWarning, match="falling back"):
        engine.build_inference_qc(
            {"x": 1.0, "y": 1.0},
            optimize=True,
            ancilla=False,
            output_encoding="one-hot",
        )


def test_invalid_output_encoding_is_rejected():
    engine = make_or_engine()

    with pytest.raises(ValueError, match="output_encoding"):
        engine.build_inference_qc(
            {"x": 0.0, "y": 0.0},
            output_encoding="binary",
        )


def test_distributed_gray_output_is_rejected():
    engine = make_or_engine()

    with pytest.raises(NotImplementedError, match="Distributed QFIE"):
        engine.build_inference_qc(
            {"x": 0.0, "y": 0.0},
            distributed=True,
            output_encoding="gray",
        )
