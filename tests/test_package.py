import QFIE
from QFIE import FuzzyEngines
from qiskit_aer import AerSimulator


def test_package_version():
    assert QFIE.__version__ == "1.2.0"


def test_aer_simulator_is_available_as_required_dependency():
    assert FuzzyEngines.AerSimulator is AerSimulator
