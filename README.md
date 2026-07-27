# Quantum Fuzzy Inference Engine

[![Made at QuasarLab](https://img.shields.io/badge/UniNA-QuasarLab-blue)](http://quasar.unina.it)
[![Documentation](https://img.shields.io/badge/docs-Read_the_Docs-brightgreen)](https://qfie.readthedocs.io/en/latest/)
[![Related paper](https://img.shields.io/badge/paper-IEEE-orange)](https://doi.org/10.1109/TFUZZ.2022.3202348)

QFIE is a Python package for building and executing Quantum Fuzzy Inference
Engines. It implements the approach introduced in *On the Implementation of
Fuzzy Inference Engines on Quantum Computers*.

## Installation

QFIE requires Python 3.10 or later. Install the package from PyPI with:

```bash
pip install QFIE
```



To install the latest development version directly from the repository:

```bash
pip install "git+https://github.com/Quasar-UniNA/QFIE.git"
```

The complete user guide and examples are available in the
[QFIE documentation](https://qfie.readthedocs.io/en/latest/).

## Research using QFIE

QFIE has been used in several application domains and extensions:

- Particle accelerator control at CERN:
  [*Quantum Fuzzy Inference Engine for Particle Accelerator Control*](https://doi.org/10.1109/TQE.2024.3374251),
  IEEE Transactions on Quantum Engineering, 2024.
- Edge detection on real NISQ hardware:
  [*Quantum Fuzzy Logic for Edge Detection: A Demonstration on NISQ Hardware*](https://doi.org/10.1016/j.asoc.2025.113866),
  Applied Soft Computing, 2025.
- Control systems for smart cities:
  [*Using Quantum Fuzzy Inference Engines in Smart Cities*](https://doi.org/10.1109/FUZZ-IEEE60900.2024.10611863),
  FUZZ-IEEE 2024.
- Interval type-2 Mamdani fuzzy systems:
  [*Hybrid Quantum-Classical Interval Type-2 Mamdani Fuzzy Systems*](https://doi.org/10.1109/FUZZ62266.2025.11152074),
  FUZZ-IEEE 2025.
- Distribution across multiple NISQ devices:
  [*Distributing Fuzzy Inference Engines on Quantum Computers*](https://doi.org/10.1109/FUZZ52849.2023.10309786),
  FUZZ-IEEE 2023.

## Citation

If you use QFIE in your research, please cite the foundational paper:

```bibtex
@article{acampora2023implementation,
  author  = {Acampora, Giovanni and Schiattarella, Roberto and Vitiello, Autilia},
  title   = {On the Implementation of Fuzzy Inference Engines on Quantum Computers},
  journal = {IEEE Transactions on Fuzzy Systems},
  volume  = {31},
  number  = {5},
  pages   = {1419--1433},
  year    = {2023},
  doi     = {10.1109/TFUZZ.2022.3202348}
}
```

<details>
<summary>BibTeX entries for QFIE applications and extensions</summary>

```bibtex
@article{acampora2024quantum,
  author    = {Acampora, Giovanni and Grossi, Michele and Schenk, Michael and Schiattarella, Roberto},
  title     = {Quantum Fuzzy Inference Engine for Particle Accelerator Control},
  journal   = {IEEE Transactions on Quantum Engineering},
  volume    = {5},
  pages     = {1--13},
  year      = {2024},
  publisher = {IEEE},
  doi       = {10.1109/TQE.2024.3374251}
}

@article{nunziata2025quantum,
  author    = {Nunziata, G. and Crisci, S. and De Gregorio, G. and Schiattarella, R. and Acampora, G. and Coraggio, L. and Itaco, N.},
  title     = {Quantum Fuzzy Logic for Edge Detection: A Demonstration on {NISQ} Hardware},
  journal   = {Applied Soft Computing},
  volume    = {185},
  pages     = {113866},
  year      = {2025},
  publisher = {Elsevier},
  doi       = {10.1016/j.asoc.2025.113866}
}

@inproceedings{acampora2024using,
  author    = {Acampora, Giovanni and Schiattarella, Roberto and Vitiello, Autilia},
  title     = {Using Quantum Fuzzy Inference Engines in Smart Cities},
  booktitle = {2024 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)},
  pages     = {1--8},
  year      = {2024},
  organization = {IEEE},
  doi       = {10.1109/FUZZ-IEEE60900.2024.10611863}
}

@inproceedings{acampora2025hybrid,
  author    = {Acampora, Giovanni and Schiattarella, Roberto and Vitiello, Autilia},
  title     = {Hybrid Quantum-Classical Interval Type-2 Mamdani Fuzzy Systems},
  booktitle = {2025 IEEE International Conference on Fuzzy Systems (FUZZ)},
  pages     = {1--6},
  year      = {2025},
  organization = {IEEE},
  doi       = {10.1109/FUZZ62266.2025.11152074}
}

@inproceedings{acampora2023distributing,
  author    = {Acampora, Giovanni and Massa, Alfredo and Schiattarella, Roberto and Vitiello, Autilia},
  title     = {Distributing Fuzzy Inference Engines on Quantum Computers},
  booktitle = {2023 IEEE International Conference on Fuzzy Systems (FUZZ)},
  pages     = {1--6},
  year      = {2023},
  organization = {IEEE},
  doi       = {10.1109/FUZZ52849.2023.10309786}
}
```

</details>

## License

QFIE is distributed under the [MIT License](LICENSE.md).
