QFIE's documentation
========================================

How to obtain the code
======================

Open Source
-----------
The ``QFIE`` package is open source and available at https://github.com/Quasar-UniNA/QFIE

Installation
------------
The package can be installed manually:

.. code-block:: bash

  git clone https://github.com/Quasar-UniNA/QFIE.git QFIE
  cd QFIE
  pip install .

or if you are planning to extend or develop code replace last command with:

.. code-block:: bash

  pip install -e .

Motivation
==========

- QFIE Python package gives the opportunity of easily implementing quantum fuzzy inference engines as those proposed in: 10.1109/TFUZZ.2022.3202348

- QFIE docs is equipped with two notebook jupyter examples:
    1. QFIE for controlling a Fan Speed System
    2. QFIE for Lighting Control.

How to cite ``QFIE``?
=============================

When using this software in your research, please cite the following publication:

Bibtex:

.. code-block:: latex

  @ARTICLE{9869303,
  author={Acampora, Giovanni and Schiattarella, Roberto and Vitiello, Autilia},
  journal={IEEE Transactions on Fuzzy Systems},
  title={On the Implementation of Fuzzy Inference Engines on Quantum Computers},
  year={2022},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TFUZZ.2022.3202348}}

.. toctree::
   :hidden:
   :caption: API:

   modules

.. toctree::
   :hidden:
   :caption: Examples:

   Fan_Speed_Control
   Lighting_Control


Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`