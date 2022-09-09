from qiskit import (
    QuantumCircuit,
    QuantumRegister,
)
import math
from . import fuzzy_partitions as fp

Qregisters = []


def generate_circuit(fuzzy_partitions):
    """Function generating a quantum circuit with width required by QFS"""
    qc = QuantumCircuit()
    for partition in fuzzy_partitions:
        # print(partition.len_partition(), partition.name)
        qc.add_register(
            QuantumRegister(
                math.ceil(math.log(partition.len_partition() + 1, 2)),
                name=partition.name,
            )
        )
        Qregisters.append(
            QuantumRegister(
                math.ceil(math.log(partition.len_partition() + 1, 2)),
                name=partition.name,
            )
        )

    return qc


def output_register(qc, output_partition):
    qc.add_register(
        QuantumRegister(output_partition.len_partition(), name=output_partition.name)
    )
    Qregisters.append(
        QuantumRegister(output_partition.len_partition(), name=output_partition.name)
    )
    return qc


def select_qreg_by_name(qc, name):
    """Function returning the quantum register in QC selected by name"""
    for qr in qc.qregs:
        if name == qr.name:
            break
    return qr


def negation_0(qc, qr, bit_string):
    """Function which insert a NOT gate if the bit in the rule is 0"""
    for index in range(len(bit_string)):
        if bit_string[index] == "0":
            qc.x(qr[index])


def convert_rule(qc, fuzzy_rule, partitions, output_partition):
    """Function which convert a fuzzy rule in the equivalent quantum circuit.
    You can use multiple times convert_rule to concatenate the quantum circuits related to different
    rules."""
    all_partition = partitions.copy()
    all_partition.append(output_partition)
    # print(output_partition)
    # print(partitions)
    # print(all_partition)
    rule = fp.fuzzy_rules().add_rules(fuzzy_rule, all_partition)
    controls = []
    targs = []
    # print(fuzzy_rule)
    # print(rule)
    for index in range(len(rule)):
        if rule[index] == "and" or rule[index] == "then":
            qr = select_qreg_by_name(qc, rule[index - 2])
            negation_0(qc, qr, rule[index - 1])
            # qc.x(qr[-1])
            for i in range(select_qreg_by_name(qc, rule[index - 2]).size - 1):
                # print(select_qreg_by_name(qc, rule[index-2])[i])
                controls.append(select_qreg_by_name(qc, rule[index - 2])[i])
            controls.append(qr[-1])
        if rule[index] == "then":
            # print(rule[index])
            # print(rule[index+2])
            # print('converted', int(rule[index+2],2))
            targs.append(
                select_qreg_by_name(qc, output_partition)[int(rule[index + 2][::-1], 2)]
            )
            # print(targs)

    # print(controls, targs)
    # scratch = select_qreg_by_name(qc, 'scratch')
    qc.mcx(controls, targs[0])
    for index in range(len(rule)):
        if rule[index] == "and" or rule[index] == "then":
            qr = select_qreg_by_name(qc, rule[index - 2])
            negation_0(qc, qr, rule[index - 1])
            # qc.x(qr[-1])


