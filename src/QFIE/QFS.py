from qiskit import (
    QuantumCircuit,
    QuantumRegister,
)
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import math
#from . import fuzzy_partitions as fp
import fuzzy_partitions as fp

Qregisters = []


def generate_circuit(fuzzy_partitions, encoding='logaritmic'):
    """Function generating a quantum circuit with width required by QFS"""
    qc = QuantumCircuit()
    for partition in fuzzy_partitions:
        if encoding=='logaritmic':
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
        if encoding == 'linear':
            qc.add_register(
                QuantumRegister(
                    partition.len_partition(),
                    name=partition.name,
                )
            )
            Qregisters.append(
                QuantumRegister(
                    partition.len_partition(),
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


def output_single_qubit_register(qc, name):
    qc.add_register(QuantumRegister(1, name=name))
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


def merge_subcounts(subcounts, output_partition):
    merged_counts = {}
    full_out_states = []
    state = ["0" for _ in range(len(output_partition.sets))]
    for i in range(len(output_partition.sets)):
        state[-i - 1] = "1"
        key = "".join(bit for bit in state)
        # TODO: reverse key string here in case of Qiskit ordering issues
        full_out_states.append(key)
        merged_counts[key] = 0
        state[-i - 1] = "0"
    for set in output_partition.sets:
        try:
            merged_counts[
                full_out_states[output_partition.sets.index(set)]
            ] = subcounts[set]["1"]
        except KeyError:
            pass
    return merged_counts


def convert_rule(qc, fuzzy_rule, partitions, output_partition, encoding='logaritmic'):
    """Function which convert a fuzzy rule in the equivalent quantum circuit.
    You can use multiple times convert_rule to concatenate the quantum circuits related to different
    rules."""
    all_partition = partitions.copy()
    all_partition.append(output_partition)
    rule = fp.fuzzy_rules().add_rules(fuzzy_rule, all_partition)
    controls = []
    targs = []
    if encoding == 'logaritmic':
        if "not" in rule: 
            raise Exception("Fuzzy Negation only managed with the linear encoding")
        for index in range(len(rule)):
            if rule[index] == "and" or rule[index] == "then":
                qr = select_qreg_by_name(qc, rule[index - 2])
                negation_0(qc, qr, rule[index - 1])
                for i in range(select_qreg_by_name(qc, rule[index - 2]).size):
                    if len(rule[index - 1]) > i:
                        controls.append(select_qreg_by_name(qc, rule[index - 2])[i])
                    else:
                        break
            if rule[index] == "then":
                targs.append(
                    select_qreg_by_name(qc, output_partition)[int(rule[index + 2][::-1], 2)]
                )
        qc.mcx(controls, targs[0])
        for index in range(len(rule)):
            if rule[index] == "and" or rule[index] == "then":
                qr = select_qreg_by_name(qc, rule[index - 2])
                negation_0(qc, qr, rule[index - 1])



    if encoding == 'linear':
        for index in range(len(rule)):
            if rule[index] == "and" or rule[index] == "then":
                if rule[index - 2] != 'not':
                    controls.append(select_qreg_by_name(qc, rule[index - 2])[rule[index-1][::-1].index('1')])
                    to_negate = False
                else: 
                    to_negate = select_qreg_by_name(qc, rule[index - 3])[rule[index-1][::-1].index('1')]
                    qc.x(select_qreg_by_name(qc, rule[index - 3])[rule[index-1][::-1].index('1')])
                    controls.append(select_qreg_by_name(qc, rule[index - 3])[rule[index-1][::-1].index('1')])
            if rule[index] == "then":
                targs.append(
                    select_qreg_by_name(qc, output_partition)[int(rule[index + 2][::-1], 2)]
                )
        qc.mcx(controls, targs[0])
        if to_negate != False: qc.x(to_negate) 

    
    



def compute_qc(backend, qc,  qc_label, n_shots, verbose=True,  transpilation_info=False, optimization_level=3):
    f""" Function computing the quantum circuit qc named qc_label on a backend
     
     Args:
          backend: quantum backend to run the quantum circuit.
          qc (QuantumCircuit): quantum circuit to execute;
          qc_label (str): quantum circuit label;
          n_shots (int): Number of shots;
          verbose (Bool): True to see detail of execution;
          transpilation_info (Bool): True to get information about transpiled qc.
            
     Return:
         A dictionary with qc_label as key and counts as value.
            
          """
    if verbose:
        try:
            backend_name = backend.backend_name
        except:
            try: backend_name = backend.DEFAULT_CONFIGURATION['backend_name']
            except: backend_name = 'AerSimulator'
        print('Running qc ' + qc_label + ' on ' + backend_name)
        

    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    transpiled_qc = pm.run(qc)
    if transpilation_info:
        print(
            "transpiled depth qc " + str(qc_label), transpiled_qc.depth()
        )
        print(
            "Operations " + str(qc_label),
            transpiled_qc.count_ops(),
        )
    print(transpiled_qc)
    sampler = SamplerV2(backend)
    job = sampler.run([transpiled_qc], shots=n_shots)
    job_result = job.result()
    
    values = job_result[0].data._data.values()
    counts = list(values)[0].get_counts()


    return {qc_label:counts}


