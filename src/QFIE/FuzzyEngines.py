""" This module implements the base class for setting up the quantum fuzzy inference engine proposed in doi: 10.1109/TFUZZ.2022.3202348. """
import numpy as np
import skfuzzy as fuzz
import math
import warnings
from copy import deepcopy
from pathlib import Path
from qiskit import (
    ClassicalRegister,
    QuantumRegister,
)
try:
    from qiskit_aer import AerSimulator
except ImportError:
    AerSimulator = None
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit import transpile
from itertools import cycle, islice, repeat, product as cartesian_product
from concurrent.futures import ThreadPoolExecutor
import time
from sympy import false, symbols, true
from sympy.logic.boolalg import And, Not, Or, SOPform


from . import fuzzy_partitions as fp
from . import QFS as QFS
#import fuzzy_partitions as fp
#import QFS as QFS


def _prepare_draw_filename(filename, label=None):
    path = Path(filename).expanduser()
    if label is not None:
        path = path.with_name(f"{label}_{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _bitstring_to_minterm(bitstring):
    return int(bitstring, 2) if bitstring else 0


def _padded_code(code, qreg_size):
    return code + ("0" * (qreg_size - len(code)))


def _partition_valid_codes(partition, qreg_size):
    return [
        _padded_code(code, qreg_size)
        for code in partition.associate_quantum_states().values()
    ]


def _all_bitstrings(n_bits):
    return [
        "".join(bits)
        for bits in cartesian_product("01", repeat=n_bits)
    ]


def _gray_code(index):
    return index ^ (index >> 1)


def _output_register_size(output_partition, output_encoding):
    if output_encoding == "gray":
        return max(1, math.ceil(math.log(output_partition.len_partition(), 2)))
    return output_partition.len_partition()


def _output_code_for_index(output_index, output_partition, output_encoding):
    if output_encoding == "gray":
        register_size = _output_register_size(output_partition, output_encoding)
        binary_format = "{0:0" + str(register_size) + "b}"
        return binary_format.format(_gray_code(output_index))

    bits = ["0" for _ in range(output_partition.len_partition())]
    bits[output_index] = "1"
    return "".join(bits)


def _output_codes(output_partition, output_encoding):
    return [
        _output_code_for_index(output_index, output_partition, output_encoding)
        for output_index in range(output_partition.len_partition())
    ]


def _collect_valid_and_invalid_input_codes(partitions, qreg_sizes):
    local_valid_codes = [
        _partition_valid_codes(partition, qreg_sizes[partition.name])
        for partition in partitions
    ]
    local_all_codes = [
        _all_bitstrings(qreg_sizes[partition.name])
        for partition in partitions
    ]

    valid_inputs = {
        "".join(codes)
        for codes in cartesian_product(*local_valid_codes)
    }
    all_inputs = {
        "".join(codes)
        for codes in cartesian_product(*local_all_codes)
    }
    invalid_inputs = all_inputs - valid_inputs
    return sorted(valid_inputs), sorted(invalid_inputs), sorted(all_inputs)


def _rule_product_and_output_index(rule, input_partitions, output_partition, encoding):
    all_partitions = input_partitions.copy()
    all_partitions.append(output_partition)
    converted_rule = fp.fuzzy_rules().add_rules(rule, all_partitions)
    original_rule = list(filter(("is").__ne__, rule.split()))

    product = []
    output_index = None

    for index, token in enumerate(converted_rule):
        if token != "and" and token != "then":
            continue

        if encoding == "linear" and converted_rule[index - 2] == "not":
            var_name = converted_rule[index - 3]
            code = converted_rule[index - 1]
            qubit_index = code[::-1].index("1")
            product.append((f"{var_name}_{qubit_index}", 0))
        else:
            var_name = converted_rule[index - 2]
            code = converted_rule[index - 1]
            if encoding == "linear":
                qubit_index = code[::-1].index("1")
                product.append((f"{var_name}_{qubit_index}", 1))
            else:
                for qubit_index, bit_value in enumerate(code):
                    product.append((f"{var_name}_{qubit_index}", int(bit_value)))

        if token == "then":
            output_index = output_partition.sets.index(original_rule[index + 2])

    return product, output_index


def _product_matches(product, assignment):
    return all(assignment[var_name] == bit_value for var_name, bit_value in product)


def _build_rule_truth_table(
    rules,
    input_partitions,
    output_partition,
    output_encoding,
    encoding,
    valid_inputs,
    invalid_inputs,
    variable_order,
):
    parsed_rules = [
        _rule_product_and_output_index(rule, input_partitions, output_partition, encoding)
        for rule in rules
    ]
    output_codes = _output_codes(output_partition, output_encoding)
    n_output_bits = len(output_codes[0])
    ones_by_output = [[] for _ in range(n_output_bits)]
    dontcares_by_output = [
        [_bitstring_to_minterm(bitstring) for bitstring in invalid_inputs]
        for _ in range(n_output_bits)
    ]

    for bitstring in valid_inputs:
        assignment = {
            variable_order[i]: int(bitstring[i])
            for i in range(len(variable_order))
        }
        matched_outputs = set()
        for product, output_index in parsed_rules:
            if _product_matches(product, assignment):
                matched_outputs.add(output_index)

        for output_index in matched_outputs:
            output_code = output_codes[output_index]
            for bit_index, bit_value in enumerate(output_code):
                if bit_value == "1":
                    ones_by_output[bit_index].append(_bitstring_to_minterm(bitstring))

    return ones_by_output, dontcares_by_output, parsed_rules


def _minimize_output_bit_sop(variables, ones, dontcares):
    return SOPform(variables, ones, dontcares)


def _sympy_expr_to_products(expr):
    if expr == false:
        return []
    if expr == true:
        return [[]]

    terms = expr.args if isinstance(expr, Or) else (expr,)
    products = []
    for term in terms:
        factors = term.args if isinstance(term, And) else (term,)
        product = []
        for factor in factors:
            if isinstance(factor, Not):
                product.append((str(factor.args[0]), 0))
            else:
                product.append((str(factor), 1))
        products.append(product)
    return products


def _product_to_string(product):
    if len(product) == 0:
        return "1"
    return " & ".join(
        var_name if bit_value == 1 else f"~{var_name}"
        for var_name, bit_value in product
    )


def _products_to_sop_string(products):
    if len(products) == 0:
        return "0"
    terms = []
    for product in products:
        term = _product_to_string(product)
        if len(product) > 1:
            term = f"({term})"
        terms.append(term)
    return " | ".join(terms)


def _original_products_by_output(parsed_rules, output_partition, output_encoding):
    output_codes = _output_codes(output_partition, output_encoding)
    products_by_output = [[] for _ in range(len(output_codes[0]))]
    for product, output_index in parsed_rules:
        for bit_index, bit_value in enumerate(output_codes[output_index]):
            if bit_value == "1":
                products_by_output[bit_index].append(product)
    return products_by_output


def _print_fuzzy_set_basis_mapping(input_partitions, output_partition, output_encoding):
    print("Fuzzy-set basis-state mapping")
    print("  Input registers use bitstrings in register order q[0]..q[n-1].")
    for partition in input_partitions:
        print(f"  Input {partition.name}:")
        for set_name, bitstring in partition.associate_quantum_states().items():
            qubit_values = ", ".join(
                f"{partition.name}_{index}={bit_value}"
                for index, bit_value in enumerate(bitstring)
            )
            print(f"    {set_name}: |{bitstring}> ({qubit_values})")

    if output_encoding == "gray":
        print(f"  Output {output_partition.name} uses Gray-encoded register bits q[0]..q[n-1]:")
    else:
        print(f"  Output {output_partition.name} uses one-hot register bits q[0]..q[n-1]:")

    for output_index, set_name in enumerate(output_partition.sets):
        bitstring = _output_code_for_index(
            output_index,
            output_partition,
            output_encoding,
        )
        print(
            f"    {set_name}: |{bitstring}> "
            f"({', '.join(f'{output_partition.name}_{i}={bit}' for i, bit in enumerate(bitstring))})"
        )


def _print_boolean_optimization_report(
    input_partitions,
    output_partition,
    output_encoding,
    optimization_data,
    output_indices=None,
    ancilla=False,
):
    if output_indices is None:
        output_indices = range(len(optimization_data["products_by_output"]))

    original_products = _original_products_by_output(
        optimization_data["parsed_rules"],
        output_partition,
        output_encoding,
    )

    _print_fuzzy_set_basis_mapping(input_partitions, output_partition, output_encoding)
    print("Boolean minimization report")
    for output_index in output_indices:
        if output_encoding == "gray":
            output_label = f"{output_partition.name}_bit_{output_index}"
        else:
            output_label = output_partition.sets[output_index]
        optimized_products = optimization_data["products_by_output"][output_index]
        products_are_disjoint = _products_are_disjoint_on_valid_inputs(
            optimized_products,
            optimization_data["valid_inputs"],
            optimization_data["variable_order"],
        )
        if products_are_disjoint:
            synthesis = "optimized SOP synthesized directly with MCX gates"
        elif ancilla:
            synthesis = "optimized SOP synthesized as OR-of-products with ancillas"
        else:
            synthesis = "optimized SOP products overlap; rule-by-rule fallback is used"

        print(f"Output {output_partition.name}[{output_index}] ({output_label})")
        print(f"  Original SOP:  {_products_to_sop_string(original_products[output_index])}")
        print(f"  Optimized SOP: {_products_to_sop_string(optimized_products)}")
        print(f"  Synthesis:     {synthesis}")


def _apply_product_as_mcx(qc, product, target, var_to_qubit):
    if len(product) == 0:
        qc.x(target)
        return

    controls = [var_to_qubit[var_name] for var_name, _ in product]
    negative_controls = [
        var_to_qubit[var_name]
        for var_name, bit_value in product
        if bit_value == 0
    ]

    for qubit in negative_controls:
        qc.x(qubit)

    if len(controls) == 1:
        qc.cx(controls[0], target)
    elif len(controls) == 2:
        qc.ccx(controls[0], controls[1], target)
    else:
        qc.mcx(controls, target)

    for qubit in reversed(negative_controls):
        qc.x(qubit)


def _apply_product_to_ancilla(qc, product, ancilla, var_to_qubit):
    _apply_product_as_mcx(qc, product, ancilla, var_to_qubit)


def _products_are_disjoint_on_valid_inputs(products, valid_inputs, variable_order):
    for bitstring in valid_inputs:
        assignment = {
            variable_order[i]: int(bitstring[i])
            for i in range(len(variable_order))
        }

        active_count = 0
        for product in products:
            if _product_matches(product, assignment):
                active_count += 1

        if active_count > 1:
            return False

    return True


def _apply_or_many_to_target(qc, term_qubits, target):
    if len(term_qubits) == 0:
        return

    if len(term_qubits) == 1:
        qc.cx(term_qubits[0], target)
        return

    for qubit in term_qubits:
        qc.x(qubit)

    if len(term_qubits) == 2:
        qc.ccx(term_qubits[0], term_qubits[1], target)
    else:
        qc.mcx(term_qubits, target)

    qc.x(target)

    for qubit in reversed(term_qubits):
        qc.x(qubit)


def _synthesize_sop_to_target_with_ancillas(
    qc,
    products,
    target,
    var_to_qubit,
    term_ancillas,
):
    if len(products) == 0:
        return

    if len(products) == 1:
        _apply_product_as_mcx(qc, products[0], target, var_to_qubit)
        return

    if len(term_ancillas) < len(products):
        raise ValueError("Not enough ancillas for SOP synthesis.")

    used_ancillas = term_ancillas[:len(products)]

    for product, anc in zip(products, used_ancillas):
        _apply_product_to_ancilla(qc, product, anc, var_to_qubit)

    _apply_or_many_to_target(qc, used_ancillas, target)

    for product, anc in reversed(list(zip(products, used_ancillas))):
        _apply_product_to_ancilla(qc, product, anc, var_to_qubit)


def _build_optimization_data(qc, rules, input_partitions, output_partition, output_encoding, encoding):
    qreg_sizes = {
        partition.name: QFS.select_qreg_by_name(qc, partition.name).size
        for partition in input_partitions
    }
    variable_order = []
    var_to_qubit = {}
    for partition in input_partitions:
        qreg = QFS.select_qreg_by_name(qc, partition.name)
        for qubit_index in range(qreg.size):
            var_name = f"{partition.name}_{qubit_index}"
            variable_order.append(var_name)
            var_to_qubit[var_name] = qreg[qubit_index]

    valid_inputs, invalid_inputs, _ = _collect_valid_and_invalid_input_codes(
        input_partitions,
        qreg_sizes,
    )
    ones_by_output, dontcares_by_output, parsed_rules = _build_rule_truth_table(
        rules,
        input_partitions,
        output_partition,
        output_encoding,
        encoding,
        valid_inputs,
        invalid_inputs,
        variable_order,
    )
    sympy_variables = symbols(" ".join(variable_order))
    if len(variable_order) == 1:
        sympy_variables = (sympy_variables,)
    products_by_output = [
        _sympy_expr_to_products(
            _minimize_output_bit_sop(sympy_variables, ones, dontcares)
        )
        for ones, dontcares in zip(ones_by_output, dontcares_by_output)
    ]

    return {
        "products_by_output": products_by_output,
        "parsed_rules": parsed_rules,
        "valid_inputs": valid_inputs,
        "variable_order": variable_order,
        "var_to_qubit": var_to_qubit,
    }


def _rules_for_output_index(parsed_rules, rules, output_index):
    return [
        rule
        for rule, (_, rule_output_index) in zip(rules, parsed_rules)
        if rule_output_index == output_index
    ]


def _rules_for_output_bit(parsed_rules, rules, output_bit_index, output_partition, output_encoding):
    return [
        rule
        for rule, (_, rule_output_index) in zip(rules, parsed_rules)
        if _output_code_for_index(rule_output_index, output_partition, output_encoding)[output_bit_index] == "1"
    ]


def _apply_rule_with_encoded_output(
    qc,
    rule,
    input_partitions,
    output_partition,
    output_encoding,
    encoding,
    var_to_qubit,
    output_qreg,
):
    product, output_index = _rule_product_and_output_index(
        rule,
        input_partitions,
        output_partition,
        encoding,
    )
    output_code = _output_code_for_index(output_index, output_partition, output_encoding)
    applied_gate = False
    for output_bit_index, bit_value in enumerate(output_code):
        if bit_value == "1":
            _apply_product_as_mcx(
                qc,
                product,
                output_qreg[output_bit_index],
                var_to_qubit,
            )
            applied_gate = True
    return applied_gate


def _var_to_qubit_for_inputs(qc, input_partitions):
    var_to_qubit = {}
    for partition in input_partitions:
        qreg = QFS.select_qreg_by_name(qc, partition.name)
        for qubit_index in range(qreg.size):
            var_to_qubit[f"{partition.name}_{qubit_index}"] = qreg[qubit_index]
    return var_to_qubit


class QuantumFuzzyEngine:
    """

    Class implementing the Quantum Fuzzy Inference Engine proposed in:

    G. Acampora, R. Schiattarella and A. Vitiello, "On the Implementation of Fuzzy Inference Engines on Quantum Computers,"
    in IEEE Transactions on Fuzzy Systems, 2022, doi: 10.1109/TFUZZ.2022.3202348.


    """

    def __init__(self, verbose=True, encoding='logaritmic'):
        self.input_ranges = {}
        self.output_range = {}
        self.input_fuzzysets = {}
        self.output_fuzzyset = {}
        self.input_partitions = {}
        self.output_partition = {}
        self.variables = {}
        self.rules = []
        self.rule_subsets = {}
        self.qc = {}
        self.verbose = verbose
        self.transpile_info = verbose
        self.encoding = encoding

    def input_variable(self, name, range):
        """Define the input variable "name" of the system.
        
        Args:
             name (str): Name of the variable as string.
             range (np array): Universe of the discourse for the input variable.
        
        Returns:
            None
        """
        if name in list(self.input_ranges.keys()):
            raise Exception("Variable name must be unambiguos")
        else:
            self.input_ranges[name] = range
            self.input_fuzzysets[name] = []
            self.input_partitions[name] = ""

    def output_variable(self, name, range):
        """Define the output variable "name" of the system.
        
        Args:
             name (str): Name of the variable as string.
             range (np array): Universe of the discourse for the output variable.
        
        Returns:
            None
        """
        self.output_range[name] = range
        self.output_fuzzyset[name] = []
        self.output_partition[name] = ""

    def add_input_fuzzysets(self, var_name, set_names, sets):
        """Set the partition for the input fuzzy variable 'var_name'.
        
        Args:
             var_name (str): name of the fuzzy variable defined with input_variable method previously.
             set_names (list): list of fuzzy sets' name as str.
             sets (list): list of scikit-fuzzy membership function objects.
        
        Returns:
            None
        """
        for set in sets:
            self.input_fuzzysets[var_name].append(set)
        is_exact_partition = np.allclose(np.sum(sets, axis=0), 1.0)
        self.input_partitions[var_name] = fp.fuzzy_partition(
            var_name,
            set_names,
            encoding=self.encoding,
            minimize_hamming=self.encoding == 'logaritmic',
            exact_partition=is_exact_partition,
        )

    def add_output_fuzzysets(self, var_name, set_names, sets):
        """Set the partition for the output fuzzy variable 'var_name'.
        
        Args:
             var_name (str): name of the fuzzy variable defined with output_variable method previously.
             set_names (list): list of fuzzy sets' name as str.
             sets (list): list of scikit-fuzzy membership function objects.
        Returns:
            None
        """
        for set in sets:
            self.output_fuzzyset[var_name].append(set)
        self.output_partition[var_name] = fp.fuzzy_partition(var_name, set_names)

    def set_rules(self, rules):
        """Set the rule-base of the system. \n
        Rules must be formatted as follows: 'if var_1 is x_i and var_2 is x_k and  and var_n is x_l then out_1 is y_k'
        
        Args:
             rules (list): list of rules as strings.
        
        Returns:
            None
        """
         
        self.rules = rules


    def filter_rules(self, rules, output_term):
        """Searches the rule list and picks only the rules corresponding to the same output value (y_k at fixed k). \n
        Rules must be formatted as follows: 'if var_1 is x_i and var_2 is x_k and  and var_n is x_l then out_1 is y_k'

        Args:
            rules (list): list of rules as strings.
            output_term (str): single output term y_k at fixed k as string.
        Returns:
            Filtered rules as a new list.
        """
        rules_subset = []
        for rule in rules:
            if f"then {list(self.output_fuzzyset.keys())[0]} is {output_term}" in rule:
                rules_subset.append(rule)
        return rules_subset

    def truncate(self, n, decimals=0):
        multiplier = 10**decimals
        return math.floor(n * multiplier + 0.5) / multiplier

    def counts_evaluator(self, n_qubits, counts):
        """Function returning the alpha values for alpha-cutting the output fuzzy sets according to the
        probability of measuring the related basis states on the output quantum register.
        
        Args:
             n_qubits (int): number of qubits in the output quantum register.
             counts (dict): counting dictionary of the output quantum register measurement.
        
        Returns:
            alpha values for alpha-cutting the output fuzzy sets as 'dict'.
        """

        output = {}
        n_shots = sum(list(counts.values()))
        counts = {k: v / n_shots for k, v in counts.items()}
        for i in range(n_qubits):
            state = [0 * k for k in range(n_qubits)]
            n = i + 1
            state[-n] = 1
            stringb = ""
            for b in state:
                stringb = str(b) + stringb
            output[stringb] = 0
        counts_keys = list(counts.keys())
        for key in counts_keys:
            if key in list(output.keys()):
                output[key] = counts[key] + output[key]
            else:
                sum_1s = 0
                for bit in key:
                    if bit == "1":
                        sum_1s = sum_1s + 1
                for num_bit in range(n_qubits):
                    if key[num_bit] == "1":
                        for selected_state in list(output.keys()):
                            if selected_state[num_bit] == "1":
                                output[selected_state] = output[selected_state] + (
                                    counts[key] / sum_1s
                                )

        return output

    def build_inference_qc(
        self,
        input_values,
        distributed=False,
        draw_qc=False,
        optimize=False,
        ancilla=False,
        verbose=False,
        output_encoding="one-hot",
        **kwargs
    ):
        """This function builds the quantum circuit implementing the QFIE, initializing the input quantum registers
        according to the 'input_value' argument.
        
        Args:
             input_values (dict): dictionary containing the crisp input values of the system.
                E.g. {'var_name_1' (str): x_1 (float), , 'var_name_n' (str): x_n (float)}
             draw_qc (Bool - default:False): True for drawing the quantum circuit built. False otherwise.
             distributed (Boolean): True to implement the distributed version of the quantum oracle. False otherwise.
             optimize (Boolean): True to minimize the Boolean function induced by the fuzzy rule base before
                synthesizing each one-hot output bit. Invalid unused encodings are treated as don't-cares, while
                valid inputs not covered by any rule preserve the current no-flip default behavior. Optimized
                circuits preserve the original oracle behavior on valid input encodings.
             ancilla (Boolean): Only used when optimize=True. True allows overlapping minimized SOP products
                to be synthesized as a true OR-of-products network with temporary ancillas, which are uncomputed
                to |0>. Disjoint products are still synthesized directly with MCX gates. With optimize=True and
                ancilla=False, overlapping products fall back to the original rule-by-rule construction and emit
                a RuntimeWarning.
             verbose (Boolean): Only used when optimize=True. True prints, for each output bit, the original
                SOP induced by the rules, the minimized SOP, and the synthesis strategy selected.
             output_encoding (str): 'one-hot' keeps the legacy one-hot output register with one target qubit per
                output fuzzy set. 'gray' uses a compressed Gray-encoded output register and builds one Boolean
                function for each output-code bit.
            :keyword filename (str): file path to save image to.
        
        Returns:
            None
        """
        if output_encoding not in ("one-hot", "gray"):
            raise ValueError("output_encoding must be 'one-hot' or 'gray'.")
        if distributed and output_encoding != "one-hot":
            raise NotImplementedError(
                "Distributed QFIE is supported only with output_encoding='one-hot'."
            )

        self.distributed = distributed
        self.output_encoding = output_encoding
        self.transpile_info = verbose

        # Print Crisp Inputs
        if self.verbose:
            print(input_values)

        # FUZZIFICATION
        fuzzyfied_values = {}
        for var_name in list(input_values.keys()):
            fuzzyfied_values[var_name] = [
                fuzz.interp_membership(
                    self.input_ranges[var_name], i, input_values[var_name]
                )
                for i in self.input_fuzzysets[var_name]
            ]
            total_membership = sum(fuzzyfied_values[var_name])
            if total_membership > 1.0 + 1e-9:
                raise Exception(
                    f"Sum of memberships must be less than or equal to 1: "
                    f"variable '{var_name}' at {input_values[var_name]} sums to {total_membership}."
                )
        if self.verbose:
            print("Input values ", fuzzyfied_values)

        # CIRCUIT SETUPS
        # Not Distributed QFIE
        if not distributed:
            self.qc["full_circuit"] = QFS.generate_circuit(
                list(self.input_partitions.values()), encoding = self.encoding
            )
            self.qc["full_circuit"] = QFS.output_register(
                self.qc["full_circuit"],
                list(self.output_partition.values())[0],
                output_encoding=output_encoding,
            )
        # Distributed QFIE
        else:
            self.out_register_name = []
            # Use output linguistic terms as labels (keys) to identify the corresponding distributed circuits
            qc_labels = self.output_partition[list(self.output_fuzzyset.keys())[0]].sets
            for label in qc_labels:
                # Create a quantum circuit corresponding to each label
                self.qc[label] = QFS.generate_circuit(
                    list(self.input_partitions.values()), encoding = self.encoding
                )
                self.qc[label] = QFS.output_single_qubit_register(self.qc[label], label)
                # Create a subset of rules corresponding to each label
                self.rule_subsets[label] = self.filter_rules(self.rules, label)

        # COMPUTING AMPLITUDES FROM FUZZIFIED VALUES
        initial_state = {}
        for var_name in list(input_values.keys()):
            if self.encoding == 'logaritmic':
                required_len = QFS.select_qreg_by_name(
                    list(self.qc.values())[0], var_name
                ).size
                initial_state[var_name] = [0 for _ in range(2**required_len)]
                used_indexes = set()
                quantum_states = self.input_partitions[var_name].associate_quantum_states()
                set_names = self.input_partitions[var_name].sets
                for set_index, set_name in enumerate(set_names):
                    bitstring = _padded_code(
                        quantum_states[set_name],
                        required_len,
                    )
                    basis_index = int(bitstring[::-1], 2)
                    used_indexes.add(basis_index)
                    initial_state[var_name][basis_index] = math.sqrt(
                        fuzzyfied_values[var_name][set_index]
                    )

                default_indexes = [
                    index
                    for index in range(2**required_len)
                    if index not in used_indexes
                ]
                if default_indexes:
                    initial_state[var_name][default_indexes[0]] = math.sqrt(
                        1 - sum(fuzzyfied_values[var_name])
                    )
                for circ in list(self.qc.values()):
                    circ.initialize(
                        initial_state[var_name], QFS.select_qreg_by_name(circ, var_name)
                    )

            if self.encoding == 'linear':

                def linear_encoding(fuzzified_values):
                    #print(sum(fuzzified_values).__round__(6))
                    input_list = [math.sqrt(i) for i in fuzzified_values]
                    n = len(input_list)  # Number of input elements
                    output_size = 2 ** n  # Size of the output list
                    output_list = [0] * output_size  # Initialize the output list with zeros

                    for i in range(n):
                        # Find the index that corresponds to the binary string with only the i-th bit set to 1
                        index = 1 << i  # This is equivalent to 2**i
                        output_list[index] = input_list[i]  # Substitute the value from the input list
                    
                    return output_list
                
                initial_state[var_name] = linear_encoding(fuzzyfied_values[var_name])
                initial_state[var_name][0] = math.sqrt(1 - sum(fuzzyfied_values[var_name]))
                for circ in list(self.qc.values()):
                    circ.initialize(
                        initial_state[var_name], QFS.select_qreg_by_name(circ, var_name)
                    )
                    #print(Statevector(circ).probabilities_dict())
                #print(self.qc['full_circuit'])
                #print('stop')


        # BUILDING ORACLES
        if not distributed:
            output_partition = list(self.output_partition.values())[0]
            input_partitions = list(self.input_partitions.values())
            if not optimize:
                if output_encoding == "one-hot":
                    for rule in self.rules:
                        QFS.convert_rule(
                            qc=self.qc["full_circuit"],
                            fuzzy_rule=rule,
                            partitions=input_partitions,
                            output_partition=output_partition,
                            encoding=self.encoding
                        )
                        self.qc["full_circuit"].barrier()
                else:
                    output_qreg = QFS.select_qreg_by_name(
                        self.qc["full_circuit"],
                        output_partition.name,
                    )
                    var_to_qubit = _var_to_qubit_for_inputs(
                        self.qc["full_circuit"],
                        input_partitions,
                    )
                    for rule in self.rules:
                        applied_gate = _apply_rule_with_encoded_output(
                            self.qc["full_circuit"],
                            rule,
                            input_partitions,
                            output_partition,
                            output_encoding,
                            self.encoding,
                            var_to_qubit,
                            output_qreg,
                        )
                        if applied_gate:
                            self.qc["full_circuit"].barrier()
            else:
                optimization_data = _build_optimization_data(
                    self.qc["full_circuit"],
                    self.rules,
                    input_partitions,
                    output_partition,
                    output_encoding,
                    self.encoding,
                )
                if verbose:
                    _print_boolean_optimization_report(
                        input_partitions,
                        output_partition,
                        output_encoding,
                        optimization_data,
                        ancilla=ancilla,
                    )
                products_by_output = optimization_data["products_by_output"]
                term_ancillas = []
                if ancilla:
                    max_products = max(
                        [
                            len(products)
                            for products in products_by_output
                            if not _products_are_disjoint_on_valid_inputs(
                                products,
                                optimization_data["valid_inputs"],
                                optimization_data["variable_order"],
                            )
                        ],
                        default=0,
                    )
                    if max_products > 1:
                        anc = QuantumRegister(max_products, "anc")
                        self.qc["full_circuit"].add_register(anc)
                        term_ancillas = list(anc)

                output_qreg = QFS.select_qreg_by_name(
                    self.qc["full_circuit"],
                    output_partition.name,
                )
                for output_index, products in enumerate(products_by_output):
                    applied_gate = False
                    added_barrier = False
                    products_are_disjoint = _products_are_disjoint_on_valid_inputs(
                        products,
                        optimization_data["valid_inputs"],
                        optimization_data["variable_order"],
                    )
                    if products_are_disjoint:
                        for product in products:
                            _apply_product_as_mcx(
                                self.qc["full_circuit"],
                                product,
                                output_qreg[output_index],
                                optimization_data["var_to_qubit"],
                            )
                            applied_gate = True
                    elif ancilla:
                        _synthesize_sop_to_target_with_ancillas(
                            self.qc["full_circuit"],
                            products,
                            output_qreg[output_index],
                            optimization_data["var_to_qubit"],
                            term_ancillas,
                        )
                        applied_gate = len(products) > 0
                    else:
                        warnings.warn(
                            "Optimized SOP products overlap on valid inputs; "
                            "falling back to original rule-by-rule synthesis for this output bit. "
                            "Use ancilla=True to synthesize the minimized SOP as an OR network.",
                            RuntimeWarning,
                        )
                        if output_encoding == "one-hot":
                            fallback_rules = _rules_for_output_index(
                                optimization_data["parsed_rules"],
                                self.rules,
                                output_index,
                            )
                            for rule in fallback_rules:
                                QFS.convert_rule(
                                    qc=self.qc["full_circuit"],
                                    fuzzy_rule=rule,
                                    partitions=input_partitions,
                                    output_partition=output_partition,
                                    encoding=self.encoding
                                )
                                self.qc["full_circuit"].barrier()
                                applied_gate = True
                                added_barrier = True
                        else:
                            fallback_rules = _rules_for_output_bit(
                                optimization_data["parsed_rules"],
                                self.rules,
                                output_index,
                                output_partition,
                                output_encoding,
                            )
                            for rule in fallback_rules:
                                product, _ = _rule_product_and_output_index(
                                    rule,
                                    input_partitions,
                                    output_partition,
                                    self.encoding,
                                )
                                _apply_product_as_mcx(
                                    self.qc["full_circuit"],
                                    product,
                                    output_qreg[output_index],
                                    optimization_data["var_to_qubit"],
                                )
                                applied_gate = True
                            if applied_gate:
                                self.qc["full_circuit"].barrier()
                                added_barrier = True
                    if draw_qc and applied_gate and not added_barrier:
                        self.qc["full_circuit"].barrier()

            self.out_register_name = list(self.output_fuzzyset.keys())[0]
            output_register = QFS.select_qreg_by_name(
                self.qc["full_circuit"],
                self.out_register_name,
            )
            out = ClassicalRegister(output_register.size)
            self.qc["full_circuit"].add_register(out)
            self.qc["full_circuit"].measure(
                output_register,
                out,
            )
            if draw_qc:
                print('draw')
                if "filename" in kwargs:
                    self.qc["full_circuit"].draw(
                        "mpl",
                        filename=_prepare_draw_filename(kwargs["filename"]),
                    )
                else:
                    print('draw1')
                    self.qc["full_circuit"].draw("mpl").show()
        else:
            self.out_register_name = []
            # Use output linguistic terms as labels (keys) to identify the corresponding distributed circuits
            qc_labels = self.output_partition[list(self.output_fuzzyset.keys())[0]].sets
            output_partition = list(self.output_partition.values())[0]
            input_partitions = list(self.input_partitions.values())
            for label in qc_labels:
                modified_output_partition = deepcopy(
                    output_partition
                )
                modified_output_partition.sets = [label]
                if not optimize:
                    for rule in self.rule_subsets[label]:
                        QFS.convert_rule(
                            qc=self.qc[label],
                            fuzzy_rule=rule,
                            partitions=input_partitions,
                            output_partition=modified_output_partition,
                            encoding=self.encoding
                        )
                        self.qc[label].barrier()
                else:
                    optimization_data = _build_optimization_data(
                        self.qc[label],
                        self.rules,
                        input_partitions,
                        output_partition,
                        output_encoding,
                        self.encoding,
                    )
                    label_output_index = output_partition.sets.index(label)
                    if verbose:
                        _print_boolean_optimization_report(
                            input_partitions,
                            output_partition,
                            output_encoding,
                            optimization_data,
                            output_indices=[label_output_index],
                            ancilla=ancilla,
                    )
                    products = optimization_data["products_by_output"][label_output_index]
                    products_are_disjoint = _products_are_disjoint_on_valid_inputs(
                        products,
                        optimization_data["valid_inputs"],
                        optimization_data["variable_order"],
                    )
                    term_ancillas = []
                    if ancilla and not products_are_disjoint and len(products) > 1:
                        anc = QuantumRegister(len(products), "anc")
                        self.qc[label].add_register(anc)
                        term_ancillas = list(anc)

                    output_qreg = QFS.select_qreg_by_name(self.qc[label], label)
                    if products_are_disjoint:
                        for product in products:
                            _apply_product_as_mcx(
                                self.qc[label],
                                product,
                                output_qreg[0],
                                optimization_data["var_to_qubit"],
                            )
                    elif ancilla:
                        _synthesize_sop_to_target_with_ancillas(
                            self.qc[label],
                            products,
                            output_qreg[0],
                            optimization_data["var_to_qubit"],
                            term_ancillas,
                        )
                    else:
                        warnings.warn(
                            "Optimized SOP products overlap on valid inputs; "
                            "falling back to original rule-by-rule synthesis for this output bit. "
                            "Use ancilla=True to synthesize the minimized SOP as an OR network.",
                            RuntimeWarning,
                        )
                        for rule in self.rule_subsets[label]:
                            QFS.convert_rule(
                                qc=self.qc[label],
                                fuzzy_rule=rule,
                                partitions=input_partitions,
                                output_partition=modified_output_partition,
                                encoding=self.encoding
                            )
                            self.qc[label].barrier()
                self.out_register_name.append(
                    list(self.output_fuzzyset.keys())[0] + " " + label
                )
                out = ClassicalRegister(1)
                self.qc[label].add_register(out)
                self.qc[label].measure(
                    QFS.select_qreg_by_name(self.qc[label], self.out_register_name[-1]),
                    out,
                )
                if draw_qc:
                    #self.qc[label].draw("mpl").show()
                    if "filename" in kwargs:
                        self.qc[label].draw(
                            "mpl",
                            filename=_prepare_draw_filename(kwargs["filename"], label),
                        )
                    else:
                        self.qc[label].draw("mpl")

    def execute(self, n_shots: int, plot_histo=False, GPU=False, **kwargs):
        """Run the inference engine.
        
        Args:
             n_shots (int): Number of shots.
             plot_histo (Bool- default False): True for plotting the counts histogram.
             GPU (Bool- default False): True for using GPU for simulation. Use False if backend is a real device.

            :keyword backend: quantum backend to run the quantum circuit. If not specified, qasm simulator is used.
            :keyword transpile_info (bool): True for getting information about transpiled qc.
                If not specified, defaults to the verbose value passed to the latest build_inference_qc call.
            :keyword optimization_level (int - default 3): Select a Value from 1 to 3 to set the optimization level in the transpiling
            :keyword defuzzification (str): name of the Defuzzification algorithm to use. If not specified, 'centroid' is used. 
        Return:
            Crisp output of the system.
        """
        # Selecting the backend
        if "backend" in kwargs:
            backend = kwargs["backend"]
        else:
            if AerSimulator is None:
                raise ImportError(
                    "qiskit_aer is required when execute() is called without a backend. "
                    "Install qiskit-aer or pass a backend explicitly."
                )
            backend = AerSimulator()

        #Checking Transpilation Command
        if "transpile_info" in kwargs:
            transp_info = bool(kwargs["transpile_info"])
        else:
            transp_info = bool(self.transpile_info)

        if "optimization_level" in kwargs and kwargs["optimization_level"] != 3: optimization_level = kwargs["optimization_level"]
        else: optimization_level = 3




        # Creating backend list if QFIE is distributed:
        if self.distributed:
            if type(backend) != list: backends_list=[backend]
            else: backends_list = backend
            backends_list = list(islice(cycle(backends_list), len(list(self.qc.keys()))))

        if GPU:
            try:
                backend.set_options(device="GPU")
            except:
                print(
                    "Not possible use GPU for this quantum backend or your device is not equipped with GPUs"
                )

        # COMPUTE NOT DISTRIBUTED ALGORITHM
        if len(self.qc) == 1:
            if type(backend) == list:
                raise 'Please to run the not distributed quantum circuit specify an unique backend not as list'

            # Execute quantum circuit
            self.counts_ = list(QFS.compute_qc(backend, self.qc["full_circuit"], "full_circuit", n_shots, self.verbose, transpilation_info=transp_info, optimization_level=optimization_level).values())[0]

        # COMPUTE DISTRIBUTED ALGORITHM
        else:
            # Distributed version
            subcounts = {}

            # Execute quantum circuits
            counts_list = list(map(QFS.compute_qc, backends_list,
                                                                list(self.qc.values()), list(self.qc.keys()),
                                                                repeat(n_shots), repeat(self.verbose),
                                                                repeat(transp_info), repeat(optimization_level)))

            for count in counts_list:
                subcounts.update(count)

            self.counts_ = QFS.merge_subcounts(
                subcounts, self.output_partition[list(self.output_fuzzyset.keys())[0]]
            )

        # Plot Counts
        if plot_histo:
            plot_histogram(
                self.counts_, color="midnightblue", figsize=(7, 10)
            ).show()

        output_partition = self.output_partition[list(self.output_fuzzyset.keys())[0]]
        if getattr(self, "output_encoding", "one-hot") == "gray":
            self.n_q = _output_register_size(output_partition, "gray")
            n_shots = sum(list(self.counts_.values()))
            normalized_counts = {
                key: value / n_shots
                for key, value in self.counts_.items()
            }
            output_dict = {
                set_name: _output_code_for_index(
                    output_index,
                    output_partition,
                    "gray",
                )[::-1]
                for output_index, set_name in enumerate(output_partition.sets)
            }
        else:
            self.n_q = len(self.output_fuzzyset[list(self.output_fuzzyset.keys())[0]])
            counts = self.counts_evaluator(n_qubits=self.n_q, counts=self.counts_)
            normalized_counts = counts
            output_dict = {
                i: []
                for i in output_partition.sets
            }

            counter = 0
            for set in list(output_dict.keys()):
                counter = counter + 1
                for i in range(self.n_q):
                    if i == self.n_q - counter:
                        output_dict[set].append("1")
                    else:
                        output_dict[set].append("0")
                output_dict[set] = "".join(output_dict[set])

        memberships = {}
        for state in list(output_dict.values()):
            if state in list(normalized_counts.keys()):
                memberships[state] = normalized_counts[state]
            else:
                memberships[state] = 0

        # DEFUZZIFICATION
        if "defuzzification" in kwargs:
            defuzz = kwargs["defuzzification"]
        else: defuzz = 'centroid'

        norm_memberships = memberships
        if self.verbose:
            print("Output Counts", memberships)
        activation = {}
        set_number = 0
        for set in list(output_dict.keys()):
            activation[set] = np.fmin(
                norm_memberships[output_dict[set]],
                self.output_fuzzyset[list(self.output_fuzzyset.keys())[0]][set_number],
            )
            set_number = set_number + 1

        activation_values = list(activation.values())[::-1]
        aggregated = np.zeros(
            self.output_fuzzyset[list(self.output_fuzzyset.keys())[0]][0].shape
        )
        for i in range(len(activation_values)):
            aggregated = np.fmax(aggregated, activation_values[i])

        return (
            fuzz.defuzz(
                self.output_range[list(self.output_fuzzyset.keys())[0]],
                aggregated,
                defuzz,
            ),
            activation_values,
        )


"""
env_light = np.linspace(120, 220, 200)
changing_rate = np.linspace(-10, 10, 200)
dimmer_control = np.linspace(0, 10, 200)



l_dark = fuzz.trapmf(env_light, [120,120,130,150])
l_medium = fuzz.trapmf(env_light, [130,  150, 190,210])
l_light = fuzz.trapmf(env_light, [190,  210, 220, 220])

r_ns = fuzz.trimf(changing_rate, [-10,-10,0])
r_zero = fuzz.trimf(changing_rate, [-10,0,10])
r_ps = fuzz.trimf(changing_rate, [0,10,10])

dm_vs = fuzz.trapmf(dimmer_control, [0,0,2,4])
dm_s = fuzz.trimf(dimmer_control, [2,4,6])
dm_b = fuzz.trimf(dimmer_control, [4,6,8])
dm_vb = fuzz.trapmf(dimmer_control, [6,8,10,10])

'''rules = ['if env_light is dark and change_rate is pos_small then dimmer_ctrl is big',
         'if env_light is dark and change_rate is zero then dimmer_ctrl is big',
         'if env_light is dark and change_rate is neg_small then dimmer_ctrl is very_big',
         'if env_light is medium and change_rate is pos_small then dimmer_ctrl is small',
         'if env_light is medium and change_rate is zero then dimmer_ctrl is big',
         'if env_light is medium and change_rate is neg_small then dimmer_ctrl is big',
         'if env_light is light and change_rate is pos_small then dimmer_ctrl is very_small',
         'if env_light is light and change_rate is zero then dimmer_ctrl is small',
         'if env_light is light and change_rate is neg_small then dimmer_ctrl is big']'''

rules = ['if env_light is dark and change_rate is not neg_small then dimmer_ctrl is big', 
         'if env_light is dark and change_rate is neg_small then dimmer_ctrl is very_big',
         'if env_light is medium and change_rate is not pos_small then dimmer_ctrl is big',
         'if env_light is medium and change_rate is pos_small then dimmer_ctrl is small',
         'if env_light is light and change_rate is pos_small then dimmer_ctrl is very_small',
         'if env_light is light and change_rate is zero then dimmer_ctrl is small',
         'if env_light is light and change_rate is neg_small then dimmer_ctrl is big']

qfie = QuantumFuzzyEngine(verbose=False, encoding='linear')
qfie.input_variable(name='env_light', range=env_light)
qfie.input_variable(name='change_rate', range=changing_rate)
qfie.output_variable(name='dimmer_ctrl', range=dimmer_control)

qfie.add_input_fuzzysets(var_name='env_light', set_names=['dark', 'medium', 'light'], sets=[l_dark, l_medium, l_light])
qfie.add_input_fuzzysets(var_name='change_rate', set_names=['neg_small', 'zero', 'pos_small'], sets=[r_ns, r_zero, r_ps])
qfie.add_output_fuzzysets(var_name='dimmer_ctrl', set_names=['very_small', 'small', 'big', 'very_big'],sets=[dm_vs, dm_s, dm_b, dm_vb])
qfie.set_rules(rules)
qfie.build_inference_qc({'env_light':170, 'change_rate':0}, encoding='linear', draw_qc=False, distributed=True)
print(qfie.qc['very_big'])
print('end')
"""
