""" This module implements the base class for setting up the quantum fuzzy inference engine proposed in doi: 10.1109/TFUZZ.2022.3202348. """
import numpy as np
import skfuzzy as fuzz
import math
from qiskit import (
    ClassicalRegister,
    execute,
    BasicAer,
)
from qiskit.visualization import plot_histogram
from . import fuzzy_partitions as fp
from . import QFS as QFS


class QuantumFuzzyEngine:
    """

    Class implementing the Quantum Fuzzy Inference Engine proposed in:

    G. Acampora, R. Schiattarella and A. Vitiello, "On the Implementation of Fuzzy Inference Engines on Quantum Computers,"
    in IEEE Transactions on Fuzzy Systems, 2022, doi: 10.1109/TFUZZ.2022.3202348.


    """

    def __init__(self, verbose = True):
        self.input_ranges = {}
        self.output_range = {}
        self.input_fuzzysets = {}
        self.output_fuzzyset = {}
        self.input_partitions = {}
        self.output_partition = {}
        self.variables = {}
        self.rules = []
        self.qc = ""
        self.verbose = verbose

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
        self.input_partitions[var_name] = fp.fuzzy_partition(var_name, set_names)

    def add_output_fuzzysets(self, var_name, set_names, sets):
        """
        Set the partition for the output fuzzy variable 'var_name'.

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
        Rules must be formatted as follows: 'if var_1 is x_i and var_2 is x_k and ... and var_n is x_l then out_1 is y_k'

        Args:
            rules (list): list of rules as strings.
        Returns:
            None
        """
        self.rules = rules

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

    def build_inference_qc(self, input_values, draw_qc=False):
        """ This function builds the quantum circuit implementing the QFIE, initializing the input quantum registers
        according to the 'input_value' argument.

        Args:
            input_values (dict): dictionary containing the crisp input values of the system.
                E.g. {'var_name_1' (str): x_1 (float), ..., 'var_name_n' (str): x_n (float)}

            draw_qc (Boolean): True for drawing the quantum circuit built. False otherwise.
        Returns:
            None
        """
        self.qc = QFS.generate_circuit(list(self.input_partitions.values()))
        self.qc = QFS.output_register(self.qc, list(self.output_partition.values())[0])
        if self.verbose:
            print(input_values)
        fuzzyfied_values = {}
        norm_values = {}
        for var_name in list(input_values.keys()):
            fuzzyfied_values[var_name] = [
                fuzz.interp_membership(
                    self.input_ranges[var_name], i, input_values[var_name]
                )
                for i in self.input_fuzzysets[var_name]
            ]
            # norm_values[var_name] = [self.truncate(float(i)/sum(fuzzyfied_values[var_name]), 3) for i in fuzzyfied_values[var_name]]
        if self.verbose:
            print("Input values ", fuzzyfied_values)
        initial_state = {}
        for var_name in list(input_values.keys()):
            initial_state[var_name] = [
                math.sqrt(fuzzyfied_values[var_name][i])
                for i in range(len(fuzzyfied_values[var_name]))
            ]
            required_len = QFS.select_qreg_by_name(self.qc, var_name).size
            while len(initial_state[var_name]) != 2**required_len:
                initial_state[var_name].append(0)
            initial_state[var_name][-1] = math.sqrt(1 - sum(fuzzyfied_values[var_name]))
            # print(initial_state)
            self.qc.initialize(
                initial_state[var_name], QFS.select_qreg_by_name(self.qc, var_name)
            )

        for rule in self.rules:
            QFS.convert_rule(
                qc=self.qc,
                fuzzy_rule=rule,
                partitions=list(self.input_partitions.values()),
                output_partition=list(self.output_partition.values())[0],
            )
            self.qc.barrier()

        self.out_register_name = list(self.output_fuzzyset.keys())[0]
        out = ClassicalRegister(len(self.output_fuzzyset[self.out_register_name]))
        self.qc.add_register(out)
        self.qc.measure(QFS.select_qreg_by_name(self.qc, self.out_register_name), out)
        if draw_qc:
            self.qc.draw("mpl").show()

    def execute(self, backend_name, n_shots, provider=None, plot_histo=False, GPU = False):
        """ Run the inference engine.

        Args:
            backend_name (str): IBMQ backend to use for computing.\n

                - Use "qasm_simulator" to simulate the run.\n

                - For real devices an IBMQ provider is required.

            n_shots (int): Number of shots.

            provider (str): IBMQ Provider.\n

                - Default 'None' to use with 'qasm_simulator' backend

            plot_histo (Boolean): True for plotting the counts histogram. False Otherwise.

            GPU (Boolean): True for using GPU for simulation. Use False if backend is a real device.



        Return:
            Crisp output of the system.
        """
        if backend_name == "qasm_simulator":
            backend = BasicAer.get_backend(backend_name)
        else:
            backend = provider.get_backend(backend_name)

        if GPU:
            backend.set_options(device='GPU')

        job = execute(self.qc, backend, shots=n_shots)
        result = job.result()
        if plot_histo:
            plot_histogram(
                job.result().get_counts(), color="midnightblue", figsize=(7, 10)
            ).show()
        self.counts_ = job.result().get_counts()
        self.n_q = len(self.output_fuzzyset[self.out_register_name])
        counts = self.counts_evaluator(n_qubits=self.n_q, counts=self.counts_)
        # normalized_counts = {k: v / total for total in (sum(counts.values()),) for k, v in counts.items()}
        normalized_counts = counts
        output_dict = {
            i: [] for i in self.output_partition[self.out_register_name].sets
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

        norm_memberships = memberships
        if self.verbose:
            print("Output Counts", memberships)
        activation = {}
        set_number = 0
        for set in list(output_dict.keys()):
            activation[set] = np.fmin(
                norm_memberships[output_dict[set]],
                self.output_fuzzyset[self.out_register_name][set_number],
            )
            set_number = set_number + 1

        activation_values = list(activation.values())[::-1]
        aggregated = np.zeros(self.output_fuzzyset[self.out_register_name][0].shape)
        for i in range(len(activation_values)):
            aggregated = np.fmax(aggregated, activation_values[i])

        return (
            fuzz.defuzz(
                self.output_range[self.out_register_name], aggregated, "centroid"
            ),
            activation_values,
        )
