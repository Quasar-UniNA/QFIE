""" This module implements the base class for setting up the quantum fuzzy inference engine proposed in doi: 10.1109/TFUZZ.2022.3202348. """
import numpy as np
import skfuzzy as fuzz
import math
from copy import deepcopy
from qiskit import (
    ClassicalRegister,
)
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit import transpile
from itertools import cycle, islice, repeat
from concurrent.futures import ThreadPoolExecutor
import time


#from . import fuzzy_partitions as fp
#from . import QFS as QFS
import fuzzy_partitions as fp
import QFS as QFS


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
        self.input_partitions[var_name] = fp.fuzzy_partition(var_name, set_names, encoding=self.encoding)

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

    def fuzzyfication(self, input_values):
        fuzzyfied_values = {}
        for var_name in list(input_values.keys()):
            fuzzyfied_values[var_name] = [
                fuzz.interp_membership(
                    self.input_ranges[var_name], i, input_values[var_name]
                )
                for i in self.input_fuzzysets[var_name]
            ]
        if self.verbose:
            print("Input values ", fuzzyfied_values)
        return fuzzyfied_values

    def build_inference_qc(
        self, input_values, distributed=False, draw_qc=False, **kwargs
    ):
        """This function builds the quantum circuit implementing the QFIE, initializing the input quantum registers
        according to the 'input_value' argument.
        
        Args:
             input_values (dict): dictionary containing the crisp input values of the system.
                E.g. {'var_name_1' (str): x_1 (float), , 'var_name_n' (str): x_n (float)}
             draw_qc (Bool - default:False): True for drawing the quantum circuit built. False otherwise.
             distributed (Boolean): True to implement the distributed version of the quantum oracle. False otherwise.
            :keyword filename (str): file path to save image to.
        
        Returns:
            None
        """
        self.distributed = distributed

        # Print Crisp Inputs
        if self.verbose:
            print(input_values)

        # FUZZIFICATION
        fuzzyfied_values = self.fuzzyfication(input_values=input_values)

        
        # CIRCUIT SETUPS
        # Not Distributed QFIE
        if not distributed:
            self.qc["full_circuit"] = QFS.generate_circuit(
                list(self.input_partitions.values()), encoding = self.encoding
            )
            self.qc["full_circuit"] = QFS.output_register(
                self.qc["full_circuit"], list(self.output_partition.values())[0]
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
                initial_state[var_name] = [
                    math.sqrt(fuzzyfied_values[var_name][i])
                    for i in range(len(fuzzyfied_values[var_name]))
                ]
                required_len = QFS.select_qreg_by_name(
                    list(self.qc.values())[0], var_name
                ).size
                while len(initial_state[var_name]) != 2**required_len:
                    initial_state[var_name].append(0)
                initial_state[var_name][-1] = math.sqrt(1 - sum(fuzzyfied_values[var_name]))
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
            for rule in self.rules:
                QFS.convert_rule(
                    qc=self.qc["full_circuit"],
                    fuzzy_rule=rule,
                    partitions=list(self.input_partitions.values()),
                    output_partition=list(self.output_partition.values())[0],
                    encoding=self.encoding
                )
                self.qc["full_circuit"].barrier()

            self.out_register_name = list(self.output_fuzzyset.keys())[0]
            out = ClassicalRegister(len(self.output_fuzzyset[self.out_register_name]))
            self.qc["full_circuit"].add_register(out)
            self.qc["full_circuit"].measure(
                QFS.select_qreg_by_name(
                    self.qc["full_circuit"], self.out_register_name
                ),
                out,
            )
            if draw_qc:
                print('draw')
                if "filename" in kwargs:
                    self.qc["full_circuit"].draw("mpl", filename=kwargs["filename"])
                else:
                    print('draw1')
                    self.qc["full_circuit"].draw("mpl").show()
        else:
            self.out_register_name = []
            # Use output linguistic terms as labels (keys) to identify the corresponding distributed circuits
            qc_labels = self.output_partition[list(self.output_fuzzyset.keys())[0]].sets
            for label in qc_labels:
                modified_output_partition = deepcopy(
                    list(self.output_partition.values())[0]
                )
                modified_output_partition.sets = [label]
                for rule in self.rule_subsets[label]:
                    QFS.convert_rule(
                        qc=self.qc[label],
                        fuzzy_rule=rule,
                        partitions=list(self.input_partitions.values()),
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
                        self.qc[label].draw("mpl", filename=label+'_'+kwargs["filename"])
                    else:
                        self.qc[label].draw("mpl")

    def execute(self, n_shots: int, plot_histo=False, GPU=False, **kwargs):
        """Run the inference engine.
        
        Args:
             n_shots (int): Number of shots.
             plot_histo (Bool- default False): True for plotting the counts histogram.
             GPU (Bool- default False): True for using GPU for simulation. Use False if backend is a real device.

            :keyword backend: quantum backend to run the quantum circuit. If not specified, qasm simulator is used.
            :keyword transpile_info (bool - default False): True for getting information about transpiled qc
            :keyword optimization_level (int - default 3): Select a Value from 1 to 3 to set the optimization level in the transpiling
            :keyword defuzzification (str): name of the Defuzzification algorithm to use. If not specified, 'centroid' is used. 
        Return:
            Crisp output of the system.
        """
        # Selecting the backend
        if "backend" in kwargs:
            backend = kwargs["backend"]
        else:
            backend = AerSimulator()

        #Checking Transpilation Command
        if "transpile_info" in kwargs and kwargs["transpile_info"] == True: transp_info = True
        else: transp_info = False

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

        self.n_q = len(self.output_fuzzyset[list(self.output_fuzzyset.keys())[0]])
        counts = self.counts_evaluator(n_qubits=self.n_q, counts=self.counts_)
        normalized_counts = counts
        self.output_dict = {
            i: []
            for i in self.output_partition[
                list(self.output_fuzzyset.keys())[0]
            ].sets
        }

        counter = 0
        for set in list(self.output_dict.keys()):
            counter = counter + 1
            for i in range(self.n_q):
                if i == self.n_q - counter:
                    self.output_dict[set].append("1")
                else:
                    self.output_dict[set].append("0")
            self.output_dict[set] = "".join(self.output_dict[set])

        memberships = {}
        for state in list(self.output_dict.values()):
            if state in list(normalized_counts.keys()):
                memberships[state] = normalized_counts[state]
            else:
                memberships[state] = 0

        # DEFUZZIFICATION
        if "defuzzification" in kwargs:
            defuzz = kwargs["defuzzification"]
        else: defuzz = 'centroid'

        norm_memberships = memberships
        self.alpha_cuts = {key: norm_memberships[value] for key, value in self.output_dict.items()}

        if self.verbose:
            print("Output Counts", memberships)
        activation = {}
        set_number = 0
        for set in list(self.output_dict.keys()):
            activation[set] = np.fmin(
                norm_memberships[self.output_dict[set]],
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