import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import fuzzy_partitions_TSK as fp
import QFS_TSK as QFS
import math
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
    IBMQ,
    BasicAer,
)
from qiskit.visualization import plot_histogram
from qiskit import IBMQ
import itertools
import random


def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]


class QFIE_TSK:
    def __init__(self):
        self.input_ranges = {}
        self.output_range = {}
        self.input_fuzzysets = {}
        self.output_constants = {}
        self.input_partitions = {}
        self.variables = {}
        self.rules = []
        self.qc = ""

    def input_variable(self, name, range):
        if name in list(self.input_ranges.keys()):
            raise Exception("Variable name must be unambiguos")
        else:
            self.input_ranges[name] = range
            self.input_fuzzysets[name] = []
            self.input_partitions[name] = ""

    def add_input_fuzzysets(self, var_name, set_names, sets):
        for set in sets:
            self.input_fuzzysets[var_name].append(set)
        self.input_partitions[var_name] = fp.fuzzy_partition(var_name, set_names)

    def set_rules(self, rules):
        self.rules = rules

    def get_bin(self, x, n):
        return format(x, "b").zfill(n)

    def build_inference_qc(self, input_values, draw_qc=False):
        """input_values must be a dictionary {'var_name': value}"""
        self.qc = QFS.generate_circuit(list(self.input_partitions.values()))
        self.output_constants = {
            self.get_bin(i, self.qc.width())[::-1]: 0
            for i in range(2 ** self.qc.width())
        }
        y_reg = QuantumRegister(self.qc.width(), name="y")
        anc_reg = QuantumRegister(1, name="ancilla")
        self.qc.add_register(y_reg)
        self.qc.add_register(anc_reg)
        self.modulus = {i: 0 for i in list(input_values.keys())}
        self.inputs = input_values
        print(input_values)
        fuzzyfied_values = {}
        for var_name in list(input_values.keys()):
            fuzzyfied_values[var_name] = [
                fuzz.interp_membership(
                    self.input_ranges[var_name], i, input_values[var_name]
                )
                for i in self.input_fuzzysets[var_name]
            ]
            # norm_values[var_name] = [self.truncate(float(i)/sum(fuzzyfied_values[var_name]), 3) for i in fuzzyfied_values[var_name]]
        print("Input values ", fuzzyfied_values)
        initial_state, self.initial_state_normalized = {}, {}
        for var_name in list(input_values.keys()):
            initial_state[var_name] = [
                math.sqrt(fuzzyfied_values[var_name][i])
                for i in range(len(fuzzyfied_values[var_name]))
            ]
            required_len = QFS.select_qreg_by_name(self.qc, var_name).size
            while len(initial_state[var_name]) != 2**required_len:
                initial_state[var_name].append(0)

            # initial_state[var_name][-1] = math.sqrt(1 - sum(fuzzyfied_values[var_name]))
            self.modulus[var_name] = np.linalg.norm(
                np.array(initial_state[var_name]), ord=2
            )
            # print(initial_state)
            self.initial_state_normalized[var_name] = [
                i / self.modulus[var_name] for i in initial_state[var_name]
            ]
            self.qc.initialize(
                self.initial_state_normalized[var_name],
                QFS.select_qreg_by_name(self.qc, var_name),
            )
        print(
            "inp_state ",
            list(itertools.product(*self.initial_state_normalized.values())),
        )

        for rule in self.rules:
            QFS.convert_rule(
                fuzzy_rule=rule,
                partitions=list(self.input_partitions.values()),
                output_constants=self.output_constants,
            )

        self.vec_out_cost = np.array(list(self.output_constants.values()))

        self.out_modulus = np.linalg.norm(self.vec_out_cost, ord=2)
        normalized_outputs = list(self.vec_out_cost / self.out_modulus)
        # normalized_outputs = [float(i) / math.sqrt(sum(list(self.output_constants.values())) for i in
        # list(self.output_constants.values())]

        print(normalized_outputs)
        self.qc.initialize(normalized_outputs, QFS.select_qreg_by_name(self.qc, "y"))
        self.qc.h(self.qc.width() - 1)

        for i in range(0, y_reg.size):
            self.qc.cswap(anc_reg[0], y_reg[y_reg.size - 1 - i], self.qc.qubits[i])
        self.qc.h(self.qc.width() - 1)
        cr = ClassicalRegister(1)
        self.qc.add_register(cr)
        self.qc.measure(anc_reg, cr)
        inp_len = self.qc.num_qubits - y_reg.size - 1
        cr2 = ClassicalRegister(inp_len)
        self.qc.add_register(cr2)
        self.qc.barrier()
        self.qc.measure(self.qc.qubits[:inp_len], cr2)
        if draw_qc:
            self.qc.draw("mpl").show()

    def execute(self, backend_name, n_shots, provider=None, plot_histo=False):

        if backend_name == "qasm_simulator":
            backend = BasicAer.get_backend(backend_name)
        else:
            backend = provider.get_backend(backend_name)

        job = execute(self.qc, backend, shots=n_shots)
        result = job.result()
        if plot_histo:
            plot_histogram(
                job.result().get_counts(), color="midnightblue", figsize=(7, 10)
            ).show()
        counts_ = job.result().get_counts()
        print(counts_)
        p_1 = 0
        for i in list(counts_.keys()):
            if i[-1] == "1":
                p_1 = p_1 + counts_[i]
        print(p_1)
        scalar_prod = 1 - 2 * p_1 / n_shots
        print("scalar prod ", scalar_prod)

        counts_sorted = {
            k: v for k, v in sorted(counts_.items(), key=lambda item: item[1])
        }
        print(counts_sorted)
        check, i = False, 1
        while check is False:

            state_to_look = list(counts_sorted.keys())[-i][:-2]
            print("state_to_look ", state_to_look)
            if self.output_constants[state_to_look] < 0:
                self.sign = -1
                check = True
            if self.output_constants[state_to_look] > 0:
                self.sign = 1
                check = True
            if self.output_constants[state_to_look] == 0:
                i = i + 1

        output = scalar_prod
        # for i in list(self.modulus.values()):
        # output = output * (i)
        output = output * (self.out_modulus**2)
        # print(math.sqrt(abs(output)))
        # return math.sqrt(abs(output))
        return output


hum = np.linspace(0, 100, 200)
temp = np.linspace(0, 45, 200)


h_dry = fuzz.trimf(hum, [0, 0, 30])
h_comfortable = fuzz.trimf(hum, [20, 37, 50])
h_humid = fuzz.trimf(hum, [40, 60, 70])
h_sticky = fuzz.trimf(hum, [60, 100, 100])

t_very_low = fuzz.trimf(temp, [0, 0, 12.5])
t_low = fuzz.trimf(temp, [9, 21, 30])
t_high = fuzz.trimf(temp, [25, 30, 35])
t_very_high = fuzz.trimf(temp, [30, 45, 45])

speed_off = "0"
speed_low = "0.3333"
speed_medium = "0.6667"
speed_fast = "1"

rules = [
    "if temp is very_low and hum is dry the speed is then " + speed_off,
    "if temp is very_low and hum is comfortable the speed is then " + speed_off,
    "if temp is very_low and hum is humid the speed is then " + speed_off,
    "if temp is very_low and hum is sticky the speed is then " + speed_low,
    "if temp is low and hum is dry the speed is then " + speed_off,
    "if temp is low and hum is comfortable the speed is then " + speed_off,
    "if temp is low and hum is humid the speed is then " + speed_low,
    "if temp is low and hum is sticky the speed is then " + speed_medium,
    "if temp is high and hum is dry the speed is then " + speed_low,
    "if temp is high and hum is comfortable the speed is then " + speed_medium,
    "if temp is high and hum is humid the speed is then " + speed_fast,
    "if temp is high and hum is sticky the speed is then " + speed_fast,
    "if temp is very_high and hum is dry the speed is then " + speed_medium,
    "if temp is very_high and hum is comfortable the speed is then " + speed_fast,
    "if temp is very_high and hum is humid the speed is then " + speed_fast,
    "if temp is very_high and hum is sticky the speed is then " + speed_fast,
]

# Visualize these universes and membership functions
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(8, 9))


ax0.plot(temp, t_very_low, "b", linewidth=1.5, label="very_low")
ax0.plot(temp, t_low, "g", linewidth=1.5, label="low")
ax0.plot(temp, t_high, "r", linewidth=1.5, label="high")
ax0.plot(temp, t_very_high, "c", linewidth=1.5, label="very_high")
ax0.set_title("T")
ax0.legend()

ax1.plot(hum, h_dry, "b", linewidth=1.5, label="dry")
ax1.plot(hum, h_comfortable, "g", linewidth=1.5, label="comfortable")
ax1.plot(hum, h_humid, "r", linewidth=1.5, label="humid")
ax1.plot(hum, h_sticky, "c", linewidth=1.5, label="sticky")
ax1.set_title("H")
ax1.legend()


# Turn off top/right axes
for ax in (ax0, ax1):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


plt.show()


qf_tsk = QFIE_TSK()
qf_tsk.input_variable(name="temp", range=temp)
qf_tsk.input_variable(name="hum", range=hum)

qf_tsk.add_input_fuzzysets(
    var_name="temp",
    set_names=["very_low", "low", "high", "very_high"],
    sets=[t_very_low, t_low, t_high, t_very_high],
)
qf_tsk.add_input_fuzzysets(
    var_name="hum",
    set_names=["dry", "comfortable", "humid", "sticky"],
    sets=[h_dry, h_comfortable, h_humid, h_sticky],
)


qf_tsk.set_rules(rules)

temp_list, h_list = [t for t in range(0, 42, 2)], [_ for _ in range(0, 105, 5)]
speed_list = []
for t in temp_list:
    for h in h_list:

        qf_tsk.build_inference_qc({"temp": t, "hum": h}, draw_qc=False)
        y_norm = qf_tsk.execute("qasm_simulator", 65536, plot_histo=False)
        y_norm = qf_tsk.sign * math.sqrt(abs(y_norm))
        if y_norm > 1:
            speed_list.append(1)
        else:
            speed_list.append(y_norm)
        print("temp hum  speed", t, h, y_norm)

from mpl_toolkits.mplot3d import Axes3D

Y, X, = (
    temp_list,
    h_list,
)
plotx, ploty, = np.meshgrid(
    np.linspace(np.min(X), np.max(X), 21), np.linspace(np.min(Y), np.max(Y), 21)
)
Z = np.zeros_like(plotx)
counter = 0
for i in range(len(temp_list)):
    for j in range(len(h_list)):
        Z[i, j] = speed_list[counter]
        counter += 1

zfig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot_surface(
    plotx,
    ploty,
    Z * 100,
    rstride=1,
    cstride=1,
    cmap="viridis",
    linewidth=0.4,
    antialiased=True,
)
ax.set_xlabel("H")
ax.set_xlim(100, 0)
ax.set_ylabel("T")
ax.set_zlabel("Speed")

ax.view_init(30, -45)
plt.show()
