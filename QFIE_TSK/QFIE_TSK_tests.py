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
            print("var_name ", var_name, initial_state[var_name])
            self.modulus[var_name] = np.linalg.norm(
                np.array(initial_state[var_name]), ord=2
            )
            # print(initial_state)
            self.initial_state_normalized[var_name] = [
                i / self.modulus[var_name] for i in initial_state[var_name]
            ]
            print(self.initial_state_normalized[var_name])
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
        print("inp_len ", inp_len, self.qc.num_qubits, y_reg.size)
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
            if qf_tsk.output_constants[state_to_look] < 0:
                self.sign = -1
                check = True
            if qf_tsk.output_constants[state_to_look] > 0:
                self.sign = 1
                check = True
            if qf_tsk.output_constants[state_to_look] == 0:
                i = i + 1

        output = scalar_prod
        for i in list(self.modulus.values()):
            output = output * (i**2)
        output = output * (self.out_modulus**2)
        # print(math.sqrt(abs(output)))
        # return math.sqrt(abs(output))
        return output


data_TB1 = [10, 17, 22, 50, 9, 11, 12, 14, 35, 20, 20, 18, 12, 8, 11, 50, 35, 30, 16]
data_PH = [
    7.1,
    7.0,
    7.3,
    7.1,
    7.3,
    7.1,
    7.2,
    7.2,
    7.0,
    7.0,
    6.9,
    7.1,
    7.2,
    7.2,
    7.1,
    7.0,
    7.0,
    7.0,
    7.1,
]
data_TE = [
    18.8,
    18.6,
    19.4,
    19.5,
    23.3,
    20.7,
    21.3,
    23.6,
    17.8,
    16.6,
    17.8,
    17.3,
    18.8,
    18.0,
    19.2,
    18.0,
    17.7,
    17.3,
    19.3,
]
data_AL = [53, 50, 46, 40, 48, 50, 50, 53, 35, 40, 42, 40, 55, 50, 49, 37, 42, 41, 42]
data_PAC = [
    1300,
    1300,
    1400,
    1400,
    900,
    900,
    900,
    900,
    1200,
    1100,
    1100,
    1100,
    900,
    1000,
    1000,
    1200,
    1200,
    1100,
    1100,
]
data_TB2 = [1, 1, 2, 1, 4, 1, 3, 4, 1, 1, 1, 1, 3, 1.5, 2, 1.5, 1.5, 1.5, 3]


def consequents(TB1, PH, TE, AL, TB2):
    c1 = 2664 * TB1 - 8093 * TB2 + 11230 * PH - 1147 * AL - 2218 * TE + 8858
    c2 = 124 * TB1 - 427 * TB2 + 761 * PH + 52 * AL - 17 * TE - 7484
    c3 = 42 * TB1 - 54 * TB2 - 1368 * PH + 10 * AL + 158 * TE + 7270
    c4 = 5 * TB1 - 34 * TB2 - 221 * PH - 8 * AL + 40 * TE + 2202
    c5 = 3 * TB1 - 6 * TB2 + 2110 * PH - 13 * AL + 2 * TE - 13918
    c6 = 22 * TB1 + 11 * TB2 + 64 * PH - 8 * AL - 9 * TE + 770
    c7 = 159 * TB1 - 14 * TB2 + 2337 * PH - 25 * AL - 69 * TE - 14819
    c8 = -13 * TB1 - 16 * TB2 + 29 * PH + 6 * AL + 41 * TE - 317
    """
    return [renormalize(c1,[-40144.8, 138980.2],[0,1]),
            renormalize(c2, [-1530.2999999999993, 6422.0999999999985], [0, 1]),
            renormalize(c3, [376.40000000000055, 4155.599999999999], [0, 1]),
            renormalize(c4, [716.6999999999998, 1557.1], [0, 1]),
            renormalize(c5, [-40.79999999999927, 1221.2000000000007], [0, 1]),
            renormalize(c6, [746.2, 1951.8], [0, 1]),
            renormalize(c7,[ -481.0999999999967, 8156.699999999997],[0,1]),
            renormalize(c8, [59.700000000000045, 1072.3], [0, 1])]
    """
    return [c1, c2, c3, c4, c5, c6, c7, c8]


PH = np.linspace(6.90, 7.50, 200)
AL = np.linspace(35.0, 60.0, 200)
TE = np.linspace(16.6, 24.6, 200)


PH_small = fuzz.trimf(PH, [6.90, 6.90, 7.30])
PH_big = fuzz.trimf(PH, [7.25, 7.50, 7.50])
# PH_dummy = fuzz.trimf(PH, [7.53,7.53,7.53])

AL_small = fuzz.trimf(AL, [35.0, 35.0, 53.0])
AL_big = fuzz.trimf(AL, [50.9, 60.0, 60.0])
# AL_dummy = fuzz.trimf(AL, [60.3,60.3,60.3])

TE_small = fuzz.trimf(TE, [16.6, 16.6, 20.1])
TE_big = fuzz.trimf(TE, [18.7, 24.6, 24.6])
# TE_dummy = fuzz.trimf(TE, [24.9, 24.9, 24.9])

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))


ax0.plot(PH, PH_small, "b", linewidth=1.5, label="cold")
ax0.plot(PH, PH_big, "g", linewidth=1.5, label="warm")
ax0.set_title("PH")
ax0.legend()

ax1.plot(AL, AL_small, "b", linewidth=1.5, label="low")
ax1.plot(AL, AL_big, "g", linewidth=1.5, label="normal")
ax1.set_title("AL")
ax1.legend()


ax2.plot(TE, TE_small, "b", linewidth=1.5, label="slow")
ax2.plot(TE, TE_big, "g", linewidth=1.5, label="fast")
ax2.set_title("TE")
ax2.legend()

# Turn off top/right axes
for ax in (ax0, ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


plt.show()


qf_tsk = QFIE_TSK()
qf_tsk.input_variable(name="varPH", range=PH)
qf_tsk.input_variable(name="varAL", range=AL)
qf_tsk.input_variable(name="varTE", range=TE)

qf_tsk.add_input_fuzzysets(
    var_name="varPH", set_names=["small", "big"], sets=[PH_small, PH_big]
)
qf_tsk.add_input_fuzzysets(
    var_name="varAL", set_names=["small", "big"], sets=[AL_small, AL_big]
)
qf_tsk.add_input_fuzzysets(
    var_name="varTE", set_names=["small", "big"], sets=[TE_small, TE_big]
)


i_data = 5
varPH, varAL, varTE, varTB1, varTB2 = (
    data_PH[i_data],
    data_AL[i_data],
    data_TE[i_data],
    data_TB1[i_data],
    data_TB2[i_data],
)
c = consequents(TB1=varTB1, TB2=varTB2, PH=varPH, TE=varTE, AL=varAL)
print("consequents ", c)

print(
    "inputs varPH, varAL, varTE, varTB1, varTB2 :", varPH, varAL, varTE, varTB1, varTB2
)
rules = [
    "if varPH is small and varAL is small and varTE is small then "
    + str(consequents(TB1=varTB1, TB2=varTB2, PH=varPH, TE=varTE, AL=varAL)[0]),
    "if varPH is small and varAL is small and varTE is big then "
    + str(consequents(TB1=varTB1, TB2=varTB2, PH=varPH, TE=varTE, AL=varAL)[1]),
    "if varPH is small and varAL is big and varTE is small then "
    + str(consequents(TB1=varTB1, TB2=varTB2, PH=varPH, TE=varTE, AL=varAL)[2]),
    "if varPH is small and varAL is big and varTE is big then "
    + str(consequents(TB1=varTB1, TB2=varTB2, PH=varPH, TE=varTE, AL=varAL)[3]),
    "if varPH is big and varAL is small and varTE is small then "
    + str(consequents(TB1=varTB1, TB2=varTB2, PH=varPH, TE=varTE, AL=varAL)[4]),
    "if varPH is big and varAL is small and varTE is big then "
    + str(consequents(TB1=varTB1, TB2=varTB2, PH=varPH, TE=varTE, AL=varAL)[5]),
    "if varPH is big and varAL is big and varTE is small then "
    + str(consequents(TB1=varTB1, TB2=varTB2, PH=varPH, TE=varTE, AL=varAL)[6]),
    "if varPH is big and varAL is big and varTE is big then "
    + str(consequents(TB1=varTB1, TB2=varTB2, PH=varPH, TE=varTE, AL=varAL)[7]),
]

qf_tsk.set_rules(rules)

qf_tsk.build_inference_qc(
    {"varPH": varPH, "varAL": varAL, "varTE": varTE}, draw_qc=True
)
y_norm = qf_tsk.execute("qasm_simulator", 65536, plot_histo=False)
y_norm = qf_tsk.sign * math.sqrt(abs(y_norm))
print(y_norm)

y = renormalize(
    y_norm,
    [
        min(list(qf_tsk.output_constants.values())),
        max(list(qf_tsk.output_constants.values())),
    ],
    [min(data_PAC), max(data_PAC)],
)
print(y)

"""''
x = np.linspace(-0.75, 0.75, 200)
v = np.linspace(-0.75, 0.75, 200)


x_NL = fuzz.trapmf(x, [-0.75, -0.75, -0.5, -0.25])
x_NS = fuzz.trimf(x, [-0.5, -0.25, 0])
x_0 = fuzz.trimf(x, [-0.25, 0, 0.25])
x_PS = fuzz.trimf(x, [0, 0.25, 0.5])
x_PL = fuzz.trapmf(x, [0.25,  0.5, 0.75, 0.75])

v_NL = fuzz.trapmf(v, [-0.75, -0.75, -0.5, -0.25])
v_NS = fuzz.trimf(v, [-0.5, -0.25, 0])
v_0 = fuzz.trimf(v, [-0.25, 0, 0.25])
v_PS = fuzz.trimf(v, [0, 0.25, 0.5])
v_PL = fuzz.trapmf(v, [0.25,  0.5, 0.75, 0.75])

rules = [
         'if x is x_NL and v is v_NL then 0.16',
         'if x is x_NL and v is v_NS then 0.16',
         'if x is x_NL and v is v_0 then 0.16',
         'if x is x_NL and v is v_PS then 0.08',
         'if x is x_NL and v is v_PL then 0',
         'if x is x_NS and v is v_NL then 0.16',
         'if x is x_NS and v is v_NS then 0.16',
         'if x is x_NS and v is v_0 then 0.08',
         'if x is x_NS and v is v_PS then 0',
         'if x is x_NS and v is v_PL then -0.08',
         'if x is x_0 and v is v_NL then 0.16',
         'if x is x_0 and v is v_NS then 0.08',
         'if x is x_0 and v is v_0 then 0',
         'if x is x_0 and v is v_PS then -0.08',
         'if x is x_0 and v is v_PL then -0.16',
         'if x is x_PS and v is v_NL then 0.08',
         'if x is x_PS and v is v_NS then 0',
         'if x is x_PS and v is v_0 then -0.08',
         'if x is x_PS and v is v_PS then -0.16',
         'if x is x_PS and v is v_PL then -0.16',
         'if x is x_PL and v is v_NL then 0',
         'if x is x_PL and v is v_NS then -0.08',
         'if x is x_PL and v is v_0 then -0.16',
         'if x is x_PL and v is v_PS then -0.16',
         'if x is x_PL and v is v_PL then -0.16'

         ]


qf_tsk = QFIE_TSK()
qf_tsk.input_variable(name='x', range=x)
qf_tsk.input_variable(name='v', range=v)

qf_tsk.add_input_fuzzysets(var_name='x', set_names=['x_NL', 'x_NS', 'x_0', 'x_PS', 'x_PL'], sets=[x_NL, x_NS, x_0, x_PS, x_PL])
qf_tsk.add_input_fuzzysets(var_name='v', set_names=['v_NL', 'v_NS', 'v_0', 'v_PS', 'v_PL'], sets=[v_NL, v_NS, v_0, v_PS, v_PL])


qf_tsk.set_rules(rules)

t = 0.1
x_0, v_0 = 0.5, 0.5
x_vec, v_vec = [x_0], [v_0]
for _ in range(60):
    print('iteration ', _)
    print('Position, velocity ', x_0, v_0)
    qf_tsk.build_inference_qc({'x':x_0, 'v':v_0}, draw_qc=False)
    F_2 = qf_tsk.execute('qasm_simulator', 65536, plot_histo=False)
    print('F^2 ', F_2)
    F = math.sqrt(abs(F_2))
    if F > 0.18:
        F = 0.18
    F = qf_tsk.sign * F
    print('F ', F)


    #F=renormalize(F, [0, 0.16], [-0.16, 0.16])
    #print(F)


    x_n = x_0 + v_0 * t + 0.5 * (F/0.2) * t * t
    v_n = v_0 + (F/0.2) * t

    x_0,v_0 =x_n,v_n

    if x_0 > 0.75:
        x_0 = 0.75
    x_vec.append(x_0)
    v_vec.append(v_0)

plt.plot([_ for _ in range(61)], x_vec)
plt.show()
plt.plot([_ for _ in range(61)], v_vec)
plt.show()

""" ""
