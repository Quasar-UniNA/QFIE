from itertools import groupby
import re
import math
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from . import FuzzyEngines as fe





def read_fis_file(file, verbose=False):
    """ Define a QuantumFuzzyEngine object from a .fis file.

        Args:
             name (str): .fis file.
             verbose (bool): True to graphically show the input and output fuzzy variables in the .fis file.

        Returns:
            QuantumFuzzyEngine Object
        """
    def from_list_to_dict(fis_group):
        dict_ = {}
        for item in fis_group[1:]:
            key, value = item.strip().split('=')
            value = value.strip()
            if value.isdigit():
                dict_[key] = int(value)
            elif '.' in value and all(c.isdigit() for c in value.replace('.', '', 1)):
                dict_[key] = float(value)
            else:

                dict_[key] = value.strip("'")
        return dict_

    def get_mf(dict_, val_range):
        n_mfs = dict_['NumMFs']
        out_dict = {}
        for i in range(1,n_mfs+1):
            string = dict_['MF'+str(i)]
            # Extract the values using regular expressions
            values = re.findall(r"[\w.-]+", string)

            # Process the extracted values
            output = [values[0].replace("'", "",1), values[1].replace("'", "",1), [float(num) for num in values[2:]]]

            if output[1]=='trimf':
                out_dict[output[0]]=fuzz.trimf(val_range, output[-1])
            if output[1]=='trapmf':
                out_dict[output[0]]=fuzz.trapmf(val_range, output[-1])
        return out_dict

    def plt_mf(fis, range_, mem, n):
        # Visualize these universes and membership functions
        if n>1: fig, axes = plt.subplots(nrows=n, figsize=(8, 12))
        else: fig, ax = plt.subplots()
        if n > 1:
            for i, ax in enumerate(axes):
                for mf in list(mem[fis[i]['Name']].keys()):
                    ax.plot(range_[fis[i]['Name']], mem[fis[i]['Name']][mf], linewidth=1.5,
                            label=mf)
                ax.set_title(fis[i]['Name'])
                ax.legend()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()
        else:
            for mf in list(mem[fis[0]['Name']].keys()):
                ax.plot(range_[fis[0]['Name']], mem[fis[0]['Name']][mf], linewidth=1.5,
                        label=mf)
            ax.set_title(fis[0]['Name'])
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()


        plt.show()


    with open(file) as f:
        fis = f.readlines()
        if '[System]\n' not in fis: raise 'Missing [System] section'
        if "Type='mamdani'\n" not in fis: raise 'Specify FIS Type. Note, that QFIE works just with Mamdani type'

        grouped_fis =[list(g) for k, g in groupby(fis, key=lambda x: x != "\n") if k]
        system_dict = from_list_to_dict(grouped_fis[0])
        n_inputs, n_outputs, n_rules = system_dict['NumInputs'], system_dict['NumOutputs'], system_dict['NumRules']
        input_fis, output_fis, = [], []
        for x in grouped_fis:
            if x[0][:3]=='[In':
                input_fis.append(from_list_to_dict(x))
            if x[0][:3]=='[Ou':
                output_fis.append(from_list_to_dict(x))
            if x[0][:3]=='[Ru':
                rules=x
        input_ranges = {input_fis[i]['Name']:np.linspace(float(input_fis[i]['Range'].strip('[]').split()[0]),
                                    float(input_fis[i]['Range'].strip('[]').split()[1]), 200) for i in range(len(input_fis))}
        output_ranges = {output_fis[i]['Name']: np.linspace(float(output_fis[i]['Range'].strip('[]').split()[0]),
                                    float(output_fis[i]['Range'].strip('[]').split()[1]), 200) for i in
                        range(len(output_fis))}
        input_mem,output_mem = {},{}
        for i in range(n_inputs):
            input_mem[input_fis[i]['Name']] = get_mf(input_fis[i], input_ranges[input_fis[i]['Name']])
        for i in range(n_outputs):
            output_mem[output_fis[i]['Name']] = get_mf(output_fis[i], output_ranges[output_fis[i]['Name']])
        if verbose:
            plt_mf(input_fis, mem=input_mem, range_=input_ranges, n=n_inputs)
            plt_mf(output_fis, mem=output_mem, range_=output_ranges, n=n_outputs)

        qfie = fe.QuantumFuzzyEngine(verbose=False)

        for _ in range(n_inputs):
            name = input_fis[_]['Name']
            mf_names = list(input_mem[name].keys())
            qfie.input_variable(name=name, range=input_ranges[name])
            qfie.add_input_fuzzysets(var_name=name, set_names=mf_names, sets=[input_mem[name][i] for i in mf_names])

        for _ in range(n_outputs):
            name = output_fis[_]['Name']
            mf_names = list(output_mem[name].keys())
            qfie.output_variable(name=name, range=output_ranges[name])
            qfie.add_output_fuzzysets(var_name=name, set_names=mf_names, sets=[output_mem[name][i] for i in mf_names])

        rules_as_list = []
        for item in rules[1:]:
            item = item.strip().split()
            inner_list = []
            for _ in item[:n_inputs+n_outputs]:
                if ',' in _:
                    _=_.replace(',', '')
                inner_list.append(int(_))
            rules_as_list.append(inner_list)

        linguistic_rules, input_names, mf_input_names, output_names, mf_output_names= [],[],[],[],[]
        for _ in range(n_inputs):
            name=input_fis[_]['Name']
            input_names.append(name)
            mf_input_names.append(list(input_mem[name].keys()))
        for _ in range(n_outputs):
            name = output_fis[_]['Name']
            output_names.append(name)
            mf_output_names.append(list(output_mem[name].keys()))
        for rule in rules_as_list:
            l_rule = 'if '
            for i in range(n_inputs):
                l_rule = l_rule + input_names[i] +' is ' + mf_input_names[i][rule[i]-1]
                if i != n_inputs-1:
                    l_rule = l_rule+ ' and '
                else:
                    l_rule = l_rule + ' then '
            for o in range(n_outputs):
                l_rule = l_rule + output_names[o] + ' is ' + mf_output_names[o][rule[n_inputs+o]-1]
                if o != n_outputs-1:
                    l_rule = l_rule+ ' and '
                else:
                    break
            linguistic_rules.append(l_rule)

        qfie.set_rules(linguistic_rules)


    return qfie

