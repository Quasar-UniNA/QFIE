import math


class fuzzy_partition:
    def __init__(self, name, sets, encoding='logaritmic', minimize_hamming=False, exact_partition=False):
        self.name = name
        self.sets = sets
        self.encoding =  encoding
        self.minimize_hamming = minimize_hamming
        self.exact_partition = exact_partition


    def len_partition(self):
        return len(self.sets)

    def register_size(self):
        """Number of qubits needed to represent this partition in logarithmic encoding.

        When minimize_hamming is enabled, an extra state is normally reserved to hold
        the leftover ("garbage") probability of an input whose membership values do not
        sum to 1. If the partition is flagged as exact (exact_partition=True), that
        reservation is skipped since the leftover probability is always zero.
        """
        n = self.len_partition()
        if self.minimize_hamming:
            if self.exact_partition:
                return max(1, math.ceil(math.log(n, 2)))
            return math.ceil(math.log(n + 1, 2))
        return math.ceil(math.log(n, 2))

    def associate_quantum_states(self):
        if self.encoding == 'logaritmic':
            len_state = self.register_size()
            binary_format = "{0:0" + str(len_state) + "b}"
            return {
                self.sets[i]: self._state_code(i, binary_format)
                for i in range(len(self.sets))
            }
        if self.encoding == 'linear':
            binary_dict = {}
            for i, element in enumerate(self.sets):
                # Create a binary string with one '1' at the i-th position (from right) and zeros elsewhere
                binary_string = ''.join('1' if j == i else '0' for j in range(len(self.sets)))
                binary_dict[element] = binary_string[::-1]
            return binary_dict

    def _state_code(self, index, binary_format):
        if self.minimize_hamming:
            gray_index = index ^ (index >> 1)
            return binary_format.format(gray_index)
        return binary_format.format(index)[::-1]



class fuzzy_rules:
    def __init__(self):
        return

    def add_rules(self, rule, partitions):
        """NB: specify in partitions the list of partitions which appears in the rule, in the order
        in which they appear"""
        split = rule.split()
        split = list(filter(("is").__ne__, split))
        converted_rule = split.copy()
        for word in split:
            for partition in partitions:
                if word == partition.name:
                    if split[split.index(word) + 1] !='not':
                        converted_rule[
                            split.index(word) + 1
                        ] = partition.associate_quantum_states()[
                            split[split.index(word) + 1]
                        ]
                    else: 
                        converted_rule[
                            split.index(word) + 2
                        ] = partition.associate_quantum_states()[
                            split[split.index(word) + 2]
                        ]
        return converted_rule
