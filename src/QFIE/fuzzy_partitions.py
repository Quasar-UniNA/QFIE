import math


class fuzzy_partition:
    def __init__(self, name, sets, encoding='logaritmic', minimize_hamming=False):
        self.name = name
        self.sets = sets
        self.encoding =  encoding
        self.minimize_hamming = minimize_hamming
       

    def len_partition(self):
        return len(self.sets)

    def associate_quantum_states(self):
        if self.encoding == 'logaritmic':
            if self.minimize_hamming:
                len_state = math.ceil(math.log(self.len_partition() + 1, 2))
            else:
                len_state = math.ceil(math.log(self.len_partition(), 2))
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
