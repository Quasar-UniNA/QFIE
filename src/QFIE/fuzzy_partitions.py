import math


class fuzzy_partition:
    def __init__(self, name, sets, encoding='logaritmic'):
        self.name = name
        self.sets = sets
        self.encoding =  encoding

    def len_partition(self):
        return len(self.sets)

    def associate_quantum_states(self):
        if self.encoding == 'logaritmic':
            len_state = math.ceil(math.log(self.len_partition(), 2))
            binary_format = "{0:0" + str(len_state) + "b}"
            return {
                self.sets[i]: binary_format.format(i)[::-1] for i in range(len(self.sets))
            }
        if self.encoding == 'linear':
            binary_dict = {}
            for i, element in enumerate(self.sets):
                # Create a binary string with one '1' at the i-th position (from right) and zeros elsewhere
                binary_string = ''.join('1' if j == i else '0' for j in range(len(self.sets)))
                binary_dict[element] = binary_string[::-1]
            return binary_dict



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
                    converted_rule[
                        split.index(word) + 1
                    ] = partition.associate_quantum_states()[
                        split[split.index(word) + 1]
                    ]
        return converted_rule
