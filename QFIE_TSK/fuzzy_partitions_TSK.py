import math


class fuzzy_partition:
    def __init__(self, name, sets):
        self.name = name
        self.sets = sets

    def len_partition(self):
        return len(self.sets)

    def associate_quantum_states(self):
        # print('len' , self.len_partition())
        # print(math.log(len(self.sets)))
        len_state = math.ceil(math.log(self.len_partition(), 2))
        # print('len_state ', len_state)
        binary_format = "{0:0" + str(len_state) + "b}"
        return {self.sets[i]: binary_format.format(i) for i in range(len(self.sets))}


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
                    # print(partition.associate_quantum_states())
        return converted_rule
