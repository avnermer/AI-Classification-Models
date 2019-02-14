from collections import Counter
from Attributes import Attributes
from Examples import Examples
from auxiliary import entropy


class Id3Model:
    def __init__(self, examples):
        self.examples = examples
        self.tree = {}

    def train(self):
        self.tree = self.__id3(self.examples, self.examples.attributes, "yes")

    def predict(self, entry):
        return self.__traverse(self.tree, entry)

    def __traverse(self, tree, entry):

        # base condition
        if tree == "yes" or tree == "no":
                return tree

        attribute = tree["#attribute"]
        subtree = [subtree for val, subtree in tree.items() if val == entry[attribute]][0]

        return self.__traverse(subtree, entry)


    def __id3(self, examples, attributes, default):
        # base conditions #

        # empty examples
        if not examples.data:
            return default

        # if all has identical classification
        elif examples.all_identical_classification():
            return examples.get_classification(examples.data[0])

        # if no more attributes
        elif not attributes.enumeration:
            return examples.most_frequent()

        else:
            best = self.choose_best_attribute(examples, attributes)
            tree = {"#attribute": best}

            # subtract: attributesCopy\{best}
            attributes_left_enum = dict(attributes.enumeration)
            attributes_left_enum.pop(best)
            possible_vals_left = dict(attributes.possible_vals)
            possible_vals_left.pop(best)
            attributes_left = Attributes(attributes_left_enum, attributes.classification_name, attributes.classes,
                                         possible_vals_left)

            # make a partition by best
            for value, examples_part in self.partition(examples, best).items():
                # recurse and add returned subtree
                tree[value] = self.__id3(examples_part, attributes_left, examples.most_frequent())
        return tree

    def gain(self, examples, a):
        # calc current entropy before division #
        data = examples.data
        data_len = len(data)
        # count positive and negative
        counter = Counter(examples.get_all_classifications())
        p = counter.get("yes")
        n = counter.get("no")
        current_entropy = entropy(p, n)

        # create dict of {value : [p , n]}
        vals = examples.attributes.get_possible_vals(a)
        vals_len = len(vals)
        val_count_dict = {a_val: [0, 0] for a_val in vals}

        # map all values to counts
        for entry in data:
            a_val = entry[a]
            entry_classification = entry[-1]
            count = val_count_dict[a_val]
            if entry_classification == "yes":
                count[0] += 1
            if entry_classification == "no":
                count[1] += 1

        # calculate gain #
        division_entropy_sum = 0
        for a_val, count in val_count_dict.items():
            pos_count = count[0]
            neg_count = count[1]
            proportion = (pos_count + neg_count) / data_len
            division_entropy_sum += proportion * entropy(pos_count, neg_count)

        return current_entropy - division_entropy_sum

    def choose_best_attribute(self, examples, attributes):
        max_gain = -1
        best = 0
        # number representation of attributes
        attributes = attributes.enumeration.keys()
        for a in attributes:
            a_gain = self.gain(examples, a)
            if (a_gain) > max_gain:
                max_gain = a_gain
                best = a
        return best

    def partition(self, examples, a):
        data = examples.data
        possible_vals = examples.attributes.possible_vals

        # partitioning data. dict of { a_val : all entries has a = val}
        data_partition = { a_val:[] for a_val in possible_vals[a]}
        for entry in data:
            val = entry[a]
            data_partition.get(val).append(entry)
        # turning dict type to { a_val : examples object}
        for val, data in data_partition.items():
            data_partition[val] = Examples(examples.attributes, data)
        return data_partition

    def print_tree(self, file_path):
        file = open(file_path, "w")
        self.__print_decision_tree(self.tree, 0, file)

    def __print_decision_tree(self, tree, depth, file):

        # base condition, a leaf
        if tree == "yes" or tree == "no":
                file.write(":" + tree)
                return
        attribute_num = tree["#attribute"]
        attribute_name = self.examples.attributes.enumeration[attribute_num]
        vals_subtrees = tree.items()
        first_line = True
        for val, subtree in sorted(vals_subtrees):
            # skip attribute name entry as it's not for printing as is
            if val == "#attribute":
                continue

            if depth == 0:
                if first_line:
                    file.write(depth * '\t' + attribute_name + "=" + val)
                    first_line = False
                else:
                    file.write("\n" + depth * '\t' + attribute_name + "=" + val)
            else:
                file.write("\n" + depth * '\t' + "|" + attribute_name + "=" + val)
            # recurse
            self.__print_decision_tree(subtree, depth + 1, file)