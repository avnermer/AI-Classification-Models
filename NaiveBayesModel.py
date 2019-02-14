from collections import Counter


class NaiveBayesModel:

    def __init__(self, examples):
        self.examples = examples
        self.classes_probs = {}
        self.conditional_prob_matrix = []

    def train(self):
        # classification probability calculation
        examples_len = self.examples.len
        attributes = self.examples.attributes
        classes_full_column = self.examples.get_all_classifications()
        classes_frequencies = Counter(classes_full_column)
        classes = attributes.classes
        self.classes_probs = { c : num / examples_len for c, num in classes_frequencies.items()}

        ''' for each attribute, create a matrix of smoothed conditional probability per class'''

        # enumerate classes
        classes_enum = attributes.classes_enum

        # for each attribute enumerate possible values, for matrix index mapping
        possible_vals_enum = attributes.possible_vals_enumeration
        conditional_probs = []
        for attribute in attributes.enumeration.items():
            attrib_num = attribute[0]
            possible_vals_len = len(attributes.get_possible_vals(attrib_num))
            # initialize matrix, smoothed
            conditional_probs.append([[1 / (classes_frequencies[c] + possible_vals_len) for c in classes]
                                      for r in range(possible_vals_len)])
        # increment probabilities in each intersection
        for entry in self.examples.data:
            classification = self.examples.get_classification(entry)
            classification_num = classes_enum[classification]
            classification_freq = classes_frequencies[classification]
            for attribute_num in range(attributes.len):
                value = entry[attribute_num]
                value_num = possible_vals_enum[attribute_num][value]
                possible_vals_len = len(attributes.get_possible_vals(attribute_num))
                smoothed_class_freq = classification_freq + possible_vals_len
                # increment freq
                conditional_probs[attribute_num][value_num][classification_num] += 1 / smoothed_class_freq

        self.conditional_prob_matrix = conditional_probs

    def predict(self, entry):
        attributes = self.examples.attributes
        max_probability = 0
        max_probability_class = 0
        # define default, for case of equal conditional probabilities
        # !!as shown in Q&A file: define by majority, and if equal, choose "yes"!!
        classes_full_column = self.examples.get_all_classifications()
        classes_frequencies = Counter(classes_full_column)
        most_common = classes_frequencies.most_common(2)
        default = most_common[0][0] if most_common[0][1] != most_common[1][1] else "yes"

        for cls in attributes.classes:
            cls_prob = self.classes_probs[cls]
            cls_num = attributes.classes_enum[cls]
            prob_multipication = cls_prob
            for attribute in attributes.enumeration.keys():
                val = entry[attribute]
                val_num = attributes.possible_vals_enumeration[attribute][val]
                prob_multipication *= self.conditional_prob_matrix[attribute][val_num][cls_num]

            if prob_multipication > max_probability:
                max_probability = prob_multipication
                max_probability_class = cls

            elif prob_multipication == max_probability:
                max_probability_class = default
        return max_probability_class

