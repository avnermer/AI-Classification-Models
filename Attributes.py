
class Attributes:
    def __init__(self, attributes_enumeration, classification_name, classes, possible_vals):
        self.enumeration = attributes_enumeration
        self.classification_name = classification_name
        self.classes = classes
        self.classes_enum = dict(zip(classes, range(len(classes))))
        self.possible_vals = possible_vals
        self.possible_vals_enumeration = { attribute: dict(
            zip(possible_vals, range(len(possible_vals))))for attribute, possible_vals in possible_vals.items()}
        self.len = len(attributes_enumeration)

    def num_of(self, a):
        return self.enumeration[a]

    def get_possible_vals(self, a):
        return self.possible_vals[a]