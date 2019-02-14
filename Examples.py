from collections import Counter

from auxiliary import column


class Examples:

    def __init__(self, attributes,  data):
        self.attributes = attributes
        self.data = data
        self.len = len(data)

    def get_all_classifications(self):
        return column(self.data, self.attributes.len)

    def get_classification(self, entry):
        return entry[-1]

    def most_frequent(self):
        all_classifications = self.get_all_classifications()
        most_common = Counter(all_classifications).most_common(2)
        count_1 = most_common[0][1]
        if len(most_common) >1:
            count_2 = most_common[1][1]
            if count_1 == count_2:
                return "yes"
        return most_common[0][0]

    def all_identical_classification(self):
        classification = self.get_classification(self.data[0])
        all_same = True
        for entry in self.data:
            if entry[-1] != classification:
                all_same = False
                break
        return all_same
