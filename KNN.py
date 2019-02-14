from collections import Counter

from auxiliary import simple_hamming_distance


class KNN:

    def __init__(self, examples):
        self.examples = examples

    def predict(self, k, test_entry):
        # map between entry index to distance, not entry itself, reducing space overhead
        distances = []
        for i in range(0, self.examples.len):
            entry = self.examples.data[i][:-1]
            dist = simple_hamming_distance(entry, test_entry)
            distances.append((i, dist))
        distances.sort(key=lambda x: x[1])
        classification_counter = Counter()
        for i in range(0, k):
            entry_index = distances[i][0]
            entry = self.examples.data[entry_index]
            entry_classification = self.examples.get_classification(entry)
            classification_counter.update({ entry_classification : 1 })
        common_classification = classification_counter.most_common(1)[0][0]
        return common_classification
