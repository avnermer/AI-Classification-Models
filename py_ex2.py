from collections import Counter, OrderedDict

from Id3Model import Id3Model
from Attributes import Attributes
from Examples import Examples
from KNN import KNN
from NaiveBayesModel import NaiveBayesModel
from auxiliary import column

if __name__ == '__main__':

    # read and store all training data
    data = []
    with open("train.txt", "r") as examples_file:
        header = examples_file.readline().strip().split("\t")
        for line in examples_file:
            if line == "\n":
                continue
            data.append(line.strip().split("\t"))

    attributes_enumeration = dict(enumerate(header[:-1]))
    classification_name = header[-1]
    attributes_possible_vals = {}
    for a in attributes_enumeration.keys():
        a_possible_vals = list(OrderedDict.fromkeys(column(data, a)))
        attributes_possible_vals[a] = a_possible_vals

    classification_index = len(header) - 1
    classes = list(OrderedDict.fromkeys(column(data, classification_index)))

    attributes = Attributes(attributes_enumeration, classification_name, classes, attributes_possible_vals)

    examples = Examples(attributes, data)

    id3_model = Id3Model(examples)
    id3_model.train()
    id3_model.print_tree("output_tree.txt")

    knn = KNN(examples)

    naive_base_model = NaiveBayesModel(examples)
    naive_base_model.train()

    with open("test.txt", "r") as test, open("output.txt", "w") as output:
        output.write("Num\tDT\tKNN\tnaiveBase\n")
        # skip header
        next(test)
        id3_correct_count = 0
        knn_correct_count = 0
        naive_bayes_correct_count = 0

        i = 1
        for line in test:
            if line == "\n":
                continue
            output.write(str(i) + "\t")
            entry = line.strip().split("\t")[:-1]
            actual_class = line.strip().split("\t")[-1]
            # ID3
            predicted_class = id3_model.predict(entry)
            output.write(predicted_class + "\t")
            if predicted_class == actual_class:
                id3_correct_count += 1
            # KNN
            predicted_class = knn.predict(5, entry)
            output.write(predicted_class + "\t")
            if predicted_class == actual_class:
                knn_correct_count += 1
            # NaiveBayes
            predicted_class = naive_base_model.predict(entry)
            output.write(predicted_class)
            if predicted_class == actual_class:
                naive_bayes_correct_count += 1
            output.write("\n")
            i += 1

        id3_success_rate = round(id3_correct_count / (i - 1), 2)
        knn_success_rate = round(knn_correct_count / (i - 1), 2)
        naive_bayes_success_rate = round(naive_bayes_correct_count / (i - 1), 2)
        output.write("\t" + str(id3_success_rate) + "\t" + str(knn_success_rate) + "\t" + str(naive_bayes_success_rate))