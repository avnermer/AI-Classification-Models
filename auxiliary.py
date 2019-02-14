from math import log


def simple_hamming_distance(x, y):
    dist = 0
    # assuming identical lengths
    for x_val, y_val in zip(x, y):
        if x_val != y_val:
            dist += 1
    return dist


# returns matrix ith column
def column(matrix, i):
    return [row[i] for row in matrix]


def entropy(p, n):

    if p + n is 0:
        return 0
    elif p is 0 or n is 0:
        return 0
    sum = p + n
    return -(p / sum) * log(p / sum, 2) - (n / sum) * log(n / sum, 2)