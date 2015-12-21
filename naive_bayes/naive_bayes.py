import csv
import copy
import random
import numpy as np
import math


def load_test_data(test_file_path):
    """
    Read in the data in the correct format
    """
    lines = csv.reader(open(test_file_path, "rb"))
    
    unformatted_data_set = list(lines)
    
    # map the data to floats for calculation purposes
    formatted_data = [map(float, data_line) for data_line in unformatted_data_set]
    return formatted_data


def split_data(test_data, split_ratio):
    """
    Splits a dataset into two pieces, one to
    be used for training and the other for
    testing
    """
    split_index = int(split_ratio * len(test_data))
    
    # randomly permute the values in place
    random.shuffle(test_data)
    
    # take slices of the determined size
    training_set = copy.copy(test_data[:split_index])
    test_data = copy.copy(test_data[split_index:])

    return training_set, test_data


def separate_by_class(dataset, class_index):
    """
    Returns a dictionary mapping the class
    values to their data values. By default this function
    assumes that the class value is stored at the last index
    """
    class_dictionary = {}
    for data_row in dataset:
        # determine what to use as a key
        # for the dictionary
        dict_key = data_row[class_index]
        
        # remove the class attribute from the
        # data so it doesn't screw up stats
        del data_row[class_index]

        if dict_key not in class_dictionary:
            class_dictionary[dict_key] = [data_row]
        else:
            class_dictionary[dict_key].append(data_row)

    return class_dictionary


def summarize(dataset):
    """
    Takes in a dataset in the format [(a, b, c), (d, e, f)]
    where each tuple represents a class value that we are considering
    """
    summaries = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*dataset)]

    return summaries


def summarize_by_class(dataset, class_index):
    separated_dict = separate_by_class(dataset, class_index)

    summarized_data_dict = {}
    for class_key, data_rows in separated_dict.iteritems():
        summary = summarize(data_rows)
        summarized_data_dict[class_key] = summary

    return summarized_data_dict


def calculate_probability(value, mean, stdev):
    """
    Takes in a value, the mean for that distribution
    and the standard devation and returns the probability
    of that value occurring. Rests on the idea that
    the distribution is normal
    """
    exponent = math.exp(- pow(value - mean, 2) / (2 * pow(stdev, 2)))

    return exponent / (stdev * pow(math.pi * 2, .5))


def calculate_class_probabilities(summaries, input_vector):
    """
    Stores a dictionary with class keys mapping to the probability
    that the input vector maps to that class.
    """
    probabilities = {}

    for class_key, class_summary in summaries.iteritems():
        # initialize the probability for the class to 1 to
        # prevent keyerrors
        probabilities[class_key] = float(1)

        for (mean, stdev), input_val  in zip(class_summary, input_vector):
            attribute_probability = calculate_probability(input_val, mean, stdev)
            probabilities[class_key] *= attribute_probability

    return probabilities


def predict(summaries, input_vector):
    """
    Given the mean and stdev summaries as well as
    an input vector, this function determines which
    class the input vector is most likely to
    fall into
    """
    class_probabilities = calculate_class_probabilities(summaries, input_vector)

    probability_tuples = [(probability, key) for key, probability in class_probabilities.items()]

    max_probability, matched_class = max(probability_tuples)

    return matched_class


def get_predictions(summaries, test_sets):
    """
    Takes in a set of summaries and a list
    of datasets to test on and generates predictions
    based upon the training data
    """
    predictions = []
    for test_data in test_sets:
        result = predict(summaries, test_data)
        predictions.append(result)

    return predictions


def get_accuracy(test_sets, predictions, class_index):
    """
    Determines the percentage of the test cases
    that we calculated accurately
    """
    actual_classes = [test_set[class_index] for test_set in test_sets]

    num_correct = sum(int(actual == prediction) for actual, prediction in zip(actual_classes, predictions))

    return float(num_correct) / len(test_sets)


def run_bayes(data_file_path, class_index = -1):
    input_data = load_test_data(data_file_path)
    split_ratio = .5

    training_data, test_data = split_data(input_data, split_ratio)

    class_summarized_data = summarize_by_class(training_data, class_index)

    predictions = get_predictions(class_summarized_data, test_data)

    accuracy = get_accuracy(test_data, predictions, class_index)

    print "ACCURACY", accuracy


if __name__ == "__main__":
    test_file_path = "pima-indians-diabetes.data"
    run_bayes(test_file_path)
