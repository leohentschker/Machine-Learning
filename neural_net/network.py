"""
Implementation of a neural net following
the tutorial http://neuralnetworksanddeeplearning.com/
"""


import numpy as np
import random


class Network(object):

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def __init__(self, sizes):

        # stores the sizes of the layers in
        # the network
        self.num_layers = len(sizes)
        self.sizes = sizes

        # we consider the first layer as binary inputs, being only
        # on or off. As a result, they have no biases. Generate
        # random biases. Generate random biases from a gaussian
        # distribution
        self.biases =  [np.random.randn(num_biases, 1) for num_biases in sizes[1:]]

        # generate a weight matrix marking the weight of the edges between neurons
        # let weight[i][j][k] be w. i represents the layer we are considering, j represents
        # the index of the neuron that the edge is originating from, and k represents
        # the index of the destination neuron
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    @staticmethod
    def sigmoid(vector):
        # compute the sigmoid exponential function on the vector
        # elementwise. Returns a vector whose elements have
        # using the sigmoid function
        return 1.0 / (1.0 + np.exp(-vector))

    def feedforward(self, input_vector):
        # initialize the output vector as the input vector
        output_vector = np.copy(input_vector)

        # iterate through the layers of the network
        # and apply the transformations
        for bias, weight in zip(self.biases, self.weights):
            output_vector = Network.sigmoid(np.dot(weight, output_vector) + bias)

        return output_vector

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data is not None:
            n_test = len(test_data)

        training_size = len(training_data)
        for epoch_number in xrange(epochs):
            random.shuffle(training_data)

            # take training_batch / mini_batch_size different random batches and
            # update on those
            mini_batches = [training_data[batch_idx: batch_idx + mini_batch_size] for batch_idx in xrange(0, training_size, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data is not None:
                print "EPOCH {0}: {1} / {2}".format(epoch_number + 1, self.evaluate(test_data), n_test)
            else:
                print "EPOCH {0} complete".format(epoch_number + 1)

    def update_mini_batch(self, mini_batch, eta):
        # initialize the two gradients to zero
        bias_grad = [np.zeros(bias.shape) for bias in self.biases]
        weight_grad = [np.zeros(weight.shape) for weight in self.weights]

        for image_vector, correct_answer in mini_batch:
            delta_bias_grad, delta_weight_grad = self.backprop(image_vector, correct_answer)

            bias_grad = [orig_val + delta for orig_val, delta in zip(bias_grad, delta_bias_grad)]
            weight_grad = [orig_val + delta for orig_val, delta in zip(weight_grad, delta_weight_grad)]
 
        self.biases = [bias - (eta / len(mini_batch)) * delta_bias for bias, delta_bias in zip(self.biases, bias_grad)]
        self.weights = [weight - (eta / len(mini_batch)) * delta_weight for weight, delta_weight in zip(self.weights, weight_grad)]

    def backprop(self, image_vector, correct_response):
        # initialize the gradients to zero    
        bias_grad = [np.zeros(bias.shape) for bias in self.biases]
        weight_grad = [np.zeros(weight.shape) for weight in self.weights]

        activation = image_vector 
        activations = [image_vector]
        sigmoids = []

        for bias, weight in zip(self.biases, self.weights):
            result_vector =  np.dot(weight, activation) + bias
            sigmoids.append(result_vector)

            activation = Network.sigmoid(result_vector)

            activations.append(activation)

        delta = self.cost_derivative(activations[-1], correct_response) * Network.sigmoid_prime(sigmoids[-1])

        # initialize the bias to be the calculated delta
        bias_grad[-1] = delta
        weight_grad[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = sigmoids[-l]
            sigmoid_prime = Network.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime
            bias_grad[-l] = delta
            weight_grad[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (bias_grad, weight_grad)

    def evaluate(self, test_data):
            test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]

            return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y
 
    @staticmethod
    def sigmoid_prime(z):
        return Network.sigmoid(z) * (1 - Network.sigmoid(z))

