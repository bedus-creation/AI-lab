import numpy as np
import math


class BackPropagation:
    def __init__(self, no_of_inputs, no_of_h_layers, no_of_outputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.bias = np.array([1, 1])
        self.hidden_layers = np.zeros(no_of_h_layers)
        self.weights = []
        self.weights.append(np.zeros(no_of_inputs * no_of_h_layers))
        self.weights.append(np.zeros(no_of_h_layers * no_of_outputs))

    def activation(self, summation):
        return 1/(1+math.exp(-summation))

    def predict(self, inputs, weights, bias):
        print(weights[0:2])
        print(inputs)
        o1 = np.dot(inputs, weights[0:2]) + self.bias
        o2 = np.dot(inputs, weights[2:]) + self.bias
        print(o1)
        # return [o1, o2]

    def train(self, inputs, target):
        for _ in range(self.threshold):
            h = self.predict(inputs, self.weights[0], self.bias[0])
            # print(h)
