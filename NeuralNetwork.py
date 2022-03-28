import random
import numpy as np
import matplotlib.pyplot as plt
#from keras.datasets import mnist


def sigmoid(matrix):
    return 1 / (1 + np.exp(-matrix))


class Layer:
    def __init__(self, size):
        self.size = size
        self.weights = None
        self.activations = np.array([[0] for _ in range(size)])
        self.biases = None


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.size = 0

    def add_layer(self, position, size):
        self.layers.insert(position, Layer(size))
        self.size += 1

    def display_network(self):
        count = 1
        for layer in self.layers:
            print('Layer ' + str(count) + ':')
            for activation in np.nditer(layer.activations):
                print(activation, end=' ')
            count += 1
            print('')

    def initialise(self):
        for i in range(self.size - 1):
            self.layers[i].weights = np.array([[((6 * random.random() - 3) / 10) for _ in range(self.layers[i].size)]
                                               for _ in range(self.layers[i+1].size)])
            self.layers[i].biases = np.array([[0.] for _ in range(self.layers[i+1].size)])

    def feed_forward(self, inputs):
        self.layers[0].activations = inputs
        for i in range(1, self.size):
            self.layers[i].activations = sigmoid(np.dot(self.layers[i-1].weights, self.layers[i-1].activations) +
                                                 self.layers[i-1].biases)

    def get_output(self, inputs):
        self.feed_forward(np.array([[_] for _ in inputs]))
        out = []
        for i in range(self.layers[-1].size):
            out.append(self.layers[-1].activations[i, 0])

        return out

    def back_propagate(self, inputs, outputs):
        self.feed_forward(inputs)
        nabla_weights = []
        nabla_biases = []
        delta_activations = [2*(self.layers[-1].activations - outputs)/self.layers[-1].size]  # Far right is layer 0, top is node 0

        for i in range(2, self.size+1):
            delta_activations.append(np.dot(np.expand_dims(self.layers[-i].weights[0,:], axis=1),
                                            delta_activations[i-2][0, 0] * self.layers[1-i].activations[0, 0] *
                                            (1 - self.layers[1-i].activations[0, 0])))

        for j in range(self.size-1):
            nabla_biases.append(delta_activations[j]*self.layers[-1-j].activations*(1-self.layers[-1-j].activations))
            nabla_weights.append(np.dot(nabla_biases[j], np.transpose(self.layers[-2-j].activations)))

        return nabla_weights, nabla_biases

    def fit(self, nabla_w, nabla_b, learning_rate, momentum):
        for i in range(self.size-1):
            self.layers[i].weights -= nabla_w[-1-i]*learning_rate
            self.layers[i].biases -= nabla_b[-1-i]*learning_rate

    def train(self, training_data, epochs=1, size=32, algorithm='sgd', momentum=0., loss='mse', activation='sigmoid',
              validate=[]):
        # training data will be cut short if not divisible by number of batch size
        # different algorithms, loss functions and activation functions not yet implemented
        accuracies = []
        for e in range(epochs):
            for i in range(len(training_data)//size):
                training_datum = training_data[i * size]
                training_inputs = np.array([[_] for _ in training_datum[0]])
                training_outputs = np.array([[_] for _ in training_datum[1]])
                w, b = self.back_propagate(training_inputs, training_outputs)
                total_weights = w
                total_biases = b
                for j in range(size-1):
                    training_datum = training_data[j + i*size+1]
                    training_inputs = np.array([[_] for _ in training_datum[0]])
                    training_outputs = np.array([[_] for _ in training_datum[1]])
                    w, b = self.back_propagate(training_inputs, training_outputs)
                    total_weights = [np.add(total_weights[k], w[k]) for k in range(self.size-1)]
                    total_biases = [np.add(total_biases[k], b[k]) for k in range(self.size-1)]

                total_weights = [total_weights[k]/size for k in range(self.size-1)]
                total_biases = [total_biases[k]/size for k in range(self.size-1)]
                self.fit(total_weights, total_biases, 7.5, momentum)

            if validate != []:
                accuracies.append(self.test(validate))

        if validate != []:
            plt.plot(np.linspace(1, epochs, epochs), accuracies, label='accuracy')
            plt.show()

    def test(self, data):
        count = 0
        for datum in data:
            if np.argmax(network.get_output(datum[0])) == np.argmax(datum[1]):
                count += 1

        return count*100/len(data)

    def save_model(self):
        return

    def import_model(self):
        return

"""
network = NeuralNetwork()
network.add_layer(0, 784)
network.add_layer(1, 128)
network.add_layer(2, 10)
network.initialise()
(train_X, train_y), (test_X, test_y) = mnist.load_data()
t1 = []
for i in range(6000):
    inputs = []
    for a in range(28):
        for b in range(28):
            inputs.append(train_X[i][a, b]/100)

    outputs = [0. for _ in range(10)]
    outputs[train_y[i]] = 1.
    t1.append([inputs, outputs])

t2 = []
for i in range(1000):
    inputs = []
    for a in range(28):
        for b in range(28):
            inputs.append(test_X[i][a, b]/100)

    outputs = [0. for _ in range(10)]
    outputs[test_y[i]] = 1.
    t2.append([inputs, outputs])

print('Training...')
network.train(t1, epochs=5, size=32, algorithm='sgd', momentum=0., loss='mse', activation='sigmoid', validate=t2)
"""
