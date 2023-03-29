import numpy as np
from random import uniform
from numpy import log as ln
from dataclasses import dataclass, field
from activation_function import *
from learning_model import *


@dataclass
class Neuron():
    id: int
    number_in_layer: int
    layer: int
    bias: float = 0.
    bias_w: float = 0.
    bias_delta_w: float = 0.
    local_gradient: float = 0.
    inputs: list = field(default_factory=list)
    weights: list = field(default_factory=list)

    activation_function: IActivationFunction = Sigmoid()

    def output(self) -> float:
        total = np.dot(self.weights, self.inputs) + self.bias * self.bias_w
        return self.activation_function.function(total)


@dataclass
class BaseNeuralNetwork():
    number_neurons_per_hidden_layer: list = field(default_factory=list)
    neurons: list = field(default_factory=list)
    base_neural_connections: dict = None
    max_iter: int = 6000
    y: float = 0.01
    learning_rate: float = 0.01
    alpha: float = 0.001

    activation_function: IActivationFunction = Sigmoid()
    learning_mode: ILearning_Mode = Stochastic()

    def hidden_layer_generation(self, number_input_params):
        for number_layer, number_neurons in enumerate(self.number_neurons_per_hidden_layer):
            for j in range(0, number_neurons):
                self.neurons.append(Neuron(id=len(self.neurons)+1,
                                           number_in_layer=j,
                                           layer=number_layer,
                                           inputs=np.array([]),
                                           bias=1,
                                           activation_function=self.activation_function))

    def creating_connections(self, x_i):
        for neuron in self.neurons:
            if neuron.layer == 0:
                neuron.inputs = x_i
            else:
                neuron.inputs = np.array(
                    [i.output() for i in self.neurons if i.layer == neuron.layer-1])

    def weights_generation(self, number_input_params):
        for neuron in self.neurons:
            if neuron.layer == 0:
                neuron.weights = np.array(
                    [uniform(-1, 1) for x in range(number_input_params)])
                neuron.bias_w = uniform(-1, 1)
                neuron.delta_w = np.array(
                    [0. for x in range(number_input_params)])
            else:
                neurons_from_previos_layer = [
                    neuron_from_previos_layer for neuron_from_previos_layer in self.neurons if neuron_from_previos_layer.layer == neuron.layer - 1]
                neuron.weights = np.array(
                    [uniform(-1, 1) for x in range(len(neurons_from_previos_layer))])
                neuron.bias_w = uniform(-1, 1)
                neuron.delta_w = np.array(
                    [0. for x in range(len(neurons_from_previos_layer))])

    def output_layer_generation(self):
        raise NotImplemented

    def weight_correction(self, true_y):
        raise NotImplemented

    def fit(self):
        raise NotImplemented

    def predict(self, x):
        raise NotImplemented

    def score(self):
        raise NotImplemented


class NeuralNetworkRegressor(BaseNeuralNetwork):

    def output_layer_generation(self, y):
        if self.number_neurons_per_hidden_layer == []:
            self.neurons.append(Neuron(id=len(self.neurons) + 1,
                                       number_in_layer=0,
                                       layer=0,
                                       inputs=np.array([]),
                                       bias=1,
                                       activation_function=Identity()))
        else:
            number_last_layer = len(self.number_neurons_per_hidden_layer)
            # берем все нейроны последнего скрытого слоя
            last_layer_neurons = [
                neuron for neuron in self.neurons if neuron.layer == number_last_layer-1]
            self.neurons.append(Neuron(id=len(self.neurons) + 1,
                                       number_in_layer=0,
                                       layer=number_last_layer,
                                       inputs=np.array([]),
                                       bias=1,
                                       activation_function=Identity()))

    def fit(self, x, y):
        if len(x) != len(y):
            print('Error, check x and y')
        else:
            if type(x[0]) == type(list()):
                number_input_params = len(x[0])
            else:
                number_input_params = 1
            self.hidden_layer_generation(number_input_params)
            self.output_layer_generation(y)
            self.weights_generation(number_input_params)
            for iter in range(0, self.max_iter):
                mean_error, self.neurons = self.learning_mode.weight_correction(
                    self.creating_connections, self.neurons, x, y, self.learning_rate)
                if mean_error < self.alpha:
                    break

    def predict(self, x):
        self.creating_connections(x)
        return self.neurons[-1].output()

    def score(self, x, y):
        y_pred = []
        for x_i in x:
            self.creating_connections(x_i)
            y_pred.append(self.neurons[-1].output())
        mae = np.mean(np.abs(np.array(y) - np.array(y_pred)))
        return mae
