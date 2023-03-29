import numpy as np
from abc import ABC, abstractmethod


class ILearning_Mode(ABC):
    @abstractmethod
    def weight_correction(neurons, x, y):
        pass

class Stochastic(ILearning_Mode):
    def weight_correction(self, creating_connections, neurons, x, y, sigma):
        error = []
        for i, x_i in enumerate(x):
            creating_connections(x_i)
            true_y = y[i]
            # Reversed нужен для того, чтобы идти в обратном порядке. То есть из нейрона 8 в нейрон 7, а не в 6 и начинать надо не с начала
            for neuron in reversed(neurons):
                # предыдущий слой относительно обратного обхода
                neurons_prev_layer = [neuron_prev_layer for neuron_prev_layer in reversed(
                    neurons) if neuron_prev_layer.layer == neuron.layer+1]
                if neuron.layer == neurons[-1].layer:
                    error.append((neuron.output() - true_y)**2)
                    neuron.local_gradient = (true_y - neuron.output()) * neuron.activation_function.derivative_function(
                        neuron.output())  # Дельта ошибки для выходного нейрон
                else:
                    for neuron_prev_layer in neurons_prev_layer:
                        neuron.local_gradient += (
                            neuron_prev_layer.weights[neuron.number_in_layer]) * neuron_prev_layer.local_gradient
                    neuron.local_gradient = neuron.local_gradient * \
                        neuron.activation_function.derivative_function(
                            neuron.output())
                vrem_inputs = np.array(neuron.inputs).astype(float)
                neuron.delta_w = vrem_inputs * neuron.local_gradient * sigma
                neuron.bias_delta_w = (neuron.bias * neuron.local_gradient * sigma)
                neuron.bias_w = neuron.bias_w + neuron.bias_delta_w
                neuron.weights = neuron.weights + neuron.delta_w
        mean_error = sum(error)/len(x)
        return mean_error, neurons


class FullPackage(ILearning_Mode):
    def weight_correction(self, creating_connections, neurons, x, y, sigma):
        error = []
        for i, x_i in enumerate(x):
            creating_connections(x_i)
            true_y = y[i]
            # Reversed нужен для того, чтобы идти в обратном порядке. То есть из нейрона 8 в нейрон 7, а не в 6 и начинать надо не с начала
            for neuron in reversed(neurons):
                # предыдущий слой относительно обратного обхода
                neurons_prev_layer = [neuron_prev_layer for neuron_prev_layer in reversed(
                    neurons) if neuron_prev_layer.layer == neuron.layer+1]
                if neuron.layer == neurons[-1].layer:
                    error.append((neuron.output() - true_y)**2)
                    neuron.local_gradient = (true_y - neuron.output()) * neuron.activation_function.derivative_function(
                        neuron.output())  # Дельта ошибки для выходного нейрон
                else:
                    for neuron_prev_layer in neurons_prev_layer:
                        neuron.local_gradient += (
                            neuron_prev_layer.weights[neuron.number_in_layer]) * neuron_prev_layer.local_gradient
                    neuron.local_gradient = neuron.local_gradient * \
                        neuron.activation_function.derivative_function(
                            neuron.output())
                vrem_inputs = np.array(neuron.inputs).astype(float)
                neuron.delta_w += vrem_inputs * neuron.local_gradient * sigma
                neuron.bias_delta_w += (neuron.bias * neuron.local_gradient * sigma)
        for neuron in reversed(neurons):
            neuron.delta_w /= len(x)
            neuron.bias_delta_w /= len(x)
            neuron.bias_w = neuron.bias_w + neuron.bias_delta_w
            neuron.weights = neuron.weights + neuron.delta_w
            neuron.bias_delta_w = 0 # обнуляем накопление
            neuron.delta_w *= 0  # обнуляем накопление
        mean_error = sum(error)/len(x)
        return mean_error, neurons
