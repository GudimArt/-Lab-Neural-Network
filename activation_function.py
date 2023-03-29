import numpy as np
from abc import ABC, abstractmethod
import math


class IActivationFunction(ABC):
    @abstractmethod
    def function(self, x) -> float:
        pass

    @abstractmethod
    def derivative_function(self, x) -> float:
        pass


class Sigmoid(IActivationFunction):
    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_function(self, x):
        return x * (1 - x)


class Tanh(IActivationFunction):
    def function(self, x):
        return 2/(1+np.exp(-2*x))-1
        # return math.tanh(x)

    def derivative_function(self, x):
        return 1-self.function(x)**2


class ArcTan(IActivationFunction):
    def function(self, x):
        return math.atan(x)

    def derivative_function(self, x):
        return 1/(x**2+1)


class Binary(IActivationFunction):
    def function(self, x):
        if x < 0:
            return 0
        return 1

    def derivative_function(self, x):
        return 0


class SoftPlus(IActivationFunction):
    def function(self, x):
        return np.log(1+np.exp(x))

    def derivative_function(self, x):
        return 1/(1+np.exp(-x))


class ReLU(IActivationFunction):
    def function(self, x):
        if x < 0:
            return 0
        else:
            return x

    def derivative_function(self, x):
        if x >= 0:
            return 1
        else:
            return 0


class Identity(IActivationFunction):
    def function(self, x):
        return x

    def derivative_function(self, x):
        return 1.0
