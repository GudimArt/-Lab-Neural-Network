import my_neural_network
from activation_function import *
from learning_model import *

if __name__ == "__main__":
  
    x = [[0,0],[0,1],[1,0],[1,1]]
    y = [0,1,1,0]
    x_test = [[0,1],[1,0]]
    y_test = [1,1]
    
    test_network_regressor= my_neural_network.NeuralNetworkRegressor(activation_function=Sigmoid(), number_neurons_per_hidden_layer=[10])
    test_network_regressor.fit(x,y)
    print(test_network_regressor.score(x_test,y_test))
    