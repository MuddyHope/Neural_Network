#now creating a class and functions within them

import numpy as np


# X is the neurons at 
X = [[1.1, 3.2, 2.5, 1.3],
          [1.0, 2.9, -2.5, 0.9],
          [0.9, -2.3, 1.1, -2.4]]



class Layer:
    
    def __init__(self, n_inputs, neurons):    #here neurons are the outputs after the inputs*weights
        self.weights = 0.1*np.random.randn(n_inputs,neurons)
        self.bias = np.zeros((1, neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias


"""X1 = np.array(X)
print(0.1*np.random.randn(X1,5))"""

layer1 = Layer(4,5)
layer2 = Layer(5,2)
layer3 = Layer(2,1)
#print(layer1.weights)
#print(layer1.bias)

layer1.forward(X)
layer2.forward(layer1.output)
layer3.forward(layer2.output)

print(layer1.output)
print(layer2.output)
print(layer3.output)
