import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np



nnfs.init() 
X, y = spiral_data(10,3)

 
#plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
#plt.show()
#print(X)


class Layer:
    
    def __init__(self, n_inputs, neurons):    #here neurons are the outputs after the inputs*weights
        self.weights = 0.1*np.random.randn(n_inputs,neurons)
        self.bias = np.zeros((1, neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias

class Activation:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

layer1 = Layer(2,3)
layer1.forward(X)
activation1 = Activation()
print("layer 1 output")
print(layer1.output)
activation1.forward(layer1.output)

print("Activation functioned output")
print(activation1.output)


#plt.scatter(activation1.output[:,0], activation1.output[:,1], c = y, cmap = 'brg')
#plt.show()



'''After the activation function, we use the softmax function to normalize'''


class Normalization:

    def softmax(self, inputs):
        #taking the activation1.output as the input

        #step 1, having the exponential values of the activation1.output
        self.exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        #step 2, sum of all the exponent values
        self.total_sum = np.sum(self.exp_values, axis = 1, keepdims=True)
        self.output = (self.exp_values/self.total_sum)


normalization1 = Normalization()
normalization1.softmax(activation1.output)




print("Normalized exponent values")
print(normalization1.exp_values)
print("Normalized output")
print(normalization1.output)

