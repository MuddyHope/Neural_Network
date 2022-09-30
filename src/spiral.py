#using datasets of spiral
from turtle import forward
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np



nnfs.init() 
X, y = spiral_data(1000,2)

 
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

print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)
plt.scatter(activation1.output[:,0], activation1.output[:,1], c = y, cmap = 'brg')
plt.show()

