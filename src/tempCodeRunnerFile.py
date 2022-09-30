#now creating a class and functions within them

import numpy as np


# X is the neurons at 
X = [[1.1, 3.2, 2.5, 1.3],
          [1.0, 2.9, -2.5, 0.9],
          [0.9, -2.3, 1.1, -2.4]]



class Layer:
    
    def __init__(self, inputs, neurons):    #here neurons are the outputs after the inputs*weights
        self.weights = np.random.randn(inputs,neurons)
        pass

print(np.random(4,5))