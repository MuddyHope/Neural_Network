import numpy as np 

#we use dot product

inputs = [1.1, 3.2, 2.5, 1.3]

weights = [[1, 3, 4, 2],
            [1, 3.1 ,3 , 1],
            [1.5, 2.4, 1.3, 0.5]]

biases = [1, 3, 1.4]


output = np.dot(weights, inputs) + biases

print(output)
