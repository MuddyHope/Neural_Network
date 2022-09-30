import numpy as np

'''
inputs = [[1.1, 3.2, 2.5, 1.3],
          [1.0, 2.9, -2.5, 0.9],
          [0.9, -2.3, 1.1, -2.4]]


weights = [[1, 3, 4, 2],
            [1, 3.1 ,3 , 1],
            [1.5, 2.4, 1.3, 0.5]]
'''
inputs = [[1.1, 3.2, 2.5, 1.3],
          [1.9, 2.9, -2.5, 0.9]]



weights = [[1, 3, 4, 2],
            [1, 3.1 ,3 , 1]]

#now we have to transpose the weights or the inputs

output = np.dot(weights, np.array(inputs).T)

#output first row has output from both batch 1 and batch 2
print(output)