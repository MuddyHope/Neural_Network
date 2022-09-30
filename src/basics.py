#we have 3 output and 1 inputs, each input (that is the hidden layer) has 3 different weights attached to it.
#each input also has 3 bias, that is added everytime it sums up.


inputs = [1.0]

weights = [0.5, 2.3, 1.4]

bias = [1]

#it should be input * weights[0] + bias to the output, we will have 3 outputs.
output = []
for i in range(len(weights)):
    temp = inputs[0] * weights[i] + bias[0]
    output.append(temp)

print(output)