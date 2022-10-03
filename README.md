# Neural_Network


1) Weights
2) Bias
3) Activation functions: Used for keeping the outputs of the layers in range and clipping the values
    3.1) ReLU (most widely used) - Rectified Linear Unit 
         max(0, input)
4) Softmax function - Used to normalize the function using exponents
    4.1) Overflow prevention - Subtracting the max_value from each batch to keep the output in range(0,1)
    4.2) Step 1 - Exponential power the input --> Math.e ** input
    4.3) Step 2 - Each exponential inputed number / Sum of all the exponential or batch values
5) Finding loss = -log(softmax_output * target_output)