import numpy as np

class Neural_Network: 

    def __init__(self):

        self.hidden_weights = np.random.uniform(size=(2, 4))
        # the reason for including this line is to create a matrix, 2x4, representing 2 input neurons & 4 hidden neurons
        # this matrix is filled with random values between 0 and 1 (by default) following a normal distribution
        self.hidden_bias = np.random.uniform(size = (4,))
        # this is a vector of length 4 representing each hidden neuron
        # represents the bias, which is the output when there is no input (y = mx + c, where c is the bias term)
        # As an easy way to understand bias, think of it like this: 
        # Let's say we have a model that takes temperature inputs to predict ice-cream sales. Even at 0 degrees celsius, some ice cream still sells.
        # Our model, with bias, will not kill the output, as a 0 degree temperature is a perfectly normal scenario. 
        # It shifts the activation function so that the model can learn meaningful patters in the full range of possible inputs
        # activation function: (inputs * weights) + bias
        # Everything after the input gets bias term
        self.output_weights = np.random.uniform(size = (4, 1))
        # This represents the output weights, where the output is connected to the hidden neurons
        # In this case, we assume there to be only one output value, hence the one column for each neuron. 
        self.output_bias = np.random.uniform(size = (1,))
        # This represents the output bias, which in this case is just one term, for one output
        # CLARIFICATION: In python, size = (1,) & size = (1,1) are different such that the former is just 1 value while the latter is 1 value corresponding to another (1x1 matrix)

    def sigmoid(self, x): 
            # the purpose of this function or a sigmoid function in general, is to squish any number into range 0 thru 1. 
            # We see 2 parameters here, self & x. The reason for self is to be able to access any method or variable anywhere else in the class. And obviously, x. 
            x = np.clip(x, -500, 500)  # Prevents extreme values
            return 1 / (1 + np.exp(-x))
            # Note: exp(x) is e^x, so changing the value of x to anything, one can know what is e to that number

    def sigmoid_derivative(self, x):
            # the purpose of this function: 
            # During backpropagation, the input x to the sigmoid function changes as the network adjusts weights and biases to minimize error. 
            # Consider this, the network increases or decreases the input given to a certain neuron to reflect its "weight" in terms of the whole model.
            # This is then accurately reflected and permanently adjusted in the NN during backprop (via this sigmoid derivitive method)
            epsilon = 1e-10 # to avoid divide by 0
            return x * (1 - x)
            # This formula is derived from: sigma = 1  / 1 + e^-x --> sigma' = x * (1-x)
       
    def forward(self, x):
            # x is my input data (vector of size 2) which essentially represents the 2 rows of the hidden weights matrix
            # The purpose of this method is forward propagate thru the Neural Network from the input to the output passing thru all the hidden layers
            # We are just showing the big picture of what is going on here
            self.hidden_sum = np.dot(x, self.hidden_weights) + self.hidden_bias
            # this is calculating the weighted sum of the hidden layers thru multiplying an input vector (representing the output from the 2 input neurons) and the hidden weights matrix and then adding the bias
            # once we calculate this sum matrix, we can apply the sigmoid function to put all of these values in the matrix between 0 and 1. 
            self.hidden_output = self.sigmoid(self.hidden_sum)
            # essentially what this function does
            # Now we can essentially do this for the outer layer as well, starting by taking hidden's output as our new "input"
            self.output_sum = np.dot(self.hidden_output, self.output_weights) + self.output_bias
            # Now we can calculate the output's output which is the final result
            self.output = self.sigmoid(self.output_sum)

            return self.output #Forward-Prop
        
    # CORRECTION ON DELTA VARIABLES: WE USE * AND NOT / BECAUSE THE LARGER THE DELTA THE MORE SENSITIVE AND THE LARGER THE ADJUSTMENT WILL BE
    # THIS IS CONVENTIONAL FOR MOST NEURAL NETWORKS (IGNORE COMMENTS THAT SAY OTHERWISE)
    def backward(self, x, y, learning_rate = 0.5):
            # this is the backprop representation, where x is the input, y is the expected output, and learning rate is an optional variable used for real-life representation of a NN.
            # this learning rate is set as a default value, no need to specify when calling function
            # here, we use the sigmoid derivitive function. We multiply the error (expected - actual) * sigmoid_derivitive(output) because we want to adjust for the sensitivity of the output to a change in weights
            # A larger derivitive means we have a larger delta and thus we need overall smaller weight adjustments to make up for it.
            output_error = y - self.output
            epsilon = 1e-10

            output_delta = output_error * (self.sigmoid_derivative(self.output) + epsilon)
            # Here we can directly see the greater the derivitive the smaller the adjustment needed, due to high sensitivity. 
            # Forward propagation uses and updates class attributes (self variables) that store the network's state, while backpropagation uses temporary local variables to calculate how to update those class attributes.
            hidden_error = np.dot(output_delta, self.output_weights.T)
            # Lets understand this dot product. Here, output_weights is a 4x1 matrix consisting of 4 rows for each hidden neuron and 1 column for the output neuron. Then we transpose this.
            # We multiply output_delta (sensitivity factor) by each value in this matrix (dot product)
            # By multiplying the senstivity factor by the original weights, we can see how much (to a scale) each hidden neuron contributed to the hidden error
            # Note: this is different from the output error calculation as this gives a matrix that shows how much each neuron contributes to the error, not the expected - actual calculation.
            hidden_delta = hidden_error * (self.sigmoid_derivative(self.hidden_output) + epsilon)
            # delta here is a matrix, showing the proportional adjustment for each hidden neuron weight leading INTO the layer
            # this means that the matrix of input neurons x hidden neurons will be adjusted according to that scale
            # the following section of this method updates the weights and biases according to the calculated delta.
            self.output_weights += learning_rate * np.dot(self.hidden_output.reshape(-1,1), output_delta.reshape(1,-1))
            # Note that without the learning rate: The weights would change by the full calculated adjustment amount each time, likely causing the network to overshoot optimal values and making training unstable.
            # Therefore, we can use a learning rate to decrease the magnitude of our adjustments each time so that we do not overshoot
            # we multiply the hidden_output (the influence of each neuron) by output_delta (how much output needs to change) in order to figure out how much to adjust weights by.
            self.output_bias += learning_rate * output_delta
            # consider a scenario where input is 0, therefore weights wont matter, and thus the bias will be directly equal to the desired proportional change in output, thus we use this equation
            self.hidden_weights += learning_rate * np.dot(x.reshape(-1,1), hidden_delta.reshape(1,-1))
            # these two lines (above and below) follow the same logic as before
            self.hidden_bias += learning_rate * hidden_delta

            return np.sum(output_error ** 2)
            # here we are returning the squared output_error, and the only purpose of this is to track the positive error in the network's current state
            # thru many iterations of adjustments and forward/back prop, this error should ideally approach 0 in a well-trained model, considering all other factors go well.
        
if __name__ == '__main__':
            # "==" in python compares actual values, not references like in java and c++
            # this is the main method, code written here is executed
            # here we present our training data: x & y, which represent the input and the expected output. 
            X = np.array([[0,0], [0,1], [1,0], [1,1]])
            # Widespread convention to use a capital 'X' for input and y as expected output prior to training
            # there can be as many rows as we want in this training input. However, there must be only 2 columns in this matrix because we have 2 input neurons
            y = np.array([[0], [1], [1], [0]])
            # we have 4 different input-output scenarios in our training data. the method of picking these values is thru XOR process:
            # [0,0] → 0  (neither input is 1)
            # [0,1] → 1  (exactly one input is 1)
            # [1,0] → 1  (exactly one input is 1)
            # [1,1] → 0  (both inputs are 1)
            # what this does is that it prevents a linear system to separate input into outputs, we would need an accurate model to represent this complex data
            # this is very good to test network efficiency.
            nn = Neural_Network()
            # Create an instance of the class to execute the training process
            for epoch in range(10000):
                # Like a typical for loop, iterates 10000 times, with an epoch being 1 singular iteration over the entire training dataset
                total_error = 0
                # reason for resetting the total error after each epoch is because input is same, so doesn't make sense to use prev error
                for i in range(len(X)):
                    output = nn.forward(X[i])
                    total_error += nn.backward(X[i], y[i])
                
                if epoch % 1000 == 0:
                    # the 'f' in print allows me to use curly braces to include expressions in my print statement
                    print(f"Epoch {epoch}, Error: {total_error}")

            # Once our training is done, the following simple mechanism describes how one would 'test' this NN
            print("\nTesting XOR Outputs:")
            for i in range(len(X)):
                prediction = nn.forward(X[i])
                print(f"Input: {X[i]}, Predicted: {prediction[0]:.4f}, Actual: {y[i][0]}")




