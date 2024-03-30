import numpy as np

# Activation functions
# takes in a numpy array and returns a numpy array
def ReLU(X):
    return np.maximum(0, X)

def ReLU_derived(X):
    return np.where(X <= 0, 0, 1)

def softmax(X):
    # prevent overflow
    X = X - np.max(X)
    return np.exp(X) / np.sum(np.exp(X))

def identity(X):
    return X

def identity_derived(X):
    return np.ones(X.shape)

# Loss functions
# takes in two numpy arrays (y_true, y_pred) and returns a scalar
# y_true, y_pred = [output_size x 1]
def MSE(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / y_true.shape[0]

def cross_entropy(y_true, y_pred):
    epsilon = 1e-10 # to prevent log(0)
    return -np.sum(y_true * np.log(y_pred + epsilon)) / y_true.shape[0]

# Loss functions derived
# takes in two numpy arrays (y_true, y_pred) and returns a numpy array
# y_true, y_pred = [output_size x 1], return = [output_size x 1]
def MSE_derived(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def cross_entropy_and_softmax_derived(y_true, y_pred):
    return y_pred - y_true

class LinearLayer:
    def __init__(
            self, index, input_size, output_size,
            activation_function, activation_derived_function):
        
        self.index = index
        self.input_size = input_size # number of neurons in previous layer
        self.output_size = output_size # number of neurons in this layer
        self.activation_function = activation_function
        self.activation_derived_function = activation_derived_function

        self.W = np.random.randn(output_size, input_size) # W = [output_size x input_size]
        self.b = np.random.randn(output_size, 1) # b = [output_size x 1]

        self.input = None # input(also denoted by A_prev) = [input_size x 1]
        self.Z = None # Z = [output_size x 1]
        self.output = None # output(also denoted by A_next) = [output_size x 1]

        self.dW = None
        self.db = None

    def forward(self, input):
        self.input = input
        self.Z = np.dot(self.W, input) + self.b
        self.output = self.activation_function(self.Z)
        return self.output
    
    def test_forward(self, input):
        return self.activation_function(np.dot(self.W, input) + self.b)
    
    # dLoss_dA is the derivative of Loss with respect to A of this layer
    # dLoss_dA = [output_size x 1]
    # returns dLoss_dA of previous layer (i.e. next layer in backward pass)
    def backward(self, dLoss_dA):
        # dA_dZ is a diagonal matrix
        dA_dZ = np.diagflat(self.activation_derived_function(self.Z))
        self.dW = np.dot(dA_dZ, np.dot(dLoss_dA, self.input.T))
        self.db = np.dot(dA_dZ, dLoss_dA)
        return np.dot(self.W.T, np.dot(dA_dZ, dLoss_dA))
    
    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

class Network:
    def __init__(self, neurons_per_layer, activation_functions, activation_derived_functions):
        self.layers = [
            LinearLayer(
                i,
                neurons_per_layer[i - 1],
                neurons_per_layer[i],
                activation_functions[i],
                activation_derived_functions[i],
            )
            for i in range(1, len(neurons_per_layer))
        ]

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, dLoss_dy):
        dLoss_dA = dLoss_dy
        for layer in reversed(self.layers):
            dLoss_dA = layer.backward(dLoss_dA)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def test_forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.test_forward(output)
        return output

    