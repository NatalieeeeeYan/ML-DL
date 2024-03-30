import numpy as np
from network import *
from data_manager import *
import pickle
import matplotlib.pyplot as plt
import os

N = 10000
learning_rate = 0.005
epochs = 1000
neurons = [1, 50, 1]
activation_functions = [None, ReLU, identity]
activation_derived_functions = [None, ReLU_derived, identity_derived]
loss_function = MSE
loss_function_derived = MSE_derived

def generate_data(N):
    # generate dataset with N points
    # x is uniform in [-pi, pi], and y = sin(x)
    # X = [1 x N], Y = [1 x N]
    X = np.random.uniform(-np.pi, np.pi, (1, N))
    Y = np.sin(X)
    return (X, Y)

if os.path.exists(f"regression/data{N}.npy"):
    X, Y = np.load(f"regression/data{N}.npy")
else:
    X, Y = generate_data(N)
    np.save(f"regression/data{N}.npy", (X, Y))

# create the data manager
data_manager = DataManager(X, Y, shuffle=False)
data_manager.divide(0.9, 0.0, 0.1) # train : valid : test = 9:0:1

def train():

    # create the network
    my_network = Network(
        neurons_per_layer=neurons,
        activation_functions=activation_functions,
        activation_derived_functions=activation_derived_functions,
    )

    for epoch in range(epochs):
        # training
        X_train, Y_train = data_manager.train_data()
        for i in range(data_manager.train_size):
            x = X_train[:, i:i+1]
            y = Y_train[:, i:i+1]

            y_pred = my_network.forward(x)
            if True in np.isnan(y_pred):
                exit(0)
            
            dLoss_dy = loss_function_derived(y, y_pred)
            my_network.backward(dLoss_dy)
            my_network.update(learning_rate)
            if i == 0:
                print(f"epoch {epoch}, loss {loss_function(y, y_pred)}")

    # save the model
    with open("regression/my_sin_model.pkl", "wb") as f:
        pickle.dump(my_network, f)

def test():
    # load the model
    my_network = None
    with open("regression/my_sin_model.pkl", "rb") as f:
        my_network = pickle.load(f)
    
    # testing
    X_test, Y_test = data_manager.test_data()
    Y_pred = my_network.test_forward(X_test)
    print(f"MSE = {MSE(Y_test, Y_pred)}")
    print(f"Avg error = {np.average(np.abs(Y_test - Y_pred))}")

    # plotting
    plt.scatter(X_test, Y_test, color='blue', s=5, label='actual')
    plt.scatter(X_test, Y_pred, color='red', s=5, label='predicted')
    plt.legend()
    plt.show()
    
test()