import numpy as np
from network import *
from data_manager import *
import pickle
from PIL import Image
import os

learning_rate = 0.05
epochs = 5000
neurons = [784, 256, 64, 12]
activation_functions = [None, ReLU, ReLU, softmax]
activation_derived_functions = [None, ReLU_derived, ReLU_derived, identity_derived]
loss_function = cross_entropy
loss_function_derived = cross_entropy_and_softmax_derived

def generate_data(path="classification/data_bmp"):
    # 博0 学1 笃2 志3
    # 切4 问5 近6 思7
    # 自8 由9 无10 用11
    X, Y = [], []
    for i in range(12):
        folder = f"{path}/{i+1}"
        for filename in os.listdir(folder):
            # turn img into (784, 1) array and normalize
            img = Image.open(f"{folder}/{filename}")
            img = img.convert("L")
            img = np.array(img)
            img = img.reshape((-1, 1))
            img = (img - np.min(img)) / ((np.max(img) - np.min(img)) * 255)
            # append to X and Y
            X.append(img)
            Y.append([int(x == i) for x in range(12)])

    X = np.concatenate(X, axis=1)
    Y = np.array(Y).T
    # shuffle X and Y
    permutation = np.random.permutation(X.shape[1])
    X = X[:, permutation]
    Y = Y[:, permutation]
    return (X, Y)

if os.path.exists("classification/dataX.npy"):
    X = np.load("classification/dataX.npy")
    Y = np.load("classification/dataY.npy")
else:
    X, Y = generate_data()
    np.save("classification/dataX.npy", X)
    np.save("classification/dataY.npy", Y)

data_manager = DataManager(X, Y, shuffle=False)
data_manager.divide(0.9, 0.0, 0.1)

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
                print(y_pred)
                exit(0)
            
            dLoss_dy = loss_function_derived(y, y_pred)
            my_network.backward(dLoss_dy)
            my_network.update(learning_rate)
            if i == 0:
                print(f"epoch {epoch}")

    # save the model
    with open("classification/my_model.pkl", "wb") as f:
        pickle.dump(my_network, f)

def test():
    # load the model
    my_network = None
    with open("classification/my_model.pkl", "rb") as f:
        my_network = pickle.load(f)
    
    # testing
    X_test, Y_true = data_manager.test_data()
    Y_pred = my_network.forward(X_test)
    true = np.argmax(Y_true, axis=0)
    pred = np.argmax(Y_pred, axis=0)
    correct = np.sum(true == pred)
    total = len(pred)
    print(f"correct: {correct}, total: {total}, accuracy: {correct / total * 100: .2f}%")

test()