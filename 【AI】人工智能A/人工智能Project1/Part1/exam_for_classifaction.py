import numpy as np
from network import *
from data_manager import *
import pickle
from PIL import Image
import os

# testing dataset path
path = "exam_data"

def generate_data(path):
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

X, Y = generate_data(path)
data_manager = DataManager(X, Y, shuffle=False)
data_manager.divide(0, 0, 1)

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
