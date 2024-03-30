import numpy as np

class DataManager:
    def __init__(self, X, Y, shuffle = True):
        self.X = X  # X = [input_size x m]
        self.Y = Y # Y = [output_size x m]
        self.m = X.shape[1] # count of data
        if shuffle: self.shuffle()
        self.X_train, self.Y_train = None, None
        self.X_valid, self.Y_valid = None, None
        self.X_test, self.Y_test = None, None


    def shuffle(self):
        permutation = np.random.permutation(self.m)
        self.X = self.X[:, permutation]
        self.Y = self.Y[:, permutation]

    def divide(self, train_ratio, valid_ratio, test_ratio):
        assert train_ratio + valid_ratio + test_ratio == 1
        self.train_size = int(self.m * train_ratio)
        self.valid_size = int(self.m * valid_ratio)
        self.test_size = self.m - self.train_size - self.valid_size

        self.X_train = self.X[:, :self.train_size]
        self.Y_train = self.Y[:, :self.train_size]

        self.X_valid = self.X[:, self.train_size:self.train_size+self.valid_size]
        self.Y_valid = self.Y[:, self.train_size:self.train_size+self.valid_size]

        self.X_test = self.X[:, self.train_size+self.valid_size:]
        self.Y_test = self.Y[:, self.train_size+self.valid_size:]
    
    # # returns a generator of batches
    # # each batch is a tuple of (X_batch, Y_batch)
    # # X_batch, Y_batch = [input_size x batch_size]
    # def training_batches(self, batch_size, shuffle=True):
    #     if shuffle:
    #         self.shuffle()
    #     for i in range(0, self.m, batch_size):
    #         yield self.X_train[:, i:i+batch_size], self.Y_train[:, i:i+batch_size]

    # X_valid, Y_valid = [input_size x valid_data_size]

    def train_data(self):
        return self.X_train, self.Y_train

    def validate_data(self):
        return self.X_valid, self.Y_valid
    
    # X_test, Y_test = [input_size x test_data_size]
    def test_data(self):
        return self.X_test, self.Y_test