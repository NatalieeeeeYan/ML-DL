from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import pickle

class DataManager:
    def __init__(self, bmp_path="classification/data_bmp", data_path="classification/data", ratio=(0.8, 0.1, 0.1)):
        if os.path.exists(f"{data_path}/training.pkl"):
            self.load_from_data(data_path)
        else:
            self.load_from_bmp(bmp_path)
            self.divide(*ratio)
            self.save(data_path)
        
    def load_from_bmp(self, bmp_path):
        self.data = datasets.ImageFolder(
            root=bmp_path,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ])
        )

    def save(self, data_path):
        with open(f"{data_path}/training.pkl", 'wb') as f:
            pickle.dump(self.train_data, f)
        with open(f"{data_path}/validation.pkl", 'wb') as f:
            pickle.dump(self.validate_data, f)
        with open(f"{data_path}/testing.pkl", 'wb') as f:
            pickle.dump(self.test_data, f)

    def load_from_data(self, data_path):
        self.train_data = pickle.load(open(f"{data_path}/training.pkl", 'rb'))
        self.validate_data = pickle.load(open(f"{data_path}/validation.pkl", 'rb'))
        self.test_data = pickle.load(open(f"{data_path}/testing.pkl", 'rb'))

    def divide(self, train_ratio=0.8, validate_ratio=0.1, test_ratio=0.1):
        # Divide Data
        train_size = int(train_ratio * len(self.data))
        validate_size = int(validate_ratio * len(self.data))
        test_size = len(self.data) - train_size - validate_size
        self.train_data, self.validate_data, self.test_data = random_split(
            self.data, [train_size, validate_size, test_size])
    
    def train_dataloader(self, batch_size=64, shuffle=True):
        return DataLoader(
            self.train_data, batch_size=batch_size, shuffle=shuffle)
    
    def validate_dataloader(self, batch_size=64, shuffle=False):
        return DataLoader(
            self.validate_data, batch_size=batch_size, shuffle=shuffle)
    
    def test_dataloader(self, batch_size=64, shuffle=False):
        return DataLoader(
            self.test_data, batch_size=batch_size, shuffle=shuffle)
    