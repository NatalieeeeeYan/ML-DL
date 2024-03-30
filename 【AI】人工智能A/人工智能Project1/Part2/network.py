import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=12, dropout=0):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.subsampling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU()
        )
        self.subsampling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(dropout)
        self.fc5 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc6 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc7 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.subsampling2(x)
        x = self.conv3(x)
        x = self.subsampling4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.dropout(x)
        x = self.fc6(x)
        x = self.fc7(x)
        return x
