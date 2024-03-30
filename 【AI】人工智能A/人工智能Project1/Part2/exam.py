import torch
from data_manager import *
from network import *

# path
bmp_path = "exam/bmp"
data_path = "exam/data"

# Load Data
data_manager = DataManager(
    bmp_path=bmp_path,
    data_path=data_path,
    ratio=(0, 0, 1)
)
test_data = data_manager.test_dataloader()

# Load Model
net = LeNet5(num_classes=12)
net.load_state_dict(torch.load('classification/model.pth'))
net.eval()

# Testing
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_data:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Total: {total}, Correct: {correct}, "
        f'Accuracy: {100 * correct / total:.2f} %')