import torch
from data_manager import *
from network import *

# Parameters
## Data
ratio = (0.8, 0.1, 0.1) # train, validate, test
batch_size = 8
## Train
num_classes = 12
epochs = 25

# Load Data
data_manager = DataManager(ratio=ratio)
training_data = data_manager.train_dataloader(batch_size=batch_size)
validation_data = data_manager.validate_dataloader(batch_size=batch_size)
test_data = data_manager.test_dataloader(batch_size=batch_size)

# Train
net = LeNet5(num_classes=num_classes)

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
loss_function = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Speed up
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
net.to(device)
loss_function.to(device)

for epoch in range(epochs):
    # Train
    net.train()
    for images, labels in training_data:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validate
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in validation_data:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch: {epoch}, Train Loss: {loss.item():.3f} '
              f'Validation Accuracy: {100 * correct / total:.3f} %')
        
    lr_scheduler.step()

# Save Model
torch.save(net.state_dict(), 'classification/model.pth')
