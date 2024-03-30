# %%
import torch
import torchvision
import torchvision.transforms as transforms

# %%
n_epochs = 5 # 模型训练 5 轮
log_interval = 30 #控制打印频率的，设 n = 30*batch_size，即n张图后打印一次进度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 根据设备是否支持GPU来选择硬件 
size = 32 # 对输入图片进行处理，拉伸为32*32的图片，这是为了复刻手写数字识别的神经网络，其输入为32*32的灰度图像
learn_rate = 0.03 # 学习率
momentum = 0.1  # 动量

# %%
!wget www.di.ens.fr/~lelarge/MNIST.tar.gz
!tar -zxvf MNIST.tar.gz
from torchvision.datasets import MNIST
transform = transforms.Compose(
    [ transforms.Resize(size), transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))]) # 正则化处理，相当于z-score
trainset = MNIST(root = './', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = MNIST(root = './', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True, num_workers=2)

# classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

# %%
examples = enumerate(testloader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_targets)
print(example_data.shape)

# %%
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  # 在输出层添加softmax函数

network = LeNet().to(DEVICE)
optimizer = optim.SGD(network.parameters(), lr=learn_rate, momentum=momentum)

# %%
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(trainloader.dataset) for i in range(n_epochs + 1)] #test_losses为横坐标，test_losses为纵坐标
test_acc = []

# %%
def train(epoch, device):
  network.train() # 调用上一步实例化对象network中的方法（该方法包内已经写好）
  for batch_idx, (data, target) in enumerate(trainloader): # 按batch_size为集合对象进行逐个处理
    data, target = data.to(device), target.to(device) # data是图片，target是标签，device是为了有GPU情况下使用GPU加速
    optimizer.zero_grad() # 开始进行BP之前将梯度设置为零，因为PyTorch会在随后的BP中累积梯度
    output = network(data) 
    loss = F.nll_loss(output, target) # 函数全称是negative log likelihood loss，下面博客有详细解释
    # https://blog.csdn.net/weixin_38145317/article/details/103288032
    loss.backward() # 根据误差进行BP
    optimizer.step()
    if batch_idx % log_interval == 0: # 控制输出频率
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(trainloader.dataset),
        100. * batch_idx / len(trainloader), loss.item()))
      train_losses.append(loss.item()) # 记录并储存train loss
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(trainloader.dataset)))

# %%
def test(device):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in testloader:
      data, target = data.to(device), target.to(device)
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1] # 找到概率最大的下标，为预测类别
      correct += pred.eq(target.data.view_as(pred)).sum() # x下面都是记录数据用于绘图，不再解释
  test_loss /= len(testloader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(testloader.dataset),
    100. * correct / len(testloader.dataset)))
  test_acc.append(correct / len(testloader.dataset))

# %%
test(DEVICE)
for epoch in range(1, n_epochs + 1):
  train(epoch, DEVICE)
  test(DEVICE)
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue') # train_losses变化曲线
plt.plot(test_counter, test_acc, color='yellow') # test集accuracy变化曲线
plt.scatter(test_counter, test_losses, color='red') # test集loss散点图
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss and test accuarcy')
plt.show()

# %%
examples = enumerate(testloader)
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
  example_data, example_targets = example_data.to(DEVICE), example_targets.to(DEVICE)
  output = network(example_data)
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0].cpu().clone().numpy(), cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
plt.show()

# %%
import os
import matplotlib.pyplot as plt

# 初始化计数器
total_images = 0
correct_predictions = 0

# 1. 遍历文件夹中的所有以png结尾的文件
image_folder = "/Users/songwenyan/FudanCourses/DeepLearning/手写体识别（Pytorch)/data/Test"
fig = plt.figure()

# 使用os.scandir获取迭代器
for i, filename in enumerate(os.scandir(image_folder), start=1):
    if ".png" in filename.name:
        image_path = os.path.join(image_folder, filename)
        answer = filename.name.split('.')[0].split('-')[-1]
        
        # 添加子图
        plt.subplot(6, 4, i)
        
        # 加载图像并进行预处理
        image = Image.open(image_path).convert("L")  # 转换为灰度图像
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 调整大小为模型所需的输入尺寸
            transforms.ToTensor(),         # 转换为张量
            transforms.Normalize((0.5), (0.5))  # 归一化
        ])
        input_image = transform(image).unsqueeze(0)  # 添加一个维度以匹配模型的输入形状

        # 2. 将图像输入模型进行推理
        output = network(input_image.to(DEVICE))
        predicted_class = output.argmax(dim=1).item()  # 获取预测的类别

        # 输出预测结果
        plt.imshow(image, cmap='gray', interpolation='none')
        plt.title(f"Prediction: {predicted_class}")
            
        # 统计正确预测数量
        total_images += 1
        if str(predicted_class) == answer:
            correct_predictions += 1
    else:
        continue

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=1.0, hspace=2.0)

# 计算正确率
accuracy = correct_predictions / total_images
print("Accuracy:", accuracy)
plt.show()



