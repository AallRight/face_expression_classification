import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
from PIL import Image
# from torch.utils.tensorboard import SummaryWriter

traincsv = "./traindata.csv"
testcsv = "./testdata.csv"

traindir = "./dataset/init_data/"

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# X = torch.rand(4,784)
# net = MLP()

# print(net)
# net(X)

class MyDataset(Dataset):
    def __init__(self, anotations_file, img_dir, transform=None, 
                target_transform = None):
        self.img_labels = pd.read_csv(anotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform


    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, 
                                self.img_labels.iloc[index, 0])
        # image = read_image(img_path)
        image = Image.open(img_path).convert("RGB");
        label = self.img_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        label = label
        return image, label

transform = torchvision.transforms.Compose([  
    torchvision.transforms.Resize((32, 32)),  # 将图像调整为32x32大小  
    torchvision.transforms.ToTensor(),       # 将PIL Image或NumPy ndarray转换为tensor，并归一化到[0.0, 1.0]  
]) 



# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.0005

# # MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='d:/computer vision/gkc/dataset/initdata/traindata/',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)

# test_dataset = torchvision.datasets.MNIST(root='d:/computer vision/gkc/dataset/initdata/testdata/',
#                                           train=False,
#                                           transform=transforms.ToTensor())

train_dataset = MyDataset(traincsv, traindir, transform=transform)
val_dataset = MyDataset(testcsv, traindir, transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers = 0,
                                           shuffle=True,
                                           drop_last = True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=batch_size,
                                          num_workers = 0,
                                          shuffle=False)


# class MyDataset(Dataset):
#     def __init__(self, anotations_file, img_dir, transform=None, 
#                 target_transform = None):
#         self.img_labels = pd.read_csv(anotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform


    
#     def __len__(self):
#         return len(self.img_labels)
    
#     def __getitem__(self, index):
#         img_path = os.path.join(self.img_dir, 
#                                 self.img_labels.iloc[index, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[index, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label


# MLP
# class MLP(nn.Module):
#     def __init__(self, **kwargs):
#         # super().__init__(*args, **kwargs)
#         super(MLP, self).__init__(**kwargs)
#         self.hidden = nn.Linear(784, 256)

# Convolutional neural network (two convolutional layers)
# class ConvNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ConvNet, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.fc = nn.Linear(8 * 8 * 32, num_classes)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         return out


import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4 * 4 * 64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out






model = ConvNet(num_classes).to(device)
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####
# 创建TensorBoardX的SummaryWriter对象
# writer = SummaryWriter()

# # 训练模型并记录指标到TensorBoard
# total_step = len(train_loader)
# for epoch in range(5):
#     for i, (images, labels) in enumerate(train_loader):
#         # 前向传播
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # 记录训练损失到TensorBoard
#         writer.add_scalar('Train/Loss', loss.item(), epoch * total_step + i)
        
#         if (i+1) % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(epoch+1, 5, i+1, total_step, loss.item()))

# # 关闭TensorBoardX的SummaryWriter
# writer.close()
# writer = SummaryWriter()

# 训练模型并记录指标到TensorBoard
total_step = len(train_loader)
for epoch in range(5):
    for i, (images, labels) in enumerate(train_loader):
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录训练损失到TensorBoard
        # writer.add_scalar('Train/Loss', loss.item(), epoch * total_step + i)
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, 5, i+1, total_step, loss.item()))

# 关闭TensorBoardX的SummaryWriter
# writer.close()
#####

# X = torch.rand(2,784)
# net = ConvNet()
# print(net)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
                                                                               100 * correct / total))


# Save the model checkpoint
torch.save(model.state_dict(), 'modelnew.ckpt')