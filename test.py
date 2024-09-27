import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# X = torch.rand(4,784)
# net = MLP()

# print(net)
# net(X)




# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='d:/computer vision/gkc/dataset/initdata/traindata/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='d:/computer vision/gkc/dataset/initdata/testdata/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


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
        image = read_image(img_path)
        label = self.img_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == '__main__':
    data_test = MyDataset('d:/computer vision/gkc/dataset/initdata/0none/', 'd:/computer vision/gkc/dataset/initdata/0none/')
    print(data_test.img_labels)