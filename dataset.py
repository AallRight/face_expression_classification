# 读取数据
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset 
import pandas as pd
import os
from torchvision.io import read_image
 
# 读取数据类
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
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



# 用于测试
if __name__ == '__main__':
    # 利用txt文件读取图片信息，txt文件包括图片路径和标签
    traintxt = './hymenoptera_data/train.txt'
    valtxt = './hymenoptera_data/val.txt'
    # 图片转换形式
    traindata_transfomer = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor格式
        transforms.Resize(60),  # 调整图像大小，调整为高度或宽度为60像素，另一边按比例调整
        transforms.RandomCrop(48),  # 裁剪图片，随机裁剪成高度和宽度均为48像素的部分
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对图像进行归一化处理。对每个通道执行了均值为0.5、标准差为0.5的归一化操作
    ])
    valdata_transfomer = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor格
        transforms.Resize(48),  # 调整图像大小，调整为高度或宽度为48像素，另一边按比例调整
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载数据
    traindataset = MyDataset(traintxt, traindata_transfomer)
    valdataset = MyDataset(valtxt, valdata_transfomer)
    print("测试集：" + str(traindataset.__len__()))
    print("训练集：" + str(valdataset.__len__()))