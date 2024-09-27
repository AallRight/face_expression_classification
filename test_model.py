import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# 定义模型类
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
        self.fc = nn.Linear(8 * 8 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# 加载已经训练好的模型参数
model = ConvNet()
model.load_state_dict(torch.load('model.ckpt'))
model.eval()  # 设置为评估模式

# 图像预处理函数
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 添加 batch 维度
    return image

# 模型预测函数
def predict_image(image_path, model):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

# 测试集文件夹路径
test_dir = 'D:/computer vision/gkc/test'

# 遍历测试集中的每个表情文件夹
for emotion_dir in os.listdir(test_dir):
    if os.path.isdir(os.path.join(test_dir, emotion_dir)):
        print(f"Predictions for Emotion: {emotion_dir}")
        for filename in os.listdir(os.path.join(test_dir, emotion_dir)):
            if filename.endswith(".jpg"):  # 只处理 jpg 图像
                image_path = os.path.join(test_dir, emotion_dir, filename)
                prediction = predict_image(image_path, model)
                print(f"Image: {filename}, Predicted Class: {prediction}")
        print("--------------------")
