# 三层卷积神经网络
import torch
 
 
# 卷积神经网络类
class SimpleConv3(torch.nn.Module):  # 继承创建神经网络的基类
    def __init__(self, classes):
        super(SimpleConv3, self).__init__()
        # 卷积层
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 2, 1)  # 输入通道3，输出通道16，3*3的卷积核，步长2，边缘填充1
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 2, 1)  # 输入通道16，输出通道32，3*3的卷积核，步长2，边缘填充1
        self.conv3 = torch.nn.Conv2d(32, 64, 3, 2, 1)  # 输入通道32，输出通道64，3*3的卷积核，步长2，边缘填充1
        # 全连接层
        self.fc1 = torch.nn.Linear(2304, 100)
        self.fc2 = torch.nn.Linear(100, classes)
 
    def forward(self, x):
        # 第一次卷积
        x = torch.nn.functional.relu(self.conv1(x))  # relu为激活函数
        # 第二次卷积
        x = torch.nn.functional.relu(self.conv2(x))
        # 第三次卷积
        x = torch.nn.functional.relu(self.conv3(x))
        # 展开成一维向量
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
 
# 用于测试
if __name__ == '__main__':
    inputs = torch.rand((1, 3, 48, 48))  # 生成一个随机的3通道、48x48大小的张量作为输入
    net = SimpleConv3(2)  # 二分类
    output = net(inputs)
    print(output)