# 测试整个模型的准确率
import torch
import torchvision.transforms as transforms
from dataset import MyDataset  # 您的数据集类
from sklearn import preprocessing  # 处理label
 
# 定义测试集的数据转换形式
valdata_transfomer = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor格式
    transforms.Resize(60, antialias=True),  # 调整图像大小，调整为高度或宽度为60像素，另一边按比例调整，antialias=True启用了抗锯齿功能
    transforms.CenterCrop(48),  # 中心裁剪图片，裁剪成高度和宽度均为48像素的部分
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对图像进行归一化处理。对每个通道执行了均值为0.5、标准差为0.5的归一化操作
])
 
if __name__ == '__main__':
    valtxt = './hymenoptera_data/val.txt'  # 测试集数据路径
 
    # 加载测试集数据
    valdataset = MyDataset(valtxt, valdata_transfomer)
 
    # 加载已训练好的模型，利用GPU进行测试
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.load('model.pth').to(device)
    net.eval()  # 将模型设置为评估模式，意味着不会进行梯度计算或反向传播
 
    # 使用 DataLoader 加载测试集数据
    valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=1, shuffle=False)
 
    correct = 0  # 被正确预测的样本数
    total = 0  # 测试样本数
 
    # 测试模型
    with torch.no_grad():
        for data in valdataloader:
            images, labels = data
            # 将标签从元组转换为tensor类型
            labels = preprocessing.LabelEncoder().fit_transform(labels)
            labels = torch.as_tensor(labels)
            # 利用GPU训练模型
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)  # 输入图像并获取模型预测结果
            _, predicted = torch.max(outputs.data, 1)  # 获取预测值中最大概率的索引
            total += labels.size(0)  # 累计测试样本数量
            correct += (predicted == labels).sum().item()  # 计算正确预测的样本数量
 
    # 计算并输出模型在测试集上的准确率
    accuracy = 100 * correct / total
    print('Test Accuracy: {:.2f}%'.format(accuracy))