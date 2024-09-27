# 训练模型
import matplotlib
 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from dataset import MyDataset
from net import SimpleConv3
import torch
import torchvision.transforms as transforms
from torch.optim import SGD  # 优化相关
from torch.optim.lr_scheduler import StepLR  # 优化相关
from sklearn import preprocessing  # 处理label
 
# 图片转换形式
traindata_transfomer = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor格式
    transforms.Resize(60, antialias=True),  # 调整图像大小，调整为高度或宽度为60像素，另一边按比例调整，antialias=True启用了抗锯齿功能
    transforms.RandomCrop(48),  # 裁剪图片，随机裁剪成高度和宽度均为48像素的部分
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对图像进行归一化处理。对每个通道执行了均值为0.5、标准差为0.5的归一化操作
])
 
if __name__ == '__main__':
    traintxt = './hymenoptera_data/train.txt'
    valtxt = './hymenoptera_data/val.txt'
 
    # 加载数据
    traindataset = MyDataset(traintxt, traindata_transfomer)
 
    # 创建卷积神经网络
    net = SimpleConv3(2)  # 二分类
    # 使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # 测试GPU是否能使用
    # print("The device is gpu later?:", next(net.parameters()).is_cuda)
    # print("The device is gpu,", next(net.parameters()).device)
 
    # 将数据提供给模型使用
    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=128, shuffle=True,
                                                  num_workers=1)  # batch_size可以自行调节
    # 优化器
    optim = SGD(net.parameters(), lr=0.1, momentum=0.9)  # 使用随机梯度下降（SGD）作为优化器，学习率0.1，动量0.9，加速梯度下降过程，lr可自行调节
    criterion = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失作为损失函数
    lr_step = StepLR(optim, step_size=200, gamma=0.1)  # 学习率调度器，动态调整学习率，每200个epoch调整一次，每次调整缩小为原来的0.1倍，step_size可自行调节
    epochs = 5  # 训练次数
    accs = []
    losss = []
    # 训练循环
    for epoch in range(0, epochs):
        batch = 0
        running_acc = 0.0  # 精度
        running_loss = 0.0  # 损失
        for data in traindataloader:
            batch += 1
            imputs, labels = data
            # 将标签从元组转换为tensor类型
            labels = preprocessing.LabelEncoder().fit_transform(labels)
            labels = torch.as_tensor(labels)
            # 利用GPU训练模型
            imputs = imputs.to(device)
            labels = labels.to(device)
            # 将数据输入至网络
            output = net(imputs)
            # 计算损失
            loss = criterion(output, labels)
            # 平均准确率
            acc = float(torch.sum(labels == torch.argmax(output, 1))) / len(imputs)
            # 累加损失和准确率，后面会除以batch
            running_acc += acc
            running_loss += loss.data.item()
 
            optim.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optim.step()  # 更新参数
 
        lr_step.step()  # 更新优化器的学习率
        # 一次训练的精度和损失
        running_acc = running_acc / batch
        running_loss = running_loss / batch
        accs.append(running_acc)
        losss.append(running_loss)
        print('epoch=' + str(epoch) + ' loss=' + str(running_loss) + ' acc=' + str(running_acc))
 
    # 保存模型
    torch.save(net, 'model.pth')  # 保存模型的权重和结构
    x = torch.randn(1, 3, 48, 48).to(device)  # # 生成一个随机的3通道、48x48大小的张量作为输入，新建的张量也要送到GPU中
    net = torch.load('model.pth')  # 从保存的.pth文件中加载模型
    net.train(False)  # 设置模型为推理模式，意味着不会进行梯度计算或反向传播
    torch.onnx.export(net, x, 'model.onnx')  # 使用ONNX格式导出模型
    # 接受模型net、示例输入x和导出的文件名model.onnx作为参数
 
    # 可视化结果
    fig = plt.figure()
    plot1, = plt.plot(range(len(accs)), accs)  # 创建一个图形对象plot1，绘制accs列表中的数据
    plot2, = plt.plot(range(len(losss)), losss)  # 创建另一个图形对象plot2，绘制losss列表中的数据
    plt.ylabel('epoch')  # 设置y轴的标签为epoch
    plt.legend(handles=[plot1, plot2], labels=['acc', 'loss'])  # 创建图例，指定图表中不同曲线的标签
    plt.show()  # 展示所绘制的图表