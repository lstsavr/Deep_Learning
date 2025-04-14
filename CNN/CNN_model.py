import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            #Conv2d是卷积层，nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
            #in_channels是输入图像的通道数，out_channels是输出的特征图的个数, kernel_size是卷积核大小, stride是步长，决定卷积核的移动速度
            #padding是边缘填充像素数，卷积核是一个小的矩阵，在图像上滑动，对图像进行特征提取
            nn.BatchNorm2d(64),
            #BatchNorm是批量归一化
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),#把4*4变成 2*2这种类似的，每块只保留最大值
            #kernel_size是池化窗口的大小，stride是步长，padding是边缘补零的像素数，默认是不补
        )
#输入尺寸若为1*48*48，此处之后输出为64*48*48
        self.conv2 = nn.Sequential( #对来自上一层的 [Batch, 64, H, W]进行卷积
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(#介个是fully connection layers)
            nn.Dropout(0.2),#正则化，随机讲20%的神经元屏蔽掉
            nn.Linear(256 * 6 * 6, 4096),#经过前三个卷积层，输出的图的size是[batch,256,6,6] 48/2**3 = 6
            nn.RReLU(inplace=True),#增加非线性能力
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),#这块更容易过拟合，因为数据有点多 故上面用dropout
            nn.RReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.RReLU(inplace=True),
            nn.Linear(256, 7),#最后输出7个神经元，这7个分别对应的是7个情绪类别
        )

        self.apply(gaussian_weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1#这里的view是为了flatten，然后输入给全连接层，x.shape[0]是Batch size)
        return self.fc(x)

def gaussian_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):#判断当前传进来的这层是卷积层还是全连接层
        nn.init.normal_(m.weight, mean=0, std=0.02)#用正态分布来修改这层的权重
        if m.bias is not None:#如果有bias，就把他初始化为0
            nn.init.constant_(m.bias, 0)

def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = FaceCNN().to(device)
    #train_dataset	数据集（已经定义好的）batch_size	每次送入几张图进行训练 shuffle=True	每个 epoch 开始时打乱数据顺序（防止模型记住顺序）
    #num_workers=4	用 4 个线程并行加载数据（提速）pin_memory=True 如果用 GPU，加速数据从 CPU → GPU 的搬运

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
#weight_decay=wt_decay是正则化，L2正则化的项，w = w - lr * (∇L + weight_decay * w)
#weight_decay 是告诉模型：参数不要太极端，惩罚大的权重，Dropout 是随机屏蔽神经元输出，打乱依赖关系


    for epoch in range(epochs):
        model.train() #训练模式，启用dropout和batchnorm
        epoch_loss = 0 #初始化这一轮的总loss

        for images, emotion in train_loader:
            images, emotion = images.to(device), emotion.to(device)

            optimizer.zero_grad() #清空上一次残留的梯度
            output = model(images) #向前传播，预测
            loss = loss_function(output, emotion) #计算损失
            loss.backward() #向后传播
            optimizer.step() #更新参数

            epoch_loss += loss.item()

        print(f'After {epoch+1} epochs, the loss rate is: {epoch_loss / len(train_loader):.4f}')

        if epoch % 5 == 0:#每训练五轮， 就进行一次模型评估，计算准确率
            model.eval()
            acc_train = validate(model, train_dataset, batch_size, device)
            acc_val = validate(model, val_dataset, batch_size, device)
            print(f'After {epoch+1} epochs, train accuracy: {acc_train:.4f}, val accuracy: {acc_val:.4f}')

    return model

def validate(model, dataset, batch_size, device):
    model.eval()
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
