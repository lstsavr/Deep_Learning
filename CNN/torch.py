import torch
import torch.nn as nn
import torchvision 
import torch.utils.data as Data
 
torch.manual_seed(325) 
 EPOCH = 1      
LR = 0.001      
BATCH_SIZE = 50 
DOWNLOAD_MNIST = True 
 
train_data = torchvision.datasets.MNIST(  
    root = './MNIST/',                      
    train = True,                            
    transform = torchvision.transforms.ToTensor(),                                                
    download=DOWNLOAD_MNIST 
)
 
test_data = torchvision.datasets.MNIST(root='./MNIST/', train=False)  # train设置为False表示获取测试集
 
train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
) 
test_x = torch.unsqueeze(test_data.data,dim=1).float()[:2000] / 255.0
# shape from (2000, 28, 28) to (2000, 1, 28, 28)
test_y = test_data.targets[:2000]
 
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
 
        self.conv1 = nn.Sequential(
            nn.Conv2d(                     
                in_channels=1,
                out_channels=16,           
                kernel_size=5,
                stride=1,                   
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     
        )
 
        self.conv2 = nn.Sequential(
            nn.Conv2d(                      
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),                               
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)       
        )
 
        self.out = nn.Linear(32*7*7,10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) 
        x = x.view(x.size(0),-1)              
        out = self.out(x)                    
        return out
 
cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR) 
loss_func = nn.CrossEntropyLoss() 
for epoch in range(EPOCH):
 
    for step,(batch_x,batch_y) in enumerate(train_loader):
        pred_y = cnn(batch_x)
        loss = loss_func(pred_y,batch_y)
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].numpy() 
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
 
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
