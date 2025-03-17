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
            nn.BatchNorm2d(64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
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

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256 * 6 * 6, 4096),
            nn.RReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.RReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.RReLU(inplace=True),
            nn.Linear(256, 7),
        )

        self.apply(gaussian_weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

def gaussian_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = FaceCNN().to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for images, emotion in train_loader:
            images, emotion = images.to(device), emotion.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, emotion)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'After {epoch+1} epochs, the loss rate is: {epoch_loss / len(train_loader):.4f}')

        if epoch % 5 == 0:
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
