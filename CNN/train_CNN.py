import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from dataset_loader import FacialExpressionDataset
from cnn_model import FaceCNN
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = FacialExpressionDataset('face_images', transform=transforms.ToTensor())
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)

model = FaceCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/20], Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "face_cnn.pth")
print("训练完成，模型已保存")
