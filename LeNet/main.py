import torch
import torch.nn as nn
import torch.optim as optim
from model import LeNet
from dataset import get_dataloader
from train import train
from test import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_dataloader()

model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, train_loader, criterion, optimizer, device, epochs=10)
test(model, test_loader, device)
