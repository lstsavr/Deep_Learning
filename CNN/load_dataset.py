import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np
from torchvision import transforms

class FacialExpressionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(os.path.join(image_dir, 'image_emotion.csv'), header=None, names=['path', 'emotion'])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.data.iloc[index, 0])
        emotion_label = int(self.data.iloc[index, 1])

        # 读取图片并转换为灰度
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 48))  # 确保尺寸一致

        # 归一化到 0-1 之间
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # 调整为 (1, 48, 48) 适配 CNN

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(emotion_label, dtype=torch.long)

# 测试数据集
if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = FacialExpressionDataset('face_images', transform=transform)
    print(f"数据集加载成功，共 {len(dataset)} 张图片！")
