import torch
import cv2
import numpy as np
from cnn_model import FaceCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FaceCNN().to(device)
model.load_state_dict(torch.load("face_cnn.pth"))
model.eval()

# 读取测试图片
image_path = "？？？？？"  # 这个路径是你自己输入的，自己输入你想检测的图片的路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (48, 48))
image = image.astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)  # (1, 48, 48)
image = np.expand_dims(image, axis=0)  # (1, 1, 48, 48)

image_tensor = torch.tensor(image, dtype=torch.float32).to(device)

with torch.no_grad():
    output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)

print(f"图片 {image_path} 的预测结果: {predicted_class.item()}")
