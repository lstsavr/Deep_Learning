import numpy as np
import pandas as pd
import cv2
import os

# 设置存储图片的路径
image_output_dir = 'face_images'
os.makedirs(image_output_dir, exist_ok=True)

# 读取 pixels.csv 并解析
data = pd.read_csv('dataset/pixels.csv', header=None)
data = np.array([list(map(int, row[0].split())) for row in data.values])  # 转换为数值

# 生成 48x48 图片
for i in range(data.shape[0]):
    face_array = data[i, :].reshape((48, 48))  # 变形
    image_path = os.path.join(image_output_dir, f'{i}.jpg')  # 文件名
    cv2.imwrite(image_path, face_array)  # 保存图片

print(f"成功转换 {data.shape[0]} 张图片，存入 {image_output_dir}")
