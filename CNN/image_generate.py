import numpy as np
import pandas as pd
import cv2
import os

image_output_dir = 'face_images'
os.makedirs(image_output_dir, exist_ok=True)

data = pd.read_csv('dataset/pixels.csv', header=None)
data = np.array([list(map(int, row[0].split())) for row in data.values])

for i in range(data.shape[0]):
    face_array = data[i, :].reshape((48, 48))  
    image_path = os.path.join(image_output_dir, f'{i}.jpg')  
    cv2.imwrite(image_path, face_array)  

print(f"convert {data.shape[0]} pictures successfully，save in {image_output_dir}")
