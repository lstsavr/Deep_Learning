import pandas as pd
import os 
# 指定数据集路径

dataset_path = 'dataset/train.csv'
output_dir = 'dataset'

# 读取数据集
df = pd.read_csv(dataset_path)

# 提取 emotion 和 pixels 数据
df_emotion = df[['emotion']]
df_pixels = df[['pixels']]

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True) 

# 保存 emotion.csv 和 pixels.csv
df_emotion.to_csv(os.path.join(output_dir, 'emotion.csv'), index=False, header=False)
df_pixels.to_csv(os.path.join(output_dir, 'pixels.csv'), index=False, header=False)

print("emotion.csv 和 pixels.csv 生成成功！")


