import pandas as pd
import os 

dataset_path = 'dataset/train.csv'
output_dir = 'dataset'

df = pd.read_csv(dataset_path)

df_emotion = df[['emotion']]
df_pixels = df[['pixels']]

os.makedirs(output_dir, exist_ok=True) 

df_emotion.to_csv(os.path.join(output_dir, 'emotion.csv'), index=False, header=False)
df_pixels.to_csv(os.path.join(output_dir, 'pixels.csv'), index=False, header=False)

print("emotion.csv and pixels.csv generate successfully！")


