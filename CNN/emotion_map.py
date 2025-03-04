import os
import pandas as pd

def image_emotion_mapping(image_dir):
    # 读取 emotion.csv
    df_emotion = pd.read_csv('dataset/emotion.csv', header=None)

    # 获取所有图片文件名
    files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    
    path_list = []
    emotion_list = []

    for file_name in files:
        index = int(os.path.splitext(file_name)[0])  # 获取图片编号
        path_list.append(file_name)
        emotion_list.append(df_emotion.iat[index, 0])  # 查 emotion

    # 生成 DataFrame 并保存
    df_result = pd.DataFrame({'path': path_list, 'emotion': emotion_list})
    df_result.to_csv(os.path.join(image_dir, 'image_emotion.csv'), index=False, header=False)

    print(f"{image_dir} 的 image_emotion.csv 生成成功！")

# 运行
image_emotion_mapping('face_images')
