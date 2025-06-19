import os
import pandas as pd
import json
import pickle
import cv2  # 导入OpenCV库
from tqdm import tqdm
import numpy as np

# --- 步骤 1: 预处理DataFrame以实现高效查找 ---

print("正在读取并预处理CSV文件...")
# 读取CSV文件
df = pd.read_csv("./BIG_picked/FSC_fish_BIG_CSV.csv")

# 核心优化：创建一个新的'filename'列，并将其设置为索引
try:
    df['filename'] = df['image'].apply(lambda path: path.split("\\")[-1].split("-", 1)[1])
    df.set_index('filename', inplace=True)
except (KeyError, IndexError) as e:
    print(f"处理 'image' 列时出错: {e}")
    print("请确保CSV文件格式正确，包含'image'列且路径格式符合预期。")
    exit()

print("CSV预处理完成，DataFrame已准备好进行快速查找。")


# --- 步骤 2: 定义一个优化的函数来获取所有图像及其关键点 ---

def get_all_image_paths_optimized(root_dir, dataframe):
    """
    高效地递归查找目录中的所有图像文件，
    利用预处理过的DataFrame快速匹配关键点，并显示进度条。
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    result_dict = {}
    image_files_to_process = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files_to_process.append(os.path.join(root, file))

    print(f"在 '{root_dir}' 中找到 {len(image_files_to_process)} 个图像文件，开始提取关键点...")

    for full_path in tqdm(image_files_to_process, desc="处理图像中"):
        filename = os.path.basename(full_path)
        if filename in dataframe.index:
            try:
                keypoint_str = dataframe.loc[filename, 'keypoint_label']
                keypoints_json = json.loads(keypoint_str)
                keypoints = [
                    [int(point['x'] / 100 * 1920), int(point['y'] / 100 * 1080)]
                    for point in keypoints_json
                ]
                if keypoints:
                    result_dict[full_path] = keypoints
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    return result_dict


# --- 步骤 3: 执行优化后的脚本 ---

root_directory = r"/data/zxy/pycharm/Adaptation_SAM_PM/BIG_picked/frames"
result_dict = get_all_image_paths_optimized(root_directory, df)

# --- 步骤 4: 保存结果 ---

output_file = r"keypoints.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(result_dict, f)

print(f"\n处理完成。共找到 {len(result_dict)} 个带有关建点的条目。")
print(f"结果已保存至: {output_file}")

# --- 步骤 5: 验证 - 绘制关键点并保存图像 ---

print("\n--- 开始验证：绘制关键点到图像上 ---")

# 创建一个用于存放验证结果的文件夹
verification_output_dir = "verification_images"
os.makedirs(verification_output_dir, exist_ok=True)
print(f"验证图像将被保存在 '{verification_output_dir}' 文件夹中。")

# 设置要验证的图像数量
num_images_to_verify = 5
count = 0

for img_path, keypoints in result_dict.items():
    if count >= num_images_to_verify:
        print(f"\n已生成 {num_images_to_verify} 张验证图像。")
        break

    try:
        # 读取图像
        # 使用cv2.imdecode处理可能包含非ASCII字符的路径
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            print(f"警告：无法读取图像文件 {img_path}")
            continue

        # 遍历该图像的所有关键点并在图上绘制
        for i, (x, y) in enumerate(keypoints):
            # 绘制一个半径为5的绿色实心圆
            cv2.circle(img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            # 在点旁边绘制点的索引号，方便对照
            cv2.putText(img, str(i + 1), (x + 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 构建输出文件的路径
        base_filename = os.path.basename(img_path)
        output_path = os.path.join(verification_output_dir, base_filename)

        # 保存带有点的图像
        # 使用cv2.imencode来处理非ASCII文件名
        is_success, im_buf_arr = cv2.imencode(f".{base_filename.split('.')[-1]}", img)
        if is_success:
            im_buf_arr.tofile(output_path)
            print(f"已生成验证图像: {output_path}")
        else:
            print(f"警告: 无法保存图像 {output_path}")

        count += 1

    except Exception as e:
        print(f"处理图像 {img_path} 时发生错误: {e}")

if count == 0 and len(result_dict) > 0:
    print("\n未能生成任何验证图像，请检查图像路径和文件权限。")