import pickle
import os

# 假设 keypoints.pkl 已经存在并可以被读取
with open("keypoints.pkl", 'rb') as f:
    result_dict = pickle.load(f)

# 创建新的字典来存储 video_scene_name: num 对
video_scene_num_dict = {}

# 遍历 result_dict，提取视频场景名称和对应数据的长度
for full_path, data_list in result_dict.items():
    # 从完整路径中提取目录部分
    dir_path = os.path.dirname(full_path)
    # 再次使用 os.path.basename 获取最后一个目录名，这就是视频场景名称
    video_scene_name = os.path.basename(dir_path)

    # 将场景名称作为键，数据列表的长度作为值存入字典
    video_scene_num_dict[video_scene_name] = len(data_list)

# --- 将新字典保存为 Pickle 文件 ---
output_filename = "video_scene_num_objects.pkl"  # 您可以自定义文件名

with open(output_filename, 'wb') as f:
    pickle.dump(video_scene_num_dict, f)

print(f"新字典已成功保存到 '{output_filename}'")