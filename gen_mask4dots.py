import os
import cv2
import torch
import numpy as np
import pickle
from tqdm import tqdm
from typing import List, Tuple
from segment_anything import sam_model_registry, SamPredictor
import shutil  # 导入 shutil 模块用于文件复制


def setup_sam_model(model_type: str = "vit_l", device: str = "cuda",
                    ckpt: str = "sam_vit_l_0b3195.pth") -> SamPredictor:
    """
    设置并加载 Segment Anything Model (SAM) 预测器。
    Args:
        model_type (str): SAM 模型类型，如 "vit_l", "vit_h"。
        device (str): 运行模型的设备，如 "cuda" 或 "cpu"。
        ckpt (str): SAM 模型检查点文件的路径。
    Returns:
        SamPredictor: 配置好的 SAM 预测器。
    """
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA 不可用，切换到 CPU。")
            device = "cpu"
        elif torch.cuda.device_count() <= int(device.split(':')[-1]):
            # 检查指定的 CUDA 设备是否存在，如果不存在则切换到 CPU
            print(f"指定 CUDA 设备 {device} 不存在，切换到 CPU。")
            device = "cpu"
    print(f"使用设备: {device}")

    print("加载 SAM 模型中...")
    try:
        # 从注册表中获取模型并加载检查点
        sam = sam_model_registry[model_type](checkpoint=ckpt)
        sam.to(device)  # 将模型移动到指定设备
        return SamPredictor(sam)
    except FileNotFoundError:
        print(f"错误: SAM 模型检查点 '{ckpt}' 未找到。请确保文件路径正确。")
        exit()  # 检查点是必需的，如果找不到就退出
    except Exception as e:
        print(f"加载 SAM 模型时发生错误: {e}")
        exit()


def segment_from_points(predictor: SamPredictor, image_path: str, fg_points: List[List[int]]) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    使用 SAM 从给定点生成图像分割掩码。
    Args:
        predictor (SamPredictor): 配置好的 SAM 预测器。
        image_path (str): 输入图像的路径。
        fg_points (List[List[int]]): 前景点坐标列表，每个点为 [x, y]。
    Returns:
        Tuple[np.ndarray, np.ndarray]: 生成的二值掩码 (0或255) 和前景点数组。
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"错误: 无法读取图像: {image_path}。请检查路径或文件是否损坏。")
        # 返回一个虚拟的空掩码和空点，以便主循环可以跳过此图像
        return np.zeros((1, 1), dtype=np.uint8), np.array([])

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    fg_array = np.array(fg_points, dtype=np.int32)
    if fg_array.size == 0:
        print(f"跳过图像: {os.path.basename(image_path)}（无前景点）")
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8), fg_array

    input_labels = np.ones(len(fg_array), dtype=int)  # 所有点都标记为前景点
    masks, _, _ = predictor.predict(
        point_coords=fg_array,
        point_labels=input_labels,
        multimask_output=False  # 只输出一个掩码
    )
    # 将掩码转换为 0 或 255 的 uint8 格式
    return (masks[0] * 255).astype(np.uint8), fg_array


def main():
    keypoints_path = r"keypoints.pkl"
    if not os.path.exists(keypoints_path):
        print(f"错误: 找不到 keypoints.pkl 文件: {keypoints_path}")
        return

    # 加载关键点数据
    with open(keypoints_path, 'rb') as f:
        result_dict = pickle.load(f)

    # 设置 SAM 模型。如果你的GPU是 cuda:0 或 cuda:1，可以将 "cuda:3" 改为 "cuda:0" 或 "cuda:1"
    # 或者直接用 "cuda" 让 PyTorch 自动选择默认 GPU。
    predictor = setup_sam_model("vit_h", "cuda:3", "sam_vit_h_4b8939.pth")

    # 遍历每个图片路径和对应的关键点
    for i, (image_path_str, fg_points) in enumerate(tqdm(result_dict.items(), desc="处理进度", unit="图像")):
        # 从 pickle 文件中读取的路径应该是原始图片的绝对路径
        image_path: str = image_path_str

        # 确保当前图片路径是绝对路径，这对于文件操作更安全
        current_image_abs_path = os.path.abspath(image_path)

        # 提取图片所在的场景目录 (例如 /data/.../frames/17-Scene-005)
        # 假设原始图片在 `.../frames/场景名/图片文件名`
        # 并且新的 GT 和 Imgs 目录将创建在 `.../frames/场景名/` 下
        image_base_dir = os.path.dirname(current_image_abs_path)

        # 定义 GT 和 Imgs 目录的完整路径
        gt_dir = os.path.join(image_base_dir, "GT")
        imgs_dir = os.path.join(image_base_dir, "Imgs")

        # 创建目标目录，如果它们不存在
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(imgs_dir, exist_ok=True)

        # 对当前图片进行分割
        mask, fg_array = segment_from_points(predictor, current_image_abs_path, fg_points)

        # 检查生成的掩码是否有效（不是虚拟掩码且非空）
        if mask.shape == (1, 1) or mask.sum() == 0:
            print(f"跳过图像: {os.path.basename(current_image_abs_path)}（生成的掩码为空或图像读取失败）")
            continue

        # 获取图片文件名和生成掩码的文件名
        image_filename = os.path.basename(current_image_abs_path)
        mask_filename = os.path.splitext(image_filename)[0] + ".png"

        # 保存生成的掩码到 GT 目录
        mask_output_path = os.path.join(gt_dir, mask_filename)
        cv2.imwrite(mask_output_path, mask)

        # 将原始图片复制到 Imgs 目录
        img_output_path: str = os.path.join(imgs_dir, image_filename)
        try:
            shutil.copy(current_image_abs_path, img_output_path)
        except Exception as e:
            print(f"复制图片 {current_image_abs_path} 到 {img_output_path} 时出错: {e}")
            continue  # 如果复制失败，则跳过删除，以免误删

        # --- 删除原始图片文件 ---
        # 在删除前，再次确认原始文件是否存在
        if os.path.exists(current_image_abs_path):
            try:
                os.remove(current_image_abs_path)
                # print(f"DEBUG: 已成功删除原始图片: {current_image_abs_path}") # 如需详细输出，可取消注释
            except FileNotFoundError:
                print(f"错误: 原始图片 {current_image_abs_path} 未找到，可能已被移动或删除。")
            except PermissionError:
                print(f"错误: 删除 {current_image_abs_path} 时权限不足。请检查文件或目录权限。")
            except OSError as e:
                print(f"错误: 删除 {current_image_abs_path} 时发生操作系统错误: {e}")
            except Exception as e:
                print(f"错误: 删除 {current_image_abs_path} 时发生未知错误: {e}")
        else:
            print(f"警告: 原始图片 {current_image_abs_path} 在尝试删除前已不存在。跳过删除操作。")

    print("\n处理完成！数据集已创建，原始图片已删除（或尝试删除，请检查日志）。")


if __name__ == "__main__":
    main()