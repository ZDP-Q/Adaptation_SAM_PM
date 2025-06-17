import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry


# 初始化模型
def init_sam(model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamPredictor(sam)


# 加载图像
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# 交互式分割
def interactive_segmentation(predictor, image, input_points, input_labels):
    # input_points: [[x1, y1], [x2, y2], ...]
    # input_labels: [1, 1, ...] (1表示前景点, 0表示背景点)

    input_points = np.array(input_points)
    input_labels = np.array(input_labels)

    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,  # 返回多个掩码
    )

    return masks, scores


# 可视化结果
def show_masks(image, masks, scores):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.imshow(mask, alpha=0.5, cmap="viridis")
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()


# 主函数
def main():
    # 初始化SAM
    predictor = init_sam()

    # 加载图像
    image_path = "your_image.jpg"  # 替换为你的图像路径
    image = load_image(image_path)

    # 设置图像嵌入
    predictor.set_image(image)

    # 设置交互点 (示例点，实际应该从用户输入获取)
    input_points = np.array([[500, 375]])  # 图像上的点坐标 [x,y]
    input_labels = np.array([1])  # 1表示前景点，0表示背景点

    # 进行预测
    masks, scores = interactive_segmentation(predictor, image, input_points, input_labels)

    # 显示结果
    show_masks(image, masks, scores)


if __name__ == "__main__":
    main()