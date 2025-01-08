import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import torch

def augment_one_image(image, device):
    """对图像进行数据增强，并保存为PNG格式"""
    # 创建增强流水线，去掉会裁剪或扭曲图像形状的增强操作

    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),  # 水平翻转
        A.VerticalFlip(p=0.5),  # 垂直翻转
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=30, p=0.5),  # 平移和旋转
        A.RandomBrightnessContrast(p=0.5),  # 亮度和对比度调整
        A.Blur(blur_limit=3, p=0.2),  # 模糊
    ])

    # 将图像转换为GPU张量以加速处理
    image_tensor = torch.from_numpy(image).to(device)

    # 将图像转换回CPU进行数据增强
    augmented = augmentation_pipeline(image=image_tensor.cpu().numpy())
    # 提取增强后的图像
    augmented_image = augmented['image']

    return augmented_image


def process_img_augment_for_one(input_dir, augmentations=5, device='cpu'):
    """处理图像和标签，进行多种数据增强，生成新数据"""
    image_folder = input_dir

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) == 0:
        print("输入文件夹中没有找到图像文件。")
        return

    # print(f"在输入文件夹中找到 {len(image_files)} 张图片。")

    image_count = 0
    # for filename in tqdm(image_files, desc="数据增强", total=len(image_files)):
    for filename in image_files:
        file_stem = os.path.splitext(filename)[0]
        image = cv2.imread(os.path.join(image_folder, filename))
        if image is None :
            print(f"文件读取失败: {filename}，跳过该文件。")
            continue

        for i in range(1, augmentations + 1):
            aug_img = augment_one_image(image,device)

            img_save_path = os.path.join(image_folder, f"{file_stem}-{i}.png")

            cv2.imwrite(img_save_path, aug_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])

            # image_count += 1

    # print(f"共生成了 {image_count} 张增强图像。")

def process_img_augment_for_one_floder(input_dir,augmentations,device='cpu'):
    folder_names = [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]

    for name in tqdm(folder_names,desc='Data enhancement'):
        path = os.path.join(input_dir, name)
        process_img_augment_for_one(path, augmentations=augmentations, device=device)




if __name__ == "__main__":
    # input_dir = './results/images_with_labels'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择 GPU 或 CPU
    input_dir_unlabel = '../data/images_without_labels'
    output_dir = r'./results/aug_data/images_without_labels'  # 输出文件夹
    augmentations = 2  # 每张图像进行多少次增强操作

    try:
        process_img_augment_for_one(input_dir_unlabel, output_dir, augmentations=augmentations,device=device)
    except Exception as e:
        print(f"发生错误: {e}")