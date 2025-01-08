import os
import cv2
import numpy as np


def compute_class_distribution(mask_folder):
    # 用于存储每个类别的像素计数
    class_pixel_count = {}

    # 统计总的像素数
    total_pixels = 0

    # 遍历文件夹中的所有mask文件
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if len(mask_files) == 0:
        print("mask文件夹中没有找到图像文件。")
        return

    print(f"Find  {len(mask_files)} images in the mask folder。")

    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)

        # 读取mask图像（假设mask图像是单通道）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取mask文件: {mask_path}")
            continue

        # 更新总像素数
        total_pixels += mask.size

        # 获取每个类别ID的像素数
        unique, counts = np.unique(mask, return_counts=True)

        # 更新类别的计数
        for cls, count in zip(unique, counts):
            if cls in class_pixel_count:
                class_pixel_count[cls] += count
            else:
                class_pixel_count[cls] = count

    # 计算每个类别ID的占比
    class_distribution = {cls: count / total_pixels for cls, count in class_pixel_count.items()}

    # 按类别ID排序
    sorted_class_distribution = dict(sorted(class_distribution.items()))
    sorted_class_pixel_count = dict(sorted(class_pixel_count.items()))

    print(f"Total number of pixels: {total_pixels}")
    print("Category ID Indicates the proportion of a category ID:")

    # 按照排序后的类别ID输出
    for cls in sorted_class_distribution:
        percentage = sorted_class_distribution[cls]
        pixel_count = sorted_class_pixel_count[cls]
        print(f"Category {cls}: Scale {percentage:.4%}, Pixel number {pixel_count}")

    # 如果需要返回结果
    # return sorted_class_distribution, sorted_class_pixel_count, total_pixels




def compute_class_distribution_one_image(mask_path):
    # 用于存储每个类别的像素计数
    class_pixel_count = {}

    # 统计总的像素数
    total_pixels = 0

    # 读取mask图像（假设mask图像是单通道）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 更新总像素数
    total_pixels += mask.size

    # 获取每个类别ID的像素数
    unique, counts = np.unique(mask, return_counts=True)

    # 更新类别的计数
    for cls, count in zip(unique, counts):
        if cls in class_pixel_count:
            class_pixel_count[cls] += count
        else:
            class_pixel_count[cls] = count
    class_distribution = {cls: count / total_pixels for cls, count in class_pixel_count.items()}

    # 按类别ID排序
    sorted_class_distribution = dict(sorted(class_distribution.items()))
    sorted_class_pixel_count = dict(sorted(class_pixel_count.items()))

    print(f"Total number of pixels: {total_pixels}")
    print("Category ID Indicates the proportion of a category ID:")

    # 按照排序后的类别ID输出
    for cls in sorted_class_distribution:
        percentage = sorted_class_distribution[cls]
        pixel_count = sorted_class_pixel_count[cls]
        print(f"Category {cls}: Scale {percentage:.4%}, Pixel number {pixel_count}")


if __name__ == "__main__":
    mask_folder = './results/all_masks_31'
    compute_class_distribution(mask_folder)
