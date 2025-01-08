import os
import cv2
import numpy as np
import csv


def compute_class_distribution_for_onemask(mask):
    unique, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    class_distribution = {int(cls): count / total_pixels for cls, count in zip(unique, counts)}
    return class_distribution


def process_multiple_masks(mask_dir, output_csv_path):
    # 获取文件夹中的所有mask文件（假设是图片格式，例如 .png）
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]  # 获取所有 .png 文件

    # 打开CSV文件准备写入结果
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入CSV表头，第一列是文件名，接下来的列是类别
        writer.writerow(['filename', 'class_id', 'pixel_count_percentage'])

        # 遍历每个mask文件
        for mask_file in mask_files:
            mask_path = os.path.join(mask_dir, mask_file)

            # 使用cv2.imread读取为单通道图像
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"无法读取文件: {mask_path}")
                continue

            # 获取每张mask的类别分布
            class_distribution = compute_class_distribution_for_onemask(mask)

            # 将结果写入CSV文件，按类别输出
            for cls, percentage in class_distribution.items():
                writer.writerow([mask_file, cls, f"{percentage:.4f}"])

            # 在每张图片的结果之间添加一个空行
            writer.writerow([])

    print(f"All statistics have been saved to {output_csv_path}")



if __name__ == '__main__':
    # 使用示例


    mask_directory = './results/all_masks_31'# 存放所有mask的文件夹
    output_csv = './results/mask_class_distribution.csv'  # 输出的CSV文件名记录每个mask中每种像素占比

    process_multiple_masks(mask_directory, output_csv)
