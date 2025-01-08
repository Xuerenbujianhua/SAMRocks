import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import torch  # 确保导入 PyTorch
import os
import shutil
import random


def split_train_test_data_for_one_folder(input_dir, output_dir, train_percent=0.8):
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        raise ValueError(f"The input directory '{input_dir}' does not exist.")

    # 如果输出目录不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有图像文件
    all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    if len(all_files) == 1:
        original_file = all_files[0]
        # 复制文件
        copy_file = f"copy_of_{original_file}"
        shutil.copy(os.path.join(input_dir, original_file), os.path.join(input_dir, copy_file))
        all_files.append(copy_file)
    # 打乱文件顺序
    random.shuffle(all_files)

    # 计算训练集和测试集的划分点
    split_point = int(len(all_files) * train_percent)

    # 划分训练集和测试集
    train_files = all_files[:split_point]
    test_files = all_files[split_point:]

    # 只将测试集文件复制到输出目录
    for file_name in test_files:
        src_file = os.path.join(input_dir, file_name)
        dest_file = os.path.join(output_dir, file_name)
        shutil.move(src_file, dest_file)  # 使用copy2以保留文件元数据

    # print(f"Total files: {len(all_files)}, Training files: {len(train_files)}, Test files: {len(test_files)}")
    # print(f"Test files saved to {output_dir}")


def split_test_img_for_one_floder(input_dir,train_percent=0.8):
    folder_names = [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
    # input_dir = data/1
    basename1 = 'Test_'+os.path.basename(input_dir)
    test_data_basename = os.path.join(os.path.dirname(input_dir),basename1)
    for name in tqdm(folder_names, desc='Test set partition', total=len(folder_names)):
        #文件夹子文件的路径
        path = os.path.join(input_dir, name)
        #path = data/1/class1

        output_dir = os.path.join(test_data_basename,name)
        # output_dir = data/2/class1
        split_train_test_data_for_one_folder(path, output_dir, train_percent=train_percent)
    #返回测试集保存位置
    return test_data_basename