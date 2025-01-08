import os
import shutil

def copy_images_and_labels(data1_dir, data2_dir, data3_dir,labelflodername='labels'):
    # 定义目标文件夹路径
    images_dir = os.path.join(data3_dir, 'images')
    labels_dir = os.path.join(data3_dir,labelflodername )

    # 如果目标文件夹不存在，创建它们
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 获取data1和data2中的文件列表
    data1_files = sorted([f for f in os.listdir(data1_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    data2_files = sorted([f for f in os.listdir(data2_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # 获取不带扩展名的文件名
    data1_file_names = {os.path.splitext(f)[0]: f for f in data1_files}
    data2_file_names = {os.path.splitext(f)[0]: f for f in data2_files}

    # 检查文件数量是否相同
    if len(data1_file_names) != len(data2_file_names):
        raise ValueError("data1 和 data2 中的文件数量不匹配！")

    # 检查文件名（忽略扩展名）是否一致
    for file_name in data1_file_names:
        if file_name not in data2_file_names:
            raise ValueError(f"文件名不匹配: {data1_file_names[file_name]} 和 {file_name}")

    # 复制文件到对应的文件夹
    for file_name in data1_file_names:
        # 复制data1的图片到images文件夹
        shutil.copy(os.path.join(data1_dir, data1_file_names[file_name]), os.path.join(images_dir, data1_file_names[file_name]))
        # 复制data2的图片到labels文件夹
        shutil.copy(os.path.join(data2_dir, data2_file_names[file_name]), os.path.join(labels_dir, data2_file_names[file_name]))

    print(f"All files are successfully copied to the {data3_dir} folder。")

if __name__ == "__main__":
    # 使用示例
    data1 = 'path_to_data1_folder'
    data2 = 'path_to_data2_folder'
    data3 = 'path_to_output_data3_folder'

    copy_images_and_labels(data1, data2, data3)
