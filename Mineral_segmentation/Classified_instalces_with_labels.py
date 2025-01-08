import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil


# 定义函数以获取主导类别ID
def get_dominant_class_id(label_path):
    # 读取标签图像
    label_image = np.array(Image.open(label_path))
    # 计算每个类别的像素数量
    unique, counts = np.unique(label_image, return_counts=True)
    # 找到出现次数最多的类别ID
    dominant_class_id = unique[np.argmax(counts)]
    return dominant_class_id

# 修改后的分类函数
def classify_images(output_folder, classes_name,images_folder='./results/instances_result/instance_images', labels_folder='./results/instances_result/instance_labels', skip_id_0=False):
    # 创建类别名称到索引的映射，并处理 Quartzite 和 Quartzitic
    label_to_value = {}
    id_for_Quartzite = None
    for idx, name in enumerate(classes_name):
            label_to_value[name] = idx

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 存储创建的文件夹名称
    created_folders = set()
    folder_id_mapping = {}  # 存储文件夹名称与ID的映射
    # 遍历标签文件夹中的所有标签文件
    for label_filename in tqdm(os.listdir(labels_folder), desc="Classify with class name"):
        # if not label_filename.endswith('.png'):
        #     continue

        # 生成图像和标签文件的路径
        label_path = os.path.join(labels_folder, label_filename)
        image_filename = label_filename  # 假设图像文件名与标签文件名相同
        image_path = os.path.join(images_folder, image_filename)

        if not os.path.exists(image_path):
            # 如果对应的图像文件不存在，跳过
            continue

        # 获取主导类别ID
        dominant_class_id = get_dominant_class_id(label_path)

        # 如果选择跳过id为0的类别，并且主导类别ID为0，则跳过当前文件
        if skip_id_0 and dominant_class_id == 0:
            continue

        # 查找主导类别对应的类别名称
        dominant_class_name = None
        for class_name, class_idx in label_to_value.items():
            if class_idx == dominant_class_id:
                dominant_class_name = class_name
                break

        if dominant_class_name is None:
            continue

        # 将 Quartzite 和 Quartzitic 统一到 Quartzite 文件夹
        if dominant_class_name == 'Quartzitic':
            dominant_class_name = 'Quartzite'

        # 生成类别名称文件夹路径
        class_folder = os.path.join(output_folder, dominant_class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
            created_folders.add(dominant_class_name)  # 记录创建的文件夹名称
            folder_id_mapping[dominant_class_name] = dominant_class_id  # 记录名称与ID的对应关系

        # 复制图像文件到对应的类别文件夹
        output_image_path = os.path.join(class_folder, image_filename)
        if not os.path.exists(output_image_path):
            shutil.copy(image_path, output_image_path)

        # 复制标签文件到对应的类别文件夹
        output_label_path = os.path.join(class_folder, label_filename)
        if not os.path.exists(output_label_path):
            shutil.copy(label_path, output_label_path)

        # print(f"Copied {image_filename} and {label_filename} to class folder {dominant_class_name}")

    sorted_folders = sorted(created_folders)
    sorted_mapping = {folder: folder_id_mapping[folder] for folder in sorted_folders}

    return sorted_folders,sorted_mapping
if __name__ == "__main__":
    classes_name = [
        'background', 'Matrix', 'Hyaline', 'Andesite', 'Tuffin', 'Clay',
        'Rhyolitic', 'Quartz', 'Quartzite', 'Kaolinite', 'Calcite',
        'Montmorilonite', 'Siderite', 'Cordierite', 'Plagioclase', 'Sandstone', 'Oolith',
        'Spherulite', 'Glauconite', 'Orthoclase', 'Anhydrite', 'Granite',
        'Biotite', 'Microcline', 'Pore', 'Pyroclast', 'Quartzitic'
    ]

    # 使用示例
    images_folder = r'C:\Users\Admin\Fengworkplace\pycharmworkplace\Start\models\Results_instances_features\instances_labels2\images'
    labels_folder = r'C:\Users\Admin\Fengworkplace\pycharmworkplace\Start\models\Results_instances_features\instances_labels2\labels'
    output_folder = '../test_classified_images_en_name'

    skip_id_0 = False
    classname,classname_mapping = classify_images(images_folder, labels_folder, output_folder, classes_name, skip_id_0)
    print(len(classname))
    print(classname)
    print(classname_mapping)