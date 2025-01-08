import os
import json
import numpy as np
import cv2
import random
import base64

def draw_shape_on_mask(mask, shape, value):
    """根据形状类型在掩码上绘制形状"""
    try:
        label = shape.get('label')
        points = shape.get('points')
        shape_type = shape.get('shape_type', 'polygon')  # 默认类型为多边形

        if shape_type == 'polygon':
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], value)
        elif shape_type == 'rectangle':
            top_left = tuple(map(int, points[0]))
            bottom_right = tuple(map(int, points[1]))
            cv2.rectangle(mask, top_left, bottom_right, value, thickness=-1)
        elif shape_type == 'circle':
            center = tuple(map(int, points[0]))
            circumference_point = tuple(map(int, points[1]))
            radius = int(np.linalg.norm(np.array(center) - np.array(circumference_point)))
            cv2.circle(mask, center, radius, value, thickness=-1)
        elif shape_type == 'line':
            pt1 = tuple(map(int, points[0]))
            pt2 = tuple(map(int, points[1]))
            cv2.line(mask, pt1, pt2, value, thickness=1)
        elif shape_type == 'point':
            pt = tuple(map(int, points[0]))
            cv2.circle(mask, pt, radius=1, color=value, thickness=-1)
        else:
            print(f"警告: 形状 '{label}' 的类型 '{shape_type}' 不支持。")
    except Exception as e:
        print(f"绘制形状 '{label}' 时出错: {e}")
#这个json_to_mask函数没问题
# def json_to_mask(data, image_shape, label_to_value, use_zeros=True):
#     """将JSON标签数据转换为掩码图像"""
#
#     if use_zeros:
#         mask = np.zeros(image_shape[:2], dtype=np.uint8)  # 以0填充背景
#     else:
#         mask = np.ones(image_shape[:2], dtype=np.uint8)  # 以1填充背景
#
#     shapes = data.get('shapes', [])
#     for shape in shapes:
#         label = shape.get('label')
#         if label in label_to_value:
#             value = label_to_value[label]
#             draw_shape_on_mask(mask, shape, value)
#         else:
#             print(f"警告: 未找到类别 '{label}'，此类别将被忽略。")
#
#     return mask


#增加了比例尺合并的部分
def json_to_mask(data, image_shape, label_to_value, use_zeros=True):
    """将JSON标签数据转换为掩码图像"""

    if use_zeros:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)  # 以0填充背景
    else:
        mask = np.ones(image_shape[:2], dtype=np.uint8)  # 以1填充背景

    shapes = data.get('shapes', [])
    for shape in shapes:
        label = shape.get('label', "")
        if 'um' in label:
            # 如果标签中包含“um”，则使用“um”的值
            value = label_to_value.get('um')
            if value is None:
                print(f"警告: 'um' 的值未在 label_to_value 中定义。")
                continue
        elif label in label_to_value:
            # 如果标签存在于 label_to_value 中，使用对应的值
            value = label_to_value[label]
        else:
            # 标签未找到，输出警告并跳过
            # print(f"警告: 未找到类别 '{label}'，此类别将被忽略。")
            continue

        draw_shape_on_mask(mask, shape, value)

    return mask


def process_folder_to_mask(input_folder, output_folder, label_to_value, use_zeros=True):
    """处理文件夹中的所有JSON文件并生成掩码"""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):

        if filename.endswith(".json"):
            json_path = os.path.join(input_folder, filename)
            base_filename = os.path.splitext(filename)[0]

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image_path = data.get('imagePath')
            if image_path:
                image_path = os.path.join(input_folder, os.path.basename(image_path))
            else:
                image_path = os.path.join(input_folder, base_filename + ".jpg")

            if os.path.exists(image_path):
                original_image = cv2.imread(image_path)
                if original_image is None:
                    print(f"无法读取图像文件: {image_path}")
                    continue
            else:
                imageData = data.get('imageData')
                if imageData:
                    image_data = base64.b64decode(imageData)
                    image_array = np.frombuffer(image_data, np.uint8)
                    original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if original_image is None:
                        print(f"无法从 JSON 解码图像数据: {json_path}")
                        continue
                else:
                    print(f"图像文件未找到且 JSON 中没有图像数据: {json_path}")
                    continue

            mask = json_to_mask(data, original_image.shape, label_to_value, use_zeros)
            output_image_path = os.path.join(output_folder, base_filename + ".png")
            cv2.imwrite(output_image_path, mask)
            print(f"The mask is processed and saved: {filename}")

def analyze_mask_image(mask_path):
    """分析掩码图像的格式、像素范围和不同像素值的数量"""
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # 以原始格式读取图像

    if mask is None:
        print(f"无法读取图像: {mask_path}")
        return

    if len(mask.shape) == 3 and mask.shape[2] == 4:
        # 具有 Alpha 通道
        mask = mask[:, :, 3]
    elif len(mask.shape) == 3:
        # 转换为灰度图像
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 获取图像数据类型
    img_dtype = mask.dtype

    # 获取像素值的范围
    min_pixel_value = np.min(mask)
    max_pixel_value = np.max(mask)

    # 获取不同像素值的数量
    unique_values, counts = np.unique(mask, return_counts=True)
    pixel_value_counts = dict(zip(unique_values, counts))

    print(f"图像路径: {mask_path}")
    print(f"图像数据类型: {img_dtype}")
    print(f"像素值范围: {min_pixel_value} - {max_pixel_value}")
    print("不同像素值的数量:")
    for value, count in pixel_value_counts.items():
        print(f"  像素值 {value}: {count} 个像素")

    return mask

def random_select_and_analyze(output_folder):
    """从输出文件夹中随机选择一个掩码图像进行分析"""
    mask_files = [f for f in os.listdir(output_folder) if f.endswith('.png')]

    if not mask_files:
        print("未找到任何掩码图像文件。")
        return

    random_mask_file = random.choice(mask_files)
    mask_path = os.path.join(output_folder, random_mask_file)

    return analyze_mask_image(mask_path)
def visualize_mask(mask, label_to_value):
    """根据类别 ID 为掩码着色"""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # 创建颜色映射
    num_classes = len(label_to_value)
    cmap = plt.get_cmap('jet', num_classes)

    # 归一化掩码值到 [0, num_classes - 1]
    norm = mcolors.Normalize(vmin=0, vmax=num_classes - 1)

    # 创建彩色图像
    mask_color = cmap(norm(mask))

    # 显示掩码
    plt.imshow(mask_color)
    plt.axis('off')
    plt.show()

# Spherulite Oolith Sandstone Cordierite

