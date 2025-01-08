import os
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label
from tqdm import tqdm

def fill_empty_labels(mask, max_distance=None, max_area=None):
    """
    填充空值标签，但仅在满足最大距离和最大区域限制的情况下进行填充。

    参数：
    - mask: 输入的标签掩码。
    - max_distance: 最大填充距离阈值（像素）。如果为 None，则不限制距离。
    - max_area: 最大填充区域面积阈值（像素个数）。如果为 None，则不限制区域大小。
    """
    # 识别空值像素（假设空值为0）
    empty_pixels = (mask == 0)
    if not np.any(empty_pixels):
        return mask  # 如果没有空值，直接返回
    # 创建非空值像素的掩码
    non_empty_mask = ~empty_pixels
    # 复制原始标签
    labels = mask.copy()
    # 计算距离变换，找到最近的非空值像素
    distances, indices = distance_transform_edt(empty_pixels, return_distances=True, return_indices=True)
    # 创建填充掩码
    fill_mask = empty_pixels.copy()
    # 应用最大距离限制
    if max_distance is not None:
        fill_mask &= (distances <= max_distance)
    # 应用最大区域限制
    if max_area is not None:
        # 连接空值区域
        structure = np.ones((3, 3), dtype=int)
        labeled_array, num_features = label(empty_pixels, structure=structure)
        for region_label in range(1, num_features + 1):
            region_mask = (labeled_array == region_label)
            region_size = np.sum(region_mask)
            if region_size > max_area:
                fill_mask[region_mask] = False  # 超出最大区域，不进行填充
    # 获取最近非空值像素的标签
    filled_labels = labels[indices[0], indices[1]]
    # 将空值像素替换为填充值
    mask[fill_mask] = filled_labels[fill_mask]
    return mask

def mask_to_color(mask, palette):
    h, w = mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        color_image[mask == class_id] = color
    return color_image

def process_masks(mask_folder, output_folder_colored, palette, flag_fill=False, max_distance=None, max_area=None):
    """
    处理掩码图像，填充空值并转换为彩色图像。

    参数：
    - mask_folder: 掩码图像的输入文件夹。
    - output_folder_colored: 彩色图像的输出文件夹。
    - palette: 颜色调色板。
    - flag_fill: 是否进行空值填充。
    - max_distance: 最大填充距离阈值（像素）。如果为 None，则不限制距离。
    - max_area: 最大填充区域面积阈值（像素个数）。如果为 None，则不限制区域大小。
    """
    if not os.path.exists(output_folder_colored):
        os.makedirs(output_folder_colored)

    mask_filenames = [f for f in os.listdir(mask_folder) if f.endswith('.png') or f.endswith('.jpg')]

    for filename in tqdm(mask_filenames, desc='paint ', total=len(mask_filenames)):
        mask_path = os.path.join(mask_folder, filename)
        mask = np.array(Image.open(mask_path))
        # 确保掩码是单通道
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        # 填充空值
        if flag_fill:
            mask_filled = fill_empty_labels(mask.copy(), max_distance=max_distance, max_area=max_area)
        else:
            mask_filled = mask.copy()
        # 保存填充后的掩码
        filled_mask_image = Image.fromarray(mask_filled.astype(np.uint8))
        filled_mask_image.save(os.path.join(mask_folder, filename))
        # 转换为彩色图像
        color_image = mask_to_color(mask_filled, palette)
        # 保存彩色图像
        color_image_pil = Image.fromarray(color_image)
        color_image_pil.save(os.path.join(output_folder_colored, filename))
    # print('上色完毕!')

if __name__ == '__main__':
    # 定义色表
    palette = [
        [0, 0, 0],         # background
        [169, 169, 169],   # Matrix - 灰色
        [176, 196, 222],   # Hyaline - 淡蓝色
        [119, 136, 153],   # Andesite - 蓝灰色
        [139, 69, 19],     # Tuffin - 棕色
        [222, 184, 135],   # Clay - 米黄色
        [210, 105, 30],    # Rhyolitic - 砖红色
        [255, 255, 255],   # Quartz - 白色（合并后的类别）
    ]

    # 使用示例
    mask_folder = '您的掩码文件夹路径'
    output_folder_colored = '彩色图像的输出文件夹路径'
    # 设置填充参数
    flag_fill = True  # 是否进行填充
    max_distance = 50  # 最大填充距离，例如50个像素
    max_area = 1000    # 最大填充区域面积，例如1000个像素

    process_masks(mask_folder, output_folder_colored, palette, flag_fill=flag_fill, max_distance=max_distance, max_area=max_area)
