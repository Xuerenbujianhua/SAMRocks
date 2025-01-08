import os

import cv2
from tqdm import tqdm
from Mineral_segmentation.Pixel_to_ratio import Pixel_to_ratio

def load_images_masks_labels(input_dir, batch_size=32):
    """
    分批加载图像、掩膜和标签数据，使用生成器防止内存溢出。

    参数：
    - image_dir: 图像文件夹路径
    - mask_dir: 掩膜文件夹路径
    - label_dir: 标签文件夹路径
    - batch_size: 每次加载的图像数量

    返回：
    - 一个生成器，每次返回一批图像、掩膜、标签和对应的文件名列表
    """
    image_dir = os.path.join(input_dir, 'images')
    mask_dir = os.path.join(input_dir, 'sams')
    label_dir = os.path.join(input_dir, 'labels')

    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    label_files = sorted(os.listdir(label_dir))

    assert len(image_files) == len(mask_files) == len(label_files), "数量不匹配，请检查文件夹中的数据"

    for i in tqdm(range(0, len(image_files), batch_size), desc="load data"):
        images, masks, labels = [], [], []
        image_names, mask_names, label_names = [], [], []

        for img_file, mask_file, label_file in zip(image_files[i:i + batch_size], mask_files[i:i + batch_size],
                                                   label_files[i:i + batch_size]):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            label_path = os.path.join(label_dir, label_file)

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            label_mask = cv2.imread(label_path)

            if img is None or mask is None or label_mask is None:
                print(f"Failed to load {img_file}, {mask_file}, or {label_file}")
                continue

            images.append(img)
            masks.append(mask)
            labels.append(label_mask)

            # 保存文件名以确保图像、掩码和标签保持一致
            image_names.append(img_file)
            mask_names.append(mask_file)
            label_names.append(label_file)

        if images:  # 仅当有数据时才返回该批次
            yield images, masks, labels, image_names, mask_names, label_names


def load_images_masks(input_dir, batch_size=32):
    """
    分批加载图像和掩膜数据，使用生成器防止内存溢出。

    参数：
    - image_dir: 图像文件夹路径
    - mask_dir: 掩膜文件夹路径
    - batch_size: 每次加载的图像数量

    返回：
    - 一个生成器，每次返回一批图像、掩膜和对应的文件名列表
    """

    image_dir = os.path.join(input_dir, 'images')
    mask_dir = os.path.join(input_dir, 'sams')

    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    assert len(image_files) == len(mask_files), "图像与掩膜的数量不匹配，请检查文件夹中的数据"

    for i in tqdm(range(0, len(image_files), batch_size), desc="加载图像和掩膜数据"):
        images, masks = [], []
        image_names, mask_names = [], []

        for img_file, mask_file in zip(image_files[i:i + batch_size], mask_files[i:i + batch_size]):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            if img is None or mask is None:
                print(f"Failed to load {img_file} or {mask_file}")
                continue

            images.append(img)
            masks.append(mask)

            # 保存文件名以确保图像和掩膜保持一致
            image_names.append(img_file)
            mask_names.append(mask_file)

        if images:  # 确保非空批次
            yield images, masks, image_names, mask_names


def load_images_masks_with_measuring_scale(input_dir, batch_size=32,user_pixel_to_mm_ratio=0.0001, min_area=500, Scale_Value_user=500,use_user_pixel_to_ratio=True):
    """
    分批加载图像和掩膜数据，使用生成器防止内存溢出。

    参数：
    - image_dir: 图像文件夹路径
    - mask_dir: 掩膜文件夹路径
    - batch_size: 每次加载的图像数量

    返回：
    - 一个生成器，每次返回一批图像、掩膜和对应的文件名列表
    """

    image_dir = os.path.join(input_dir, 'images')
    mask_dir = os.path.join(input_dir, 'sams')

    if not os.path.exists(mask_dir) or not os.path.exists(image_dir):
        return None
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    assert len(image_files) == len(mask_files), "图像与掩膜的数量不匹配，请检查文件夹中的数据"


    for i in tqdm(range(0, len(image_files), batch_size), desc="加载图像和掩膜数据"):
        images, masks = [], []
        image_names, mask_names = [], []
        pixel_to_mm_ratios = []
        for img_file, mask_file in zip(image_files[i:i + batch_size], mask_files[i:i + batch_size]):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            if img is None or mask is None:
                print(f"Failed to load {img_file} or {mask_file}")
                continue

            images.append(img)
            masks.append(mask)
            #计算比例尺
            if use_user_pixel_to_ratio:
                pixel_to_mm_ratio = user_pixel_to_mm_ratio
            else:

                pixel_to_mm_ratio = Pixel_to_ratio(img_path, pixel_to_mm_ratio=user_pixel_to_mm_ratio, min_area=min_area,
                                           Scale_Value_user=Scale_Value_user,
                                           use_user_pixel_to_ratio=use_user_pixel_to_ratio)

            pixel_to_mm_ratios.append(abs(pixel_to_mm_ratio))

            # 保存文件名以确保图像和掩膜保持一致
            image_names.append(img_file)
            mask_names.append(mask_file)

        if images:  # 确保非空批次
            # yield images, masks, image_names, mask_names,pixel_to_mm_ratios
            yield images, masks, image_names, pixel_to_mm_ratios
