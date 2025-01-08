import cv2
import numpy as np
import os
from tqdm import tqdm
from collections import Counter

from Mineral_segmentation.Instance import Instance, InstanceManager
import torch
import torchvision
from torch import nn

def filter_by_non_transparent_ratio(img, transparency_threshold=0.1):
    # 读取四通道图像
    # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 保留alpha通道

    # 检查是否是四通道图像
    if img.shape[2] != 4:
        raise ValueError("The image does not have an alpha channel.")

    # 提取alpha通道（透明度）
    alpha_channel = img[:, :, 3]

    # 计算非透明像素的数量
    non_transparent_pixels = np.count_nonzero(alpha_channel > 0)

    # 计算图像总像素数
    total_pixels = img.shape[0] * img.shape[1]

    # 计算非透明区域的比例
    non_transparent_ratio = non_transparent_pixels / total_pixels

    # 判断是否过滤掉图像
    return non_transparent_ratio < transparency_threshold


def extract_instances_for_labels(images, masks, labels, area_limit=4,
                                 outpath_images="./results/instances_result/instance_images",
                                 outpath_masks="./results/instances_result/instance_masks",
                                 outpath_labels="./results/instances_result/instance_labels",
                                 image_names=None,
                                 include_background=True,
                                 include_matrix=True,
                                 transparency_threshold=0):
    """
    从图像和掩码中提取实例，并为每个实例动态添加标签信息，保留实例边缘信息，避免重复提取，并添加验证。

    参数：
    - images: 图像的 numpy 数组列表。
    - masks: 对应的掩码列表。
    - Tags_For_testing_and_evaluation_purposes_only: 每个掩码的标签列表，长度应与 masks 一致。
    - area_limit: 忽略面积小于该值的实例。
    - outpath_images: 实例图像保存路径。
    - outpath_masks: 实例掩码保存路径。
    - outpath_labels: 实例标签保存路径。
    - image_names: 原始图像的名称列表，必须与 images 对应。
      transparency_threshold 线条及面积占比,小于此值会被过滤，为0表示不过滤
    返回：
    - InstanceManager 对象，包含所有提取的实例，并带有标签信息。
    """

    assert image_names is not None, "必须提供图像名称列表以保持实例的原始图像名称。"
    assert len(images) == len(image_names), "图像列表和图像名称列表的长度必须一致。"
    assert len(masks) == len(labels), "掩码列表和标签列表的长度必须一致。"

    instance_info = []

    # 确保输出目录存在
    os.makedirs(outpath_images, exist_ok=True)
    os.makedirs(outpath_masks, exist_ok=True)
    os.makedirs(outpath_labels, exist_ok=True)

    total_instances = 0
    total_area_covered = 0

    for img_index, (img, mask, img_name, label) in tqdm(enumerate(zip(images, masks, image_names, labels)),
                                                        desc="Extracting instances", total=len(images)):
        original_image_name = img_name
        reshaped_mask = mask.reshape(-1, mask.shape[2])
        unique_colors = np.unique(reshaped_mask, axis=0)

        total_image_area = img.shape[0] * img.shape[1]
        image_area_covered = 0

        global_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

        for color_index, color in enumerate(unique_colors):
            # 检查是否要忽略背景类别
            if not include_background and (np.all(color == [0, 0, 0])):

                continue
            # 生成当前颜色区域的二值掩码
            instance_mask = cv2.inRange(mask, color, color)

            if np.sum(instance_mask) == 0:
                continue

            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 忽略太小的实例
            for contour_index, contour in enumerate(contours):
                if cv2.contourArea(contour) <= area_limit:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                img_height, img_width = img.shape[:2]
                if x + w > img_width:
                    w = img_width - x
                if y + h > img_height:
                    h = img_height - y

                #通过box判断是否是基质，如提取到的实例的box大小与原图相同，则认为其是基质
                #实际上加上连通性分析可能更好
                Size_ratio=0.95
                if not include_matrix and (img_width * Size_ratio <= w <=img_width  or img_height*Size_ratio <=h<=img_height) :
                    continue
                cropped_instance = img[y:y + h, x:x + w]
                cropped_mask = instance_mask[y:y + h, x:x + w]
                cropped_label = label[y:y + h, x:x + w]

                cropped_instance_with_alpha = cv2.cvtColor(cropped_instance, cv2.COLOR_BGR2BGRA)
                cropped_instance_with_alpha[:, :, 3] = cropped_mask

                #过滤线条及面积占比过低的实例
                if transparency_threshold != 0 and filter_by_non_transparent_ratio(cropped_instance_with_alpha,transparency_threshold):
                    continue

                instance_filename = f'instance_{img_index}_{color_index}_{contour_index}.png'
                mask_filename = f'instance_{img_index}_{color_index}_{contour_index}.png'
                label_filename = f'instance_{img_index}_{color_index}_{contour_index}.png'

                image_path = os.path.join(outpath_images, instance_filename)
                mask_path = os.path.join(outpath_masks, mask_filename)
                label_path = os.path.join(outpath_labels, label_filename)

                # 保存裁剪的实例图像和掩码到磁盘（保存为无压缩 PNG）
                cv2.imwrite(image_path, cropped_instance_with_alpha, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(mask_path, cropped_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                # 保存为单通道掩码格式
                cropped_label = cropped_label.astype(np.uint8)
                cv2.imwrite(label_path, cropped_label, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                # 获取类别标签
                label_id = np.bincount(np.array(cropped_label).flatten()).argmax()
                instance = Instance(
                    instances_id=(color_index - 1) + contour_index,
                    image_path=image_path,
                    mask_path=mask_path,
                    bbox=(x, y, w, h),
                    original_image_index=img_index,
                    label_id=label_id,
                    label_path=label_path
                )

                instance.original_image_name = original_image_name
                instance_info.append(instance)

                instance_area = np.sum(cropped_mask > 0)
                image_area_covered += instance_area
                global_mask[y:y + h, x:x + w] = np.maximum(global_mask[y:y + h, x:x + w], cropped_mask)

                total_instances += 1

        coverage_ratio = image_area_covered / total_image_area * 100
        total_area_covered += image_area_covered

    total_image_pixels = sum([img.shape[0] * img.shape[1] for img in images])
    total_coverage_ratio = total_area_covered / total_image_pixels * 100
    print(f"A total of  {total_instances} instances are extracted, covering the total image area of  {total_coverage_ratio:.2f}%.")

    return InstanceManager(instance_info)


def extract_instances_unlabels(images, masks, measuring_scales,Default_measuring_scales=0.0001,area_limit=20, crop_flag=0,outpath_images="./results/instances_result_unlabel/images",
                      outpath_masks="./results/instances_result_unlabel/masks", image_names=None,include_background=True,include_matrix=True,transparency_threshold=0,image_floder_id=0):
    """
    从图像和掩码中提取实例，并保存为图像文件。

    参数：
    - images: 图像的 numpy 数组列表。
    - masks: 对应的掩码列表。
    - area_limit: 忽略面积小于该值的实例。
    - outpath: 输出文件的路径。
    - image_names: 原始图像的名称列表，必须与 images 对应。
    crop_flag是否要裁剪带哦图像的右下角，若为0表示不裁剪，否则表示裁剪比例
    返回：
    - InstanceManager 对象，包含所有提取的实例。
    """
    assert image_names is not None, "必须提供图像名称列表以保持实例的原始图像名称。"
    assert len(images) == len(image_names), "图像列表和图像名称列表的长度必须一致。"
    #检查和处理比例尺
    if all(x is None for x in measuring_scales):
        measuring_scales = np.ones(len(images))
    else :
        measuring_scales = [x if x is not None else Default_measuring_scales for x in measuring_scales]

    # # 将图像和掩码转移到 GPU 上，下面的操作会出现问题
    # images = [torch.tensor(image, device=device) for image in images]
    # masks = [torch.tensor(mask, device=device) for mask in masks]

    instance_info = []
    # 确保输出目录存在
    os.makedirs(outpath_images, exist_ok=True)
    os.makedirs(outpath_masks, exist_ok=True)


    total_instances = 0
    total_area_covered = 0
    # 遍历每张图像和对应的掩码
    for img_index, (img, mask, img_name,measuring_scale) in tqdm(enumerate(zip(images, masks, image_names,measuring_scales)),
                                                 desc="Extracting instances", total=len(images)):
        # 使用 image_names 中的文件名作为 original_image_name
        original_image_name = img_name

        if crop_flag != 0:
            # 获取图像的高度和宽度
            height, width = img.shape[:2]

            # 定义裁剪区域（去掉右下角）
            # img = img[:int(height *crop_flag), :int(width *crop_flag)]  # 例如裁剪100像素
            # mask = mask[:int(height *crop_flag), :int(width *crop_flag)]  # 例如裁剪100像素

            # 计算需要裁剪的高度和宽度
            crop_height = int(height * (1 - crop_flag))
            crop_width = int(width * (1 - crop_flag))
            # （去掉右下角）
            img[height - crop_height:, width - crop_width:] = 0
            mask[height - crop_height:, width - crop_width:] = 0

        reshaped_mask = mask.reshape(-1, mask.shape[2])
        unique_colors = np.unique(reshaped_mask, axis=0)

        total_image_area = img.shape[0] * img.shape[1]
        image_area_covered = 0

        # global_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

        for color_index, color in enumerate(unique_colors):
            # 检查是否要忽略背景类别
            if not include_background and np.all(color == [0, 0, 0]):
                continue
            instance_mask = cv2.inRange(mask, color, color)

            if np.sum(instance_mask) == 0:
                continue

            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            for contour_index, contour in enumerate(contours):
                # 忽略太小的实例
                if cv2.contourArea(contour) < area_limit:
                    continue

                # 计算边界框
                x, y, w, h = cv2.boundingRect(contour)
                img_height, img_width = img.shape[:2]
                # 确保边界框不超出图像大小
                if x + w > img_width:
                    w = img_width - x
                if y + h > img_height:
                    h = img_height - y
                #通过box判断是否是基质，如提取到的实例的box大小与原图相同，则认为其是基质
                #实际上加上连通性分析可能更好
                Size_ratio=0.90
                if not include_matrix and (img_width * Size_ratio <= w <=img_width  or img_height*Size_ratio <=h<=img_height) :
                    continue
                # 裁剪实例和掩码
                cropped_instance = img[y:y + h, x:x + w]
                cropped_mask = instance_mask[y:y + h, x:x + w]

                #增加透明通道
                cropped_instance_with_alpha = cv2.cvtColor(cropped_instance, cv2.COLOR_BGR2BGRA)
                cropped_instance_with_alpha[:, :, 3] = cropped_mask

                #过滤线条及面积占比过低的实例 不为0 则表示过滤
                if transparency_threshold != 0 and filter_by_non_transparent_ratio(cropped_instance_with_alpha,transparency_threshold):
                    continue
                # 定义保存路径
                instance_filename = f'instance_{img_index}_{color_index}_{contour_index}.png'
                mask_filename = f'instance_{img_index}_{color_index}_{contour_index}.png'
                image_path = os.path.join(outpath_images, instance_filename)
                mask_path = os.path.join(outpath_masks, mask_filename)

                # # 保存裁剪的实例图像和掩码到磁盘
                # cv2.imwrite(image_path, cropped_instance)
                # cv2.imwrite(mask_path, cropped_mask)
                # 保存裁剪的实例图像和掩码到磁盘（保存为无压缩 PNG）
                cv2.imwrite(image_path, cropped_instance_with_alpha, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(mask_path, cropped_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                # 创建 Instance 对象，并将其添加到列表中
                instance = Instance(
                    instances_id= img_index+contour_index+contour_index,
                    image_path=image_path,
                    mask_path=mask_path,
                    bbox=(x, y, w, h),
                    original_image_index=img_index,
                    image_floder_id=image_floder_id,
                    measuring_scale = measuring_scale
                )
                # cluster_mask=None
                # 使用图像的真实文件名作为 original_image_name
                instance.original_image_name = original_image_name
                instance_info.append(instance)

                instance_area = np.sum(cropped_mask > 0)
                image_area_covered += instance_area
                # global_mask[y:y + h, x:x + w] = np.maximum(global_mask[y:y + h, x:x + w], cropped_mask)

                total_instances += 1

        total_area_covered += image_area_covered

    total_image_pixels = sum([img.shape[0] * img.shape[1] for img in images])
    total_coverage_ratio = total_area_covered / total_image_pixels * 100
    # print(f"总共提取了 {total_instances} 个实例，覆盖了 {total_coverage_ratio:.2f}% 的总图像面积。")
    print(f"A total of  {total_instances} instances are extracted, covering the total image area of  {total_coverage_ratio:.2f}%.")

    return InstanceManager(instance_info)

