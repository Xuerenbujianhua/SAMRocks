import cv2
import numpy as np
import os
from tqdm import tqdm
import os
import numpy as np
import cv2
from tqdm import tqdm


def reassemble_images(instance_manager, images, outpath="./results/reassembled_images"):
    """
    根据实例的边界框和原始图像索引，将彩色掩码拼接回原图，并生成每张图像的拼接结果，
    重新保存的文件保持原始图像的文件名。

    参数：
    - instance_manager: InstanceManager 对象，包含所有实例。
    - images: 原始图像的列表，每个元素是 np.ndarray 类型。
    - outpath: 拼接后图像保存的输出文件夹。

    返回：
    - reassembled_image_paths: 拼接完成的图像路径列表。
    """

    # 确保输出文件夹存在
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # 初始化一个字典，用于存储每张原始图像的重建结果，按图像名称索引
    reassembled_images = {}

    # 初始化路径列表
    reassembled_image_paths = []

    # 遍历每个实例，按图片名称将彩色掩码拼接回原图
    for instance in tqdm(instance_manager.instances, desc="Reassembling images"):

        try:
            # 获取实例对应的原始图像名称和原始图片索引
            original_image_name = instance.original_image_name
            image_index = instance.original_image_index

            # 初始化 reassembled_images 中对应的图像副本
            if original_image_name not in reassembled_images:
                # 初始化为原图的副本，保持为三通道图像
                reassembled_images[original_image_name] = np.copy(images[image_index])


            # 加载彩色掩码
            colored_mask = cv2.imread(instance.colored_mask_path, cv2.IMREAD_UNCHANGED)

            # 检查掩码文件是否丢失或损坏
            if colored_mask is None:
                print(
                    f"Error: Mask image for {instance.original_image_name} could not be read. Skipping this instance.")
                continue

            # 获取实例的边界框 (x, y, w, h)
            x, y, w, h = instance.bbox

            # 检查边界框是否越界
            img_h, img_w = reassembled_images[original_image_name].shape[:2]
            if x + w > img_w or y + h > img_h:
                print(
                    f"Error: Bbox for instance {instance.original_image_name} exceeds image boundaries. Skipping this instance.")
                continue

            # 如果掩码大小与边界框不匹配，调整掩码大小
            if colored_mask.shape[:2] != (h, w):
                print(
                    f"Resizing mask for instance {instance.original_image_name} from {colored_mask.shape[:2]} to {(h, w)}")
                colored_mask = cv2.resize(colored_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # 获取原图中边界框位置的区域
            mask_region = reassembled_images[original_image_name][y:y + h, x:x + w]

            # 使用彩色掩码覆盖原图区域
            if colored_mask.shape[2] == 4:  # 如果掩码有4个通道（RGBA），忽略Alpha通道
                colored_mask = colored_mask[:, :, :3]

            # 找到非零区域（掩码覆盖的区域）
            mask_nonzero = np.any(colored_mask != 0, axis=2)

            # 合并彩色掩码到原图
            mask_region[mask_nonzero] = colored_mask[mask_nonzero]

        except Exception as e:
            print(f"Unexpected error processing instance {instance.original_image_name}: {e}")
            continue  # 跳过这个实例，继续处理下一个实例

    # 保存拼接后的每张图像
    for original_image_name, reassembled_image in reassembled_images.items():
        try:
            output_image_path = os.path.join(outpath, original_image_name)
            cv2.imwrite(output_image_path, reassembled_image)
            reassembled_image_paths.append(output_image_path)
        except Exception as e:
            print(f"Error saving image {original_image_name}: {e}")

    return reassembled_image_paths

def reassemble_images_with_Principal_axis(instance_manager, images, outpath="./results/reassembled_images"):
    """
    根据实例的边界框和原始图像索引，将彩色掩码拼接回原图，并生成每张图像的拼接结果，
    重新保存的文件保持原始图像的文件名。

    参数：
    - instance_manager: InstanceManager 对象，包含所有实例。
    - images: 原始图像的列表，每个元素是 np.ndarray 类型。
    - outpath: 拼接后图像保存的输出文件夹。

    返回：
    - reassembled_image_paths: 拼接完成的图像路径列表。
    """

    # 确保输出文件夹存在
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # 初始化一个字典，用于存储每张原始图像的重建结果，按图像名称索引
    reassembled_images = {}

    # 初始化路径列表
    reassembled_image_paths = []

    # 遍历每个实例，按图片名称将彩色掩码拼接回原图
    for instance in tqdm(instance_manager.instances, desc="Reassembling images"):

        try:
            # 获取实例对应的原始图像名称和原始图片索引
            original_image_name = instance.original_image_name
            image_index = instance.original_image_index

            # 初始化 reassembled_images 中对应的图像副本
            if original_image_name not in reassembled_images:
                # 初始化为原图的副本，保持为三通道图像
                reassembled_images[original_image_name] = np.copy(images[image_index])

            # 加载彩色掩码
            # colored_mask = cv2.imread(instance.colored_mask_path, cv2.IMREAD_UNCHANGED)
            colored_mask = instance.get_dynamic_attribute('principal_axis')
            # 检查掩码文件是否丢失或损坏
            if colored_mask is None:
                print(
                    f"Error: Mask image for {instance.original_image_name} could not be read. Skipping this instance.")
                continue

            # 获取实例的边界框 (x, y, w, h)
            x, y, w, h = instance.bbox

            # 检查边界框是否越界
            img_h, img_w = reassembled_images[original_image_name].shape[:2]
            if x + w > img_w or y + h > img_h:
                print(
                    f"Error: Bbox for instance {instance.original_image_name} exceeds image boundaries. Skipping this instance.")
                continue

            # 如果掩码大小与边界框不匹配，调整掩码大小
            if colored_mask.shape[:2] != (h, w):
                print(
                    f"Resizing mask for instance {instance.original_image_name} from {colored_mask.shape[:2]} to {(h, w)}")
                colored_mask = cv2.resize(colored_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # 获取原图中边界框位置的区域
            mask_region = reassembled_images[original_image_name][y:y + h, x:x + w]

            # 使用彩色掩码覆盖原图区域
            if colored_mask.shape[2] == 4:  # 如果掩码有4个通道（RGBA），忽略Alpha通道
                colored_mask = colored_mask[:, :, :3]

            # 找到非零区域（掩码覆盖的区域）
            mask_nonzero = np.any(colored_mask != 0, axis=2)

            # 合并彩色掩码到原图
            mask_region[mask_nonzero] = colored_mask[mask_nonzero]

        except Exception as e:
            print(f"Unexpected error processing instance {instance.original_image_name}: {e}")
            continue  # 跳过这个实例，继续处理下一个实例

    # 保存拼接后的每张图像
    for original_image_name, reassembled_image in reassembled_images.items():
        try:
            output_image_path = os.path.join(outpath, original_image_name)
            cv2.imwrite(output_image_path, reassembled_image)
            reassembled_image_paths.append(output_image_path)
        except Exception as e:
            print(f"Error saving image {original_image_name}: {e}")

    return reassembled_image_paths


def reassemble_image_masks(instance_manager, images, output_directory="./results/reassemble_image_masks"):
    """
    将每个实例的掩码重新拼接回原图，并保存完整的掩码图像。

    参数：
    - instance_manager: InstanceManager 对象，管理所有实例。
    - images: 原始图像的列表，用于作为基底。
    - output_directory: 保存拼接后掩码图像的输出目录。

    返回：
    - reassembled_mask_paths: 重新拼接的完整掩码图像的保存路径列表。
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    reassembled_masks = {}

    for idx, instance in tqdm(enumerate(instance_manager.instances), desc="Reassembling masks",
                              total=len(instance_manager.instances)):
        try:
            mask = instance.get_mask()  # 获取当前实例的二维掩码
            # 如果掩码是多通道的，转换为单通道
            if mask.ndim == 3:
                mask = mask[:, :, -1]  # 假设使用Alpha通道
                # 确保掩码是单通道且数据类型为uint8
            mask = mask.astype(np.uint8)

            bbox = instance.bbox
            original_image_name = instance.original_image_name
            image_index = instance.original_image_index

            # 初始化每个图像的完整掩码图为二维数组
            if original_image_name not in reassembled_masks:
                reassembled_masks[original_image_name] = np.zeros(images[image_index].shape[:2], dtype=np.uint8)

            x, y, w, h = bbox

            img_h, img_w = reassembled_masks[original_image_name].shape
            if x + w > img_w or y + h > img_h:
                print(
                    f"Error: Bbox for instance {original_image_name} exceeds image boundaries. Skipping this instance.")
                continue

            if mask.shape != (h, w):
                print(
                    f"Mismatch between mask size {mask.shape} and bbox size {(h, w)} for instance {original_image_name}. Skipping this instance.")
                continue

            current_mask = reassembled_masks[original_image_name][y:y + h, x:x + w]

            mask_nonzero = mask > 0
            # current_mask[mask_nonzero] = mask[mask_nonzero]
            # 仅在目标区域为0时赋值，避免覆盖已有的 `id`
            assignment_mask = mask_nonzero & (current_mask == 0)
            current_mask[assignment_mask] = mask[assignment_mask]


        except Exception as e:
            print(f"Unexpected error processing instance {original_image_name}: {e}")
            continue

    reassembled_mask_paths = []
    for original_image_name, reassembled_mask in reassembled_masks.items():
        try:
            # reassembled_mask
            unique_ids = np.unique(reassembled_mask[reassembled_mask != 0].flatten())
            # print("Unique IDs (excluding 0):", unique_ids)
            # output_path = os.path.join(output_directory, original_image_name)

            base_name, _ = os.path.splitext(original_image_name)
            output_path = os.path.join(output_directory, f"{base_name}.png")# # 修改输出路径以使用PNG格式-格式无损，可以防止意外的像素值
            # 保存为二维单通道图像，保持ID值不变
            # print(reassembled_mask.shape)
            cv2.imwrite(output_path, reassembled_mask)
            reassembled_mask_paths.append(output_path)
        except Exception as e:
            print(f"Error saving image {original_image_name}: {e}")

    return reassembled_mask_paths
