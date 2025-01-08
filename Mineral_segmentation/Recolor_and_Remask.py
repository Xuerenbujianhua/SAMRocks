
import cv2
import numpy as np
import os
from tqdm import tqdm

def recolor_instance_masks(instance_manager, palette, outpath="./results/recolored_instances"):
    """
    对每个实例重新着色并保存彩色掩码，同时将彩色掩码路径存储在每个 Instance 对象中。

    参数：
    - instance_manager: InstanceManager 对象，包含所有实例。
    - predicted_labels: 对应每个实例的聚类预测标签列表。
    - palette: 类别对应的调色板。
    - outpath: 保存着色后实例的输出路径。
    """
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    for idx, instance in tqdm(enumerate(instance_manager.instances), desc="Recoloring instances",
                              total=len(instance_manager.instances)):
        # 加载当前实例的掩码和图像
        # instance_image = instance.get_image()
        mask = instance.get_mask()

        # 获取对应的预测标签，并选择调色板中的颜色
        # predicted_class = predicted_labels[idx]
        # predicted_class = np.unique(mask, return_counts=True)[0][np.argmax(np.unique(mask, return_counts=True)[1])]
        predicted_class = instance.cluster_id
        # 如果标签超出调色板范围，则分配随机颜色
        if predicted_class < len(palette):
            color = palette[predicted_class]
        else:
            # 随机颜色分配
            print(f"实例-{idx} 类别未知")
            color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        # instance_image=instance_image[:, :, 3]
        # 创建彩色掩码，根据类别重新着色
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)  # 保留四通道
        for c in range(3):
            colored_mask[mask > 0, c] = color[c]  # 上色

        # 将原有的透明度通道保留到新掩码中
        # colored_mask[:, :, 3] = mask  # 保留透明度（alpha 通道）
        # colored_mask[:, :, 3] = 255
        # 将四通道彩色掩码转换为三通道
        #         colored_mask = colored_mask[:, :, :3]  # 移除 Alpha 通道
        # 生成掩码保存路径，并保存彩色掩码
        output_path = os.path.join(outpath, f'colored_instance_{idx}.png')
        cv2.imwrite(output_path, colored_mask)

        # 将彩色掩码路径保存到实例中
        instance.colored_mask_path = output_path

        # 清理当前实例的内存
        instance.clear_memory()




def remask_instance_masks(instance_manager, batch_size=32, outpath="./results/masked_instances"):
    """
    分批次为实例掩码重新赋值，并保存到磁盘，以二维掩码标签的形式。

    参数:
    - instance_manager: InstanceManager对象，包含要处理的实例。
    - predicted_labels: 聚类或预测得到的标签列表。
    - batch_size: 每次处理的实例批次大小。
    - outpath: 掩码实例保存路径。

    返回:
    - instance_manager: 更新后的InstanceManager对象，实例的掩码被重新赋值。
    """
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    total_instances = len(instance_manager.instances)

    # 分批处理实例
    for i in tqdm(range(0, total_instances, batch_size), desc="Remasking instances"):
        batch_instances = instance_manager.instances[i:i + batch_size]

        for idx, instance in enumerate(batch_instances):
            instance_mask = instance.get_mask()  # 获取当前实例的掩码（二维或三通道）
            # print(np.unique(instance_mask))
            # #获取类别更新后的新类别
            # predicted_class = np.unique(instance_mask, return_counts=True)[0][np.argmax(np.unique(instance_mask, return_counts=True)[1])]
            predicted_class = instance.cluster_id
            # print(f'{instance.instances_id}-{predicted_class}')

            # print(predicted_class)
            # 如果是四通道图像，只保留前3通道（或只保留Alpha通道作为透明度信息）
            if len(instance_mask.shape) == 3 and instance_mask.shape[2] == 4:
                instance_mask = cv2.cvtColor(instance_mask, cv2.COLOR_BGRA2GRAY)  # 转换为灰度图
            # if predicted_class>25:
            #     print(predicted_class)
            # 生成二维的聚类掩码标签
            # instance_mask原始为0和255二值图像
            cluster_mask = np.zeros_like(instance_mask, dtype=np.uint8)
            cluster_mask[instance_mask > 0] = predicted_class  # 将掩码区域赋值为预测的类别
            # print(np.unique(cluster_mask))

            # 保存新的掩码到磁盘
            mask_output_path = os.path.join(outpath, f'masked_instance_{i + idx}.png')
            cv2.imwrite(mask_output_path, cluster_mask)

            # 更新InstanceManager中的掩码路径和内存中的掩码
            instance.mask_path = mask_output_path
            instance._mask = cluster_mask

    return instance_manager
