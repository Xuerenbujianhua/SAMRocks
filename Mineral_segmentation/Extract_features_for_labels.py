import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA


# 特征投影器类
class FeatureProjector(nn.Module):
    def __init__(self, input_dim, target_dim=1024):
        super(FeatureProjector, self).__init__()
        self.fc = nn.Linear(input_dim, target_dim)

    def forward(self, x):
        return self.fc(x)


# 通过插值或池化将特征调整到固定大小
def resize_feature(feature, target_dim=1024):
    current_dim = len(feature)
    if current_dim > target_dim:
        stride = current_dim // target_dim
        resized_feature = np.max(feature.reshape(-1, stride), axis=1)  # 最大池化压缩特征
    elif current_dim < target_dim:
        resized_feature = np.interp(np.linspace(0, current_dim - 1, target_dim), np.arange(current_dim), feature)
    else:
        resized_feature = feature
    return resized_feature


# 特征提取和处理
def extract_features_for_labels(instance_manager, feature_extractor, method='projection', batch_size=200,
                                n_components=30, target_dim=1024, outpath="./results/features",
                                flag_record_all_features=False):
    """
    从InstanceManager中提取实例的特征，并根据每个实例的label属性对特征进行分类和降维处理。

    参数：
    - instance_manager: InstanceManager 对象，管理所有实例。
    - feature_extractor: 用于提取特征的自定义特征提取器。
    - method: 特征处理方法，可以是 'projection' 或 'pooling'。
    - batch_size: 每次处理的实例数量。
    - n_components: PCA 的目标维度数。
    - target_dim: 投影或池化后的目标特征维度。
    - outpath: 保存提取和降维后特征的路径。

    返回：
    - feature_paths_label: 每个实例保存的特征文件路径列表。
    - pca_model: 训练好的增量式 PCA 模型，用于后续操作。
    - label_feature_means: 每个标签对应的平均特征字典。
    - all_label_features: 记录所有标签特征的字典 (如果 `flag_record_all_features` 为 True)。
    """

    # 确保保存特征的文件夹存在
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    feature_paths_label = []  # 保存每个实例的特征文件路径
    label_feature_means = {}  # 初始化标签平均特征字典
    all_label_features = {}  # 记录所有标签特征
    scaler = StandardScaler()
    pca = IncrementalPCA(n_components=n_components)  # 增量式PCA
    processed_count = 0  # 记录处理的实例数量

    instances = instance_manager.instances  # 获取所有实例
    # 如果选择特征投影方法，初始化投影器
    if method == 'projection':
        projector = None  # 这里我们等到第一个 batch 后才会初始化
    for i in tqdm(range(0, len(instances), batch_size), desc="Extracting and reducing features"):
        batch_instances = instances[i:i + batch_size]
        batch_features = []
        batch_labels = []

        for instance in batch_instances:
            try:
                label = instance.get_label()
                label = np.unique(label, return_counts=True)[0][np.argmax(np.unique(label, return_counts=True)[1])]
                instance_image = instance.get_image()  # 加载图像

                instance_image = cv2.cvtColor(instance_image, cv2.COLOR_BGRA2BGR)

                instance_tensor = torch.from_numpy(instance_image).float().permute(2, 0, 1)
                instance_tensor = instance_tensor.unsqueeze(0).to(feature_extractor.device)

                feature = feature_extractor.extract(instance_tensor)
                feature = feature.detach().cpu().numpy().flatten()  # 确保特征不需要梯度，并转换为 NumPy

                # 调整特征长度到 target_dim
                resized_feature = resize_feature(feature, target_dim=target_dim)
                #                 # 处理特征
                if method == 'projection':
                    # 初始化投影器
                    if projector is None:
                        input_dim = target_dim  # 因为特征已经被调整为 target_dim
                        projector = FeatureProjector(input_dim=input_dim, target_dim=target_dim).to(
                            feature_extractor.device)

                    # 将调整后的特征转换为 PyTorch 张量，并进行投影
                    feature_tensor = torch.from_numpy(resized_feature).float().to(feature_extractor.device)
                    projected_feature = projector(feature_tensor).cpu().detach().numpy().flatten()
                    batch_features.append(projected_feature)

                elif method == 'pooling':
                    # 如果需要使用池化处理（此处省略，因为 resize_feature 已统一处理）
                    batch_features.append(resized_feature)
                #

                # batch_features.append(resized_feature)
                batch_labels.append(label)

                if flag_record_all_features:
                    all_label_features.setdefault(label, []).append(resized_feature)

            except Exception as e:
                print(f"Error extracting features for instance {instance.original_image_name}: {e}")
                continue
            finally:
                instance.clear_memory()

        try:
            batch_features = np.array(batch_features)
        except Exception as e:
            print(f"Error converting batch_features to NumPy array: {e}")
            continue

        if batch_features.shape[0] == 0:
            continue

        if processed_count == 0:
            batch_scaled = scaler.fit_transform(batch_features)
        else:
            batch_scaled = scaler.transform(batch_features)

        current_samples = batch_scaled.shape[0]

        if current_samples < pca.n_components:
            padding_samples = pca.n_components - current_samples
            zero_padding = np.zeros((padding_samples, batch_scaled.shape[1]))
            batch_scaled = np.vstack((batch_scaled, zero_padding))

        pca.partial_fit(batch_scaled)
        batch_reduced = pca.transform(batch_scaled)

        for j, instance in enumerate(batch_instances):
            # 保存降维后的特征到单独的文件中
            instance_reduced_feature_path = os.path.join(outpath, f"{instance.original_image_name}_reduced_feature.npy")
            np.save(instance_reduced_feature_path, batch_reduced[j])
            instance.set_feature_path(instance_reduced_feature_path)  # 更新路径为降维后的特征路径
            feature_paths_label.append(instance_reduced_feature_path)  # 将降维后的路径添加到列表

        for label in np.unique(batch_labels):
            label_indices = np.where(np.array(batch_labels) == label)[0]
            label_features = batch_reduced[label_indices]
            if label not in label_feature_means:
                label_feature_means[label] = []
            label_feature_means[label].append(np.mean(label_features, axis=0))

        processed_count += len(batch_instances)

        del batch_features, batch_scaled, batch_reduced

    for label in label_feature_means:
        if len(label_feature_means[label]) > 0:
            label_feature_means[label] = np.mean(label_feature_means[label], axis=0)
        else:
            label_feature_means[label] = None

    true_labels_list = sorted(label_feature_means.keys())
    sorted_label_feature_means = {label: label_feature_means[label] for label in true_labels_list}

    return feature_paths_label, pca, sorted_label_feature_means, all_label_features
