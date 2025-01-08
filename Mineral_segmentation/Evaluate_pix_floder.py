

import os
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_masks_from_folder_pix(folder_path):
    """从文件夹中加载所有mask"""
    mask_files = {os.path.splitext(f)[0]: os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                  f.endswith('.png')}
    return mask_files


def calculate_confusion_matrix_pix(true_masks, pred_masks, class_mapping):
    """计算混淆矩阵"""
    y_true = np.concatenate([mask.flatten() for mask in true_masks])
    y_pred = np.concatenate([mask.flatten() for mask in pred_masks])

    labels = list(class_mapping.values())
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return conf_matrix


def plot_confusion_matrix_pix(conf_matrix, class_mapping, output_path):
    """绘制并保存混淆矩阵"""
    labels = list(class_mapping.keys())
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(output_path)
    plt.close()

#
# def evaluate_pix(true_masks_folder, pred_masks_folder, class_mapping, output_dir):
#     """评估语义分割模型的表现"""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # 加载真实标签和预测标签
#     true_masks_dict = load_masks_from_folder_pix(true_masks_folder)
#     pred_masks_dict = load_masks_from_folder_pix(pred_masks_folder)
#
#     # 只评估两个文件夹中都有的文件
#     common_files = set(true_masks_dict.keys()).intersection(pred_masks_dict.keys())
#     if not common_files:
#         print("没有找到匹配的文件进行评估。")
#         return
#
#     true_masks = [cv2.imread(true_masks_dict[file], cv2.IMREAD_GRAYSCALE) for file in common_files]
#     pred_masks = [cv2.imread(pred_masks_dict[file], cv2.IMREAD_GRAYSCALE) for file in common_files]
#
#     print(f"Found {len(common_files)} matching files for evaluation.")
#
#     # 计算混淆矩阵
#     conf_matrix = calculate_confusion_matrix_pix(true_masks, pred_masks, class_mapping)
#
#     # 绘制混淆矩阵
#     plot_confusion_matrix_pix(conf_matrix, class_mapping, os.path.join(output_dir, 'confusion_matrix.png'))
#
#     # 将混淆矩阵转换为分类报告
#     y_true = np.concatenate([mask.flatten() for mask in true_masks])
#     y_pred = np.concatenate([mask.flatten() for mask in pred_masks])
#
#     labels = list(class_mapping.values())
#     class_names = list(class_mapping.keys())
#
#     # 分类报告
#     report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True)
#     report_df = pd.DataFrame(report).transpose()
#
#     # 保存分类报告
#     report_path = os.path.join(output_dir, 'classification_report_pix.csv')
#     report_df.to_csv(report_path)
#
#     # 输出整体准确率
#     overall_accuracy = accuracy_score(y_true, y_pred)
#     print(report)
#     print(f"Overall Accuracy: {overall_accuracy:.4f}")
#     print(f"Classification report saved to {report_path}")
#     print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix_pix.png')}")
#
#     return overall_accuracy, report_df,conf_matrix

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


def evaluate_pix(true_masks_folder, pred_masks_folder, class_mapping, output_dir, include_background=True):
    """评估语义分割模型的表现，支持是否评估 background 类别"""

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载真实标签和预测标签
    true_masks_dict = load_masks_from_folder_pix(true_masks_folder)
    pred_masks_dict = load_masks_from_folder_pix(pred_masks_folder)

    # 只评估两个文件夹中都有的文件
    common_files = set(true_masks_dict.keys()).intersection(pred_masks_dict.keys())
    if not common_files:
        print("没有找到匹配的文件进行评估。")
        return

    true_masks = [cv2.imread(true_masks_dict[file], cv2.IMREAD_GRAYSCALE) for file in common_files]
    pred_masks = [cv2.imread(pred_masks_dict[file], cv2.IMREAD_GRAYSCALE) for file in common_files]

    print(f"Found {len(common_files)} matching files for evaluation.")

    # 计算混淆矩阵
    conf_matrix = calculate_confusion_matrix_pix(true_masks, pred_masks, class_mapping)

    # 绘制混淆矩阵
    plot_confusion_matrix_pix(conf_matrix, class_mapping, os.path.join(output_dir, 'confusion_matrix.png'))

    # 将混淆矩阵转换为分类报告
    y_true = np.concatenate([mask.flatten() for mask in true_masks])
    y_pred = np.concatenate([mask.flatten() for mask in pred_masks])

    # 如果不包括 background 类别，移除 background 相关的标签
    if not include_background:
        background_label = class_mapping.get('background')
        if background_label is not None:
            # 移除 y_true 和 y_pred 中的 background 标签
            mask_background = (y_true != background_label) & (y_pred != background_label)
            y_true = y_true[mask_background]
            y_pred = y_pred[mask_background]

            # 移除 background 类别的标签
            labels = [label for label in class_mapping.values() if label != background_label]
            class_names = [class_name for class_name in class_mapping.keys() if
                           class_mapping[class_name] != background_label]
        else:
            labels = list(class_mapping.values())
            class_names = list(class_mapping.keys())
    else:
        labels = list(class_mapping.values())
        class_names = list(class_mapping.keys())

    # 分类报告
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # 保存分类报告
    report_path = os.path.join(output_dir, 'classification_report_pix.csv')
    report_df.to_csv(report_path)

    # 输出整体准确率
    # overall_accuracy = accuracy_score(y_true, y_pred)
    weighted_avg = report["weighted avg"]
    weighted_precision = weighted_avg['precision']
    # print("{:.2f}".format(weighted_precision))  #
    print(weighted_avg)
    print(report)
    print(f"Overall Accuracy: {weighted_precision:.4f}")
    print(f"Classification report saved to {report_path}")
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix_pix.png')}")

    return weighted_precision, report_df, conf_matrix

