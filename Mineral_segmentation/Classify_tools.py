import os
import time
import logging
from datetime import datetime
from collections import Counter
import shutil
import json

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm  # 用于显示进度条
import torch.nn.functional as F
import seaborn as sns
# 修复通道数问题，定义一个自定义转换，将四通道图像转换为三通道
class RGBA2RGB(object):
    def __call__(self, img):
        if img.mode == 'RGBA':
            img = img.convert('RGB')  # 丢弃Alpha通道
        return img


# 定义焦点损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce_loss(inputs, targets)
        pt = torch.exp(logpt)
        loss = -self.alpha * ((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# 封装模型训练和评估流程
class ImageClassifier:
    def __init__(self, data_dir, model_name='resnet18', batch_size=32, num_epochs=10, learning_rate=0.001,patience=30,
                 lossname='CrossEntropyLoss'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.lossname = lossname
        self.current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.models_result_dir = 'models_result'
        os.makedirs(self.models_result_dir, exist_ok=True)
        self.model_dir = os.path.join(self.models_result_dir, model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_save_path = os.path.join(self.model_dir, f'{model_name}_{self.current_time}.pth')

        # 模型选项
        self.model_name = model_name
        self.model = self._initialize_model(model_name)

        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # 加载数据集和数据预处理
        self.train_loader, self.test_loader = self._load_data()

        # 定义损失函数和优化器
        self.set_loss_function(self.lossname)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 设置日志记录
        log_filename = os.path.join(self.model_dir, f'{self.model_name}_{self.current_time}.log')
        self.logger = logging.getLogger(self.model_name)

        # 防止重复添加 handler 导致日志重复
        if not self.logger.hasHandlers():
            handler = logging.FileHandler(log_filename)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # 保存 class_names
        self.save_class_names(self.model_dir)

        # 清理旧的模型结果，保留最近的两个
        self.cleanup_old_models()

    def _initialize_model(self, model_name):
        """根据模型名称初始化预训练模型，并替换最后一层"""
        num_classes = self._get_num_classes()
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'inception_v3':
            model = models.inception_v3(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

        else:
            raise ValueError(
                "Unsupported model name. Choose from 'resnet18', 'mobilenet_v2', 'efficientnet_b0', etc.")
        return model

    def _get_num_classes(self):
        """自动推断类别数"""
        dataset = datasets.ImageFolder(root=self.data_dir)
        return len(dataset.classes)

    def _load_data(self):
        """加载数据集并进行训练集、测试集划分"""

        # 图像预处理
        transform = transforms.Compose([
            RGBA2RGB(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)
        self.class_names = dataset.classes  # 获取类别名称

        # 划分训练集和测试集
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # # 应用不同的预处理
        # train_dataset.Dataset.transform = train_transform
        # test_dataset.Dataset.transform = val_transform

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def compute_class_weights(self):
        """计算类别权重以处理类别不平衡"""
        labels = [self.train_loader.dataset.dataset.targets[i] for i in self.train_loader.dataset.indices]
        label_counts = Counter(labels)
        num_samples = len(labels)
        num_classes = len(label_counts)
        class_weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 0)
            class_weights.append(num_samples / (num_classes * count))
        self.class_weights = torch.tensor(class_weights).float().to(self.device)

    def set_loss_function(self, lossname='CrossEntropyLoss'):
        """设置损失函数，允许自定义"""
        if lossname == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        elif lossname == 'WeightedCrossEntropyLoss':
            self.compute_class_weights()
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        elif lossname == 'FocalLoss':
            self.criterion = FocalLoss()
        else:
            raise ValueError("Unsupported loss function name.")

    def train_model(self):
        """训练模型"""
        self.model.train()
        train_loss_history = []
        train_acc_history = []
        min_loss = np.Inf
        patience = 0
        start_time = time.time()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            # 使用tqdm显示进度条
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 计算损失和准确率
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 更新进度条
                progress_bar.set_postfix({'loss': loss.item(), 'accuracy': correct / total})

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = correct / total
            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)

            self.logger.info(
                f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
            patience  += 1
            if min_loss > epoch_loss:
                print(
                    f'Save at {epoch} - [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
                self.save_model(epoch+ 1,epoch_loss,self.model_save_path)
                min_loss = epoch_loss
                patience = 0
            if patience >= self.patience:
                break
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f'Training completed in {elapsed_time:.2f} seconds.')

        self._plot_curves(train_loss_history, train_acc_history)

    def train_model_Continue_form_local_pth(self, filepath):
        """继续训练模型"""
        # 加载检查点
        # checkpoint = torch.load(checkpoint_path)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
        # min_loss = checkpoint.get('min_loss', np.Inf)
        patience = 0  # 重置耐心计数

        state_dict = torch.load(filepath)
        self.model.load_state_dict(state_dict)
        # self.load_model(checkpoint_path)
        # 重新初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.to(self.device)
        self.model.train()
        train_loss_history = []
        train_acc_history = []

        start_time = time.time()
        for epoch in range(start_epoch, self.num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            # 使用 tqdm 显示进度条
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 计算损失和准确率
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 更新进度条
                progress_bar.set_postfix({'loss': loss.item(), 'accuracy': correct / total})

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = correct / total
            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)

            self.logger.info(
                f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

            patience += 1
            if min_loss > epoch_loss:
                print(
                    f'Save at {epoch} - [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
                # 保存模型，包括当前的 epoch 和最小损失
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'min_loss': epoch_loss,
                }, self.model_save_path)
                min_loss = epoch_loss
                patience = 0
            if patience >= self.patience:
                break
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f'Training completed in {elapsed_time:.2f} seconds.')

        self._plot_curves(train_loss_history, train_acc_history)

    def evaluate_model(self):
        """测试模型并生成分类报告和混淆矩阵"""
        self.model.eval()

        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        # 计算准确率和其他指标
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Balanced Accuracy: {balanced_acc:.4f}')
        self.logger.info(f'Test Accuracy: {accuracy:.4f}')
        self.logger.info(f'Balanced Accuracy: {balanced_acc:.4f}')

        # 打印分类报告和混淆矩阵
        print("Classification Report:")
        # report = classification_report(all_labels, all_preds, target_names=self.class_names)
        all_classes = list(range(len(self.class_names)))
        report = classification_report(
            all_labels,
            all_preds,
            labels=all_classes,
            target_names=self.class_names,
            zero_division=0
        )
        print(report)
        self.logger.info("Classification Report:\n" + report)

        # print("Confusion Matrix:")
        # cm = confusion_matrix(all_labels, all_preds)
        # print(cm)
        # self.logger.info("Confusion Matrix:\n" + str(cm))
        # 生成混淆矩阵
        # cm = confusion_matrix(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds, labels=all_classes)
        # 将混淆矩阵转换为 DataFrame 以便于查看
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)

        # 打印和记录完整的混淆矩阵
        # print("Confusion Matrix:")
        # print(cm_df)
        self.logger.info("Confusion Matrix:\n" + cm_df.to_string())
        #可视化混淆矩阵
        # plt.figure(figsize=(10,7))
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=self.class_names,yticklabels=self)
        # plt.title("Confusion Matrix")
        # plt.xlabel('Predicted label')
        # plt.ylabel('True label')
        # plt.show()
        return report,cm
    def _plot_curves(self, train_loss_history, train_acc_history):
        """绘制损失曲线和准确率曲线"""
        epochs = range(1, len(train_loss_history) + 1)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss_history, label='Training Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc_history, label='Training Accuracy')
        plt.title('Training Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        # 保存损失和准确率曲线
        plot_filename = os.path.join(self.model_dir, f'{self.model_name}_{self.current_time}_training_curves.png')
        plt.savefig(plot_filename)
        self.logger.info(f'Training curves saved to {plot_filename}')
        plt.show()

    def save_model(self, epoch,epoch_loss,filepath=None):
        """保存模型"""
        if filepath is None:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.model_dir, f'{self.model_name}_{current_time}.pth')
        torch.save(self.model.state_dict(), filepath)

        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': self.model.state_dict(),  # 确保这个键存在
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'min_loss': epoch_loss,
        # }, self.filepath)

        self.logger.info(f'Model saved to {filepath}')

    def save_class_names(self, directory):
        """保存类别名称到JSON文件"""
        class_names_path = os.path.join(directory, 'class_names.json')
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f)
        self.logger.info(f'Class names saved to {class_names_path}')

    def cleanup_old_models(self):
        """保留最近的两个模型，删除多余的"""
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
        model_files.sort()
        if len(model_files) > 3:
            files_to_delete = model_files[:-3]
            for file_name in files_to_delete:
                file_path = os.path.join(self.model_dir, file_name)
                os.remove(file_path)
                self.logger.info(f'Deleted old model file: {file_path}')

    def predict(self, image_path):
        """预测单张图像的类别"""
        self.model.eval()
        img = Image.open(image_path)
        transform = transforms.Compose([
            RGBA2RGB(),
            transforms.Resize(256),
            transforms.CenterCrop(224),  # 调整图像大小为224x224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            _, predicted = torch.max(outputs, 1)

        return self.class_names[predicted.item()]

# 定义一个用于加载模型并进行预测的类
# 定义一个用于加载模型并进行预测的类
# 定义一个用于加载模型并进行预测的类
# 定义一个用于加载模型并进行预测的类 -- loaclmodelpredict.py 已经存在了
class ImagePredictor:
    def __init__(self, model_path, model_name='resnet18'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载 class_names
        self.model_dir = os.path.dirname(model_path)
        class_names_path = os.path.join(self.model_dir, 'class_names.json')
        if not os.path.exists(class_names_path):
            raise FileNotFoundError(f'Class names file not found at {class_names_path}')
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        num_classes = len(self.class_names)
        self.model = self._initialize_model(model_name, num_classes)
        self.model = self.model.to(self.device)
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 定义转换
        self.transform = transforms.Compose([
            RGBA2RGB(),
            transforms.Resize(256),
            transforms.CenterCrop(224),  # 调整图像大小为224x224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _initialize_model(self, model_name, num_classes):
        """根据模型名称初始化模型"""
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'inception_v3':
            model = models.inception_v3(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

        else:
            raise ValueError("Unsupported model name.")
        return model

    def predict(self, image_path):
        """预测单张图像的类别"""
        img = Image.open(image_path)
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img)
        #     _, predicted = torch.max(outputs, 1)
        # return self.class_names[predicted.item()]
            # 使用 Softmax 计算每个类别的概率
            probabilities = F.softmax(outputs, dim=1)

            # 获取预测的类别索引及其对应的概率
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_prob = probabilities[0][predicted_class].item()

        return predicted_class, predicted_prob


# 使用示例
if __name__ == "__main__":
    # 参数设置：可以灵活调整基础模型、学习率、batch size等
    classifier = ImageClassifier(
        data_dir='../data/classified_images_with_labels',
        model_name='mobilenet_v2',
        batch_size=32,
        num_epochs=5,
        learning_rate=0.001,
        lossname='FocalLoss'  # 可选 'CrossEntropyLoss', 'WeightedCrossEntropyLoss', 'FocalLoss'
    )

    # 训练模型
    classifier.train_model()

    # 测试并评价模型
    classifier.evaluate_model()

    # # 使用模型进行预测
    # predictor = ImagePredictor(
    #     model_path=classifier.model_save_path,
    #     model_name='mobilenet_v2'
    # )
    # predicted_class = predictor.predict('path_to_image.png')
    # print(f'Predicted Class: {predicted_class}')
