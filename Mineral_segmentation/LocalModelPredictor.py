import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms
from PIL import Image
import json

class LocalModelPredictor:
    def __init__(self, model_path, model_name='resnet18',Num_classes=None):
        """
        初始化加载本地模型，并准备预测。
        :param model_path: 模型文件路径（.pth 文件）
        :param model_name: 模型名称（例如 'resnet18', 'mobilenet_v2' 等）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载 class_names
        self.model_dir = os.path.dirname(model_path)
        if Num_classes is  None:
            class_names_path = os.path.join(self.model_dir, 'class_names.json')
            if not os.path.exists(class_names_path):
                raise FileNotFoundError(f'Class names file not found at {class_names_path}')
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)

            num_classes = len(self.class_names)
        else :
            num_classes = Num_classes
        # 初始化模型并加载权重
        self.model = self._initialize_model(model_name, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        # 定义图像转换
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _initialize_model(self, model_name, num_classes):
        """
        根据模型名称初始化模型，并修改最后一层以适应特定类别数量。
        :param model_name: 模型名称
        :param num_classes: 类别数量
        """
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=False)
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
        elif model_name == 'densenet121':
            model = models.densenet121(pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=False)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

        else:
            raise ValueError("Unsupported model name.")
        return model

    def predict(self, input_path):
        """
        预测单张图像或文件夹中所有图像的类别。
        :param input_path: 图像文件路径或文件夹路径
        :return: 预测结果（字典，包含每个图像的预测类别和概率）
        """
        if os.path.isfile(input_path):
            # 输入是单个文件，预测该文件的类别
            # return {input_path: self._predict_single_image(input_path)}
            return  self._predict_single_image(input_path)
        elif os.path.isdir(input_path):
            # 输入是一个文件夹，预测文件夹中所有图像的类别
            results = {}
            for image_name in os.listdir(input_path):
                image_path = os.path.join(input_path, image_name)
                if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    results[image_name] = self._predict_single_image(image_path)
            return results
        else:
            raise ValueError(f"Invalid input path: {input_path}. It should be a valid image file or a directory.")

    def _predict_single_image(self, image_path):
        """
        辅助函数，用于预测单张图像的类别。
        :param image_path: 图像文件路径
        :return: 预测类别及其概率
        """
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_prob = probabilities[0][predicted_class].item()

        return predicted_class, predicted_prob


# 使用示例
if __name__ == "__main__":
    # 初始化加载本地模型
    model_predictor = LocalModelPredictor(
        model_path='path_to_your_model.pth',
        model_name='mobilenet_v2'
    )

    # 对单张图像进行预测
    result = model_predictor.predict('path_to_image.png')
    print(result)

    # 对文件夹中的所有图像进行预测
    results = model_predictor.predict('path_to_image_folder/')
    for image_name, (pred_class, prob) in results.items():
        print(f'Image: {image_name} | Predicted Class: {pred_class} | Probability: {prob:.4f}')
