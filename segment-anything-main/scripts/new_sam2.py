import os
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import jaccard_score  # 用于计算 IoU

# 类别和调色板
classes = [
    'background', 'Matrix', 'Hyaline', 'Andesite', 'Tuffin', 'Clay',
    'Rhyolitic', 'Quartz', 'Quartzite', 'Kaolinite', 'Calcite',
    'Montmorilonite', 'Siderite', 'Cordierite', 'Plagioclase', 'Sandstone', 'Oolith',
    'Spherulite', 'Glauconite', 'Orthoclase', 'Anhydrite', 'Granite',
    'Biotite', 'Microcline', 'Pore', 'Pyroclast', 'Quartzitic'
]

palette = [
    [0, 0, 0], [169, 169, 169], [176, 196, 222], [119, 136, 153], [139, 69, 19],
    [222, 184, 135], [210, 105, 30], [255, 255, 255], [190, 190, 190], [240, 230, 140],
    [250, 235, 215], [244, 164, 96], [181, 101, 29], [72, 61, 139], [255, 248, 220],
    [210, 180, 140], [205, 133, 63], [250, 128, 114], [127, 255, 0], [255, 228, 181],
    [192, 192, 192], [128, 128, 128], [139, 69, 19], [244, 164, 96], [169, 169, 169],
    [255, 69, 0], [190, 190, 190]
]

# 设置SAM模型的路径和设备
sam_checkpoint = r"C:\Users\Admin\Desktop\XueRenworkplace\NotUseCoda\segment-anything-Main\segment-anything-Main\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的 SAM 模型
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_model.to(device)

# 添加语义输出层（假设要细调的是最后的输出层）
num_classes = len(classes)
sam_model.image_encoder.fc = nn.Conv2d(256, num_classes, kernel_size=(1, 1))  # 修改输出层为预测指定类别
sam_model.to(device)

# 冻结大部分参数，只训练最后的输出层
for param in sam_model.parameters():
    param.requires_grad = False

for param in sam_model.image_encoder.fc.parameters():
    param.requires_grad = True  # 仅训练新的语义输出层

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(sam_model.image_encoder.fc.parameters(), lr=1e-4)


# 自定义数据集类
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_filenames = sorted(os.listdir(images_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.images_dir, image_filename)
        label_path = os.path.join(self.labels_dir, image_filename.replace('.jpg', '.png'))

        # 读取图像和标签
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# 数据预处理，调整为SAM模型预期的1024x1024大小
transform = transforms.Compose([
    transforms.ToPILImage(),  # 将numpy数组转换为PIL图像
    transforms.Resize((1024, 1024)),  # 将图像调整为1024x1024
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据
images_dir = r"C:\Users\Admin\Fengworkplace\pycharmworkplace\Start\data\images_with_labels\images"
labels_dir = r"C:\Users\Admin\Fengworkplace\pycharmworkplace\Start\data\images_with_labels\labels"
train_dataset = SegmentationDataset(images_dir, labels_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)


# 微调SAM模型并评价
def train_sam_model(model, train_loader, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        iou_scores = []  # IoU 评价
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model.image_encoder(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算IoU
            predicted_classes = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
            true_classes = labels.cpu().numpy().flatten()
            iou = jaccard_score(true_classes, predicted_classes, average='macro', zero_division=1)
            iou_scores.append(iou)

        avg_loss = running_loss / len(train_loader)
        avg_iou = np.mean(iou_scores)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}")

        # 保存每个epoch后的模型
        torch.save(model.state_dict(), f'sam_model_epoch_{epoch + 1}.pth')


# 开始训练
train_sam_model(sam_model, train_loader, num_epochs=10)


# 生成预测和可视化结果
def generate_semantic_masks(image_path, output_dir):
    # 读取输入图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将图像转换为Tensor并进行预测
    image_tensor = torch.tensor(image_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    semantic_output = sam_model.image_encoder(image_tensor)
    semantic_classes = torch.argmax(semantic_output, dim=1).squeeze(0).cpu().numpy()

    # 初始化用于存储可视化结果的图像
    annotated_image = np.zeros_like(image_rgb)

    # 创建图例，用于每个类别的颜色
    legend_patches = []

    for i, class_name in enumerate(classes):
        legend_patches.append(mpatches.Patch(color=np.array(palette[i]) / 255, label=class_name))

    # 为每个类别附加颜色到语义分割结果
    for i, color in enumerate(palette):
        annotated_image[semantic_classes == i] = color

    # 可视化图像并添加图例
    plt.figure(figsize=(10, 10))

    plt.imshow(annotated_image)
    plt.title('Semantic Segmentation Result with Legend')
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.axis('off')

    # 保存输出结果
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_annotated.png'))
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # 保存新的标签图像
    output_mask_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_semantic.png'))
    cv2.imwrite(output_mask_path, semantic_classes)

    return annotated_image, semantic_classes


# 示例运行
image_path = r"C:\Users\Admin\Fengworkplace\pycharmworkplace\Start\data\images_with_labels\images\1-1-1-6.jpg"
output_dir = r"C:\Users\Admin\Fengworkplace\pycharmworkplace\Start\data\images_with_labels\output"
generate_semantic_masks(image_path, output_dir)
# Traceback (most recent call last):
#   File "C:\Users\Admin\Desktop\XueRenworkplace\NotUseCoda\segment-anything-Main\segment-anything-Main\scripts\new_sam2.py", line 136, in <module>
#     train_sam_model(sam_model, train_loader, num_epochs=10)
#   File "C:\Users\Admin\Desktop\XueRenworkplace\NotUseCoda\segment-anything-Main\segment-anything-Main\scripts\new_sam2.py", line 112, in train_sam_model
#     loss = criterion(outputs, Tags_For_testing_and_evaluation_purposes_only)
#   File "C:\Users\Admin\AppData\Roaming\Python\Python38\site-packages\torch\nn\modules\module.py", line 1518, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "C:\Users\Admin\AppData\Roaming\Python\Python38\site-packages\torch\nn\modules\module.py", line 1527, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "C:\Users\Admin\AppData\Roaming\Python\Python38\site-packages\torch\nn\modules\loss.py", line 1179, in forward
#     return F.cross_entropy(input, target, weight=self.weight,
#   File "C:\Users\Admin\AppData\Roaming\Python\Python38\site-packages\torch\nn\functional.py", line 3053, in cross_entropy
#     return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
# RuntimeError: input and target batch or spatial sizes don't match: target [2, 1024, 1280], input [2, 256, 64, 64]
