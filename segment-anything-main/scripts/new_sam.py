import os
import numpy as np
import torch
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from sklearn.metrics import jaccard_score  # 用于计算 IoU

# 类别名称
classes = [
    'background', 'Matrix', 'Hyaline', 'Andesite', 'Tuffin', 'Clay',
    'Rhyolitic', 'Quartz', 'Quartzite', 'Kaolinite', 'Calcite',
    'Montmorilonite', 'Siderite', 'Cordierite', 'Plagioclase', 'Sandstone', 'Oolith',
    'Spherulite', 'Glauconite', 'Orthoclase', 'Anhydrite', 'Granite',
    'Biotite', 'Microcline', 'Pore', 'Pyroclast', 'Quartzitic'
]

# 调色板（每个类别对应的颜色）
palette = [
    [0, 0, 0], [169, 169, 169], [176, 196, 222], [119, 136, 153], [139, 69, 19],
    [222, 184, 135], [210, 105, 30], [255, 255, 255], [190, 190, 190], [240, 230, 140],
    [250, 235, 215], [244, 164, 96], [181, 101, 29], [72, 61, 139], [255, 248, 220],
    [210, 180, 140], [205, 133, 63], [250, 128, 114], [127, 255, 0], [255, 228, 181],
    [192, 192, 192], [128, 128, 128], [139, 69, 19], [244, 164, 96], [169, 169, 169],
    [255, 69, 0], [190, 190, 190]
]

# 设置 SAM 模型的路径和设备
sam_checkpoint = r"C:\Users\Admin\Desktop\XueRenworkplace\NotUseCoda\segment-anything-Main\segment-anything-Main\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练的 SAM 模型
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

# 创建 SAM 自动掩码生成器
mask_generator = SamAutomaticMaskGenerator(sam)

# 加载语义分割模型（DeepLabV3）
num_classes = 26  # 假设你有 26 个类别
semantic_model = deeplabv3_resnet50(pretrained=True)

# 替换模型的分类头，适应自定义类别数
semantic_model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
semantic_model = semantic_model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(semantic_model.classifier.parameters(), lr=1e-4)


# 自定义带有图像和掩码标签的Dataset类
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, num_classes, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_filenames = sorted(os.listdir(images_dir))  # 获取图像文件名
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.images_dir, image_filename)
        label_path = os.path.join(self.labels_dir, image_filename.replace('.jpg', '.png'))  # 假设标签为 png

        # 读取图像和标签
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # 单通道掩码

        # 检查标签中的类别索引是否超出范围，并限制标签值
        label = np.clip(label, 0, self.num_classes - 1)  # 保证标签范围在 [0, num_classes - 1]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# 加载带标签的训练数据
images_dir = r"C:\Users\Admin\Fengworkplace\pycharmworkplace\Start\data\images_with_labels\images"
labels_dir = r"C:\Users\Admin\Fengworkplace\pycharmworkplace\Start\data\images_with_labels\labels"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = SegmentationDataset(images_dir, labels_dir, num_classes=num_classes, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)


# 使用少量标注数据微调语义分割模型，并输出中间结果
def train_model(semantic_model, train_loader, num_epochs=5, save_interval=1):
    semantic_model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        iou_scores = []  # 用于保存 IoU 分数

        # 使用 tqdm 显示进度条
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = semantic_model(images)["out"]
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算每个 batch 的 IoU
            predicted_classes = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
            true_classes = labels.cpu().numpy().flatten()
            iou = jaccard_score(true_classes, predicted_classes, average='macro', zero_division=1)
            iou_scores.append(iou)

        avg_loss = running_loss / len(train_loader)
        avg_iou = np.mean(iou_scores)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}")

        # 保存中间模型权重
        if (epoch + 1) % save_interval == 0:
            torch.save(semantic_model.state_dict(), f'./checkpoint/semantic_model_epoch_{epoch + 1}.pth')


# 训练模型并保存中间结果
train_model(semantic_model, train_loader, num_epochs=500, save_interval=20)


# 生成 SAM 掩码、并附加类别信息，输出图像和中间结果
def generate_semantic_masks(image_path, output_dir):
    # 读取输入图像并生成 SAM 掩码
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用 SAM 自动生成分割掩码
    masks = mask_generator.generate(image_rgb)

    # 切换语义分割模型到评估模式
    semantic_model.eval()

    # 通过微调后的语义分割模型生成语义信息
    with torch.no_grad():
        image_tensor = torch.tensor(image_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
        semantic_output = semantic_model(image_tensor)["out"]
        semantic_classes = torch.argmax(semantic_output, dim=1).squeeze(0).cpu().numpy()

    # 初始化用于存储可视化结果的图像
    annotated_image = np.zeros_like(image_rgb)
    sam_visualization = np.zeros_like(image_rgb)

    # 创建图例，用于每个类别的颜色
    legend_patches = []

    for i, class_name in enumerate(classes):
        legend_patches.append(mpatches.Patch(color=np.array(palette[i]) / 255, label=class_name))

    # SAM 掩码的可视化
    for mask_data in masks:
        mask = mask_data["segmentation"]
        for c in range(3):  # 将掩码应用到可视化图像上
            sam_visualization[:, :, c] = np.where(mask == 1, 255, sam_visualization[:, :, c])

    # 输出 SAM 分割结果
    output_sam_mask_path = os.path.join(output_dir, "sam_mask_visualization.png")
    cv2.imwrite(output_sam_mask_path, sam_visualization)

    # 为每个类别附加颜色到语义分割结果
    for i, color in enumerate(palette):
        annotated_image[semantic_classes == i] = color

    # 可视化图像并添加图例
    plt.figure(figsize=(10, 10))

    # 绘制 SAM 掩码结果
    plt.subplot(1, 3, 1)
    plt.imshow(sam_visualization)
    plt.title('SAM Segmentation Result')
    plt.axis('off')

    # 绘制语义分割结果
    plt.subplot(1, 3, 2)
    plt.imshow(annotated_image)
    plt.title('Semantic Segmentation Result')
    plt.axis('off')

    # 绘制带有图例的语义分割图像
    plt.subplot(1, 3, 3)
    plt.imshow(annotated_image)
    plt.title('Semantic Segmentation with Legend')
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
# Pixel value counts (excluding 0):
# Pixel Value 0: 65388 pixels
# Pixel Value 2: 792973 pixels
# Pixel Value 3: 141170 pixels
# Pixel Value 7: 207968 pixels
# Pixel Value 25: 103221 pixels
#效果较差
# Epoch [499/500], Loss: 0.1098, IoU: 0.5884
# Epoch [500/500], Loss: 0.1064, IoU: 0.6866