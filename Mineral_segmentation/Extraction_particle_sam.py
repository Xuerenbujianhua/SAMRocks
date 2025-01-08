import cv2  # type: ignore 
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
import random
from typing import Any, Dict, List
from natsort import natsorted
import shutil

class MaskGenerator:
    def __init__(self, input_path: str, model_type: str = "vit_h",
                 checkpoint: str = r"..\checkpoints\sam_vit_h_4b8939.pth",
                 device: str = "cpu",
                 convert_to_rle: bool = False,
                 output_folder: str = "images",  # 新增参数：结果保存路径
                 **kwargs):
        self.input_path = input_path
        self.output_folder = output_folder
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device
        self.convert_to_rle = convert_to_rle
        self.amg_kwargs = kwargs
        self.generator = self._initialize_generator()

    def _initialize_generator(self) -> SamAutomaticMaskGenerator:
        print("Loading model ")
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        _ = sam.to(device=self.device)
        output_mode = "coco_rle" if self.convert_to_rle else "binary_mask"
        return SamAutomaticMaskGenerator(sam, output_mode=output_mode, **self.amg_kwargs)

    @staticmethod
    def generate_random_color() -> tuple:
        return tuple(random.randint(0, 255) for _ in range(3))

    @staticmethod
    def save_colored_mask(masks: List[Dict[str, Any]], path: str, image_shape) -> None:
        colored_mask = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        for mask_data in masks:
            mask = mask_data["segmentation"]
            if mask.dtype == np.bool_:
                mask = mask.astype(np.uint8)
            mask_resized = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
            color = MaskGenerator.generate_random_color()
            colored_layer = np.zeros_like(colored_mask)
            for i in range(3):
                colored_layer[:, :, i] = mask_resized * color[i]
            colored_mask = cv2.add(colored_mask, colored_layer)
        cv2.imwrite(path, colored_mask)

    def process_images(self) -> None:
        if os.path.isdir(self.input_path):  # 如果是文件夹
            os.makedirs(self.output_folder, exist_ok=True)
            targets = natsorted([f for f in os.listdir(self.input_path) if not os.path.isdir(os.path.join(self.input_path, f))])
            targets = [os.path.join(self.input_path, f) for f in targets]
        elif os.path.isfile(self.input_path):  # 如果是单张图片
            os.makedirs(self.output_folder, exist_ok=True)
            targets = [self.input_path]
        else:
            print(f"输入 '{self.input_path}' 无效，请检查路径。")
            return

        num = 1
        for t in targets:
            print(f"Processing '{t}'...{num}")
            image = cv2.imread(t)
            if image is None:
                print(f"无法加载 '{t}'，跳过此文件...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = self.generator.generate(image)
            if not masks:
                print(f"未生成任何掩码文件，跳过 '{t}'...")
                continue
            base = os.path.basename(t)
            base = os.path.splitext(base)[0]
            save_path = os.path.join(self.output_folder, f"{base}.png")
            self.save_colored_mask(masks, save_path, image.shape)
            num += 1
        print(f"Finished! - Process {num-1} images")

    def move_images(self):
        input_folder = self.input_path
        images_folder = os.path.join(input_folder, 'images')
        os.makedirs(images_folder, exist_ok=True)
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        for item in os.listdir(input_folder):
            item_path = os.path.join(input_folder, item)
            if os.path.isfile(item_path) and os.path.splitext(item)[1].lower() in image_extensions:
                shutil.move(item_path, images_folder)

    def move_sams(self):
        src_root = self.input_path
        dst_dir = os.path.dirname(src_root)
        for root, dirs, files in os.walk(src_root):
            if 'sams' in dirs:
                src_path = os.path.join(root, 'sams')
                shutil.move(src_path, dst_dir)
                print(f"Moved directory  {src_path} to {dst_dir}")
                break
        else:
            print("未找到名为 'sams' 的文件夹。")
