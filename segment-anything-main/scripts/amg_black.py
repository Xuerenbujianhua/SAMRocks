import cv2  # type: ignore
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import os
from typing import Any, Dict, List
from natsort import natsorted

# 设置命令行参数解析
parser = argparse.ArgumentParser(
    description=(
        "Runs automatic masks generation on an input image or directory of images, "
        "and outputs masks as either a single combined PNG or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=False,
    default="input_images/",
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=False,
    default="output_images_black/",
    help=(
        "Path to the directory where masks will be output. Output will be either a combined PNG per image "
        "or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=False,
    default="vit_h",
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=False,
    default=r"C:\Users\Admin\Desktop\XueRenworkplace\NotUseCoda\segment-anything-Main\segment-anything-Main\sam_vit_h_4b8939.pth",
    help="The path to the SAM checkpoint to use for masks generation.",
)

# parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
parser.add_argument("--device", type=str, default="cpu", help="The device to run generation on.")
parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a combined PNG. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the masks more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate masks.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, masks generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-masks-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected masks regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)

# 保存组合后的掩码图像
def save_combined_mask(masks: List[Dict[str, Any]], path: str, image_shape) -> None:
    combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for mask_data in masks:
        mask = mask_data["segmentation"]

        if mask.dtype == np.bool_:
            mask = mask.astype(np.uint8)

        mask_resized = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

        combined_mask = cv2.bitwise_or(combined_mask, mask_resized * 255)

    cv2.imwrite(path, combined_mask)


# 获取掩码生成器参数
def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

# 分批处理图片
def process_images_in_batches(targets: List[str], generator, output_dir: str, batch_size: int = 100) -> None:
    total_images = len(targets)
    image_counter = 1  # 用于记录当前处理的图片编号

    for i in range(0, total_images, batch_size):
        batch = targets[i:i + batch_size]
        for t in batch:
            print(f"Processing image {image_counter}/{total_images}: '{t}'...")
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                image_counter += 1
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            masks = generator.generate(image)

            base = os.path.basename(t)
            base = os.path.splitext(base)[0]
            save_path = os.path.join(output_dir, f"{base}.png")#保存的文件名称
            save_combined_mask(masks, save_path, image.shape)

            image_counter += 1  # 处理完一张图片，计数器加1

        print(f"Batch {i // batch_size + 1}/{(total_images + batch_size - 1) // batch_size} processed.")

# 主函数
def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = natsorted(
            [f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))]
        )
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    process_images_in_batches(targets, generator, args.output)

    print("Done!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
