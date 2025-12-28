import numpy as np
import json
import cv2
from pycocotools import mask as mask_utils
from pathlib import Path

def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def calculate_dice(pred_mask, gt_mask):
    """Calculate Dice coefficient."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    if total == 0:
        return 0.0
    return 2 * intersection / total

def calculate_metrics(pred_mask, gt_mask):
    """Calculate comprehensive metrics."""
    iou = calculate_iou(pred_mask, gt_mask)
    dice = calculate_dice(pred_mask, gt_mask)

    # intersection = np.logical_and(pred_mask, gt_mask).sum()
    # pred_area = pred_mask.sum()
    # gt_area = gt_mask.sum()

    # precision = intersection / pred_area if pred_area > 0 else 0
    # recall = intersection / gt_area if gt_area > 0 else 0
    # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'iou': iou,
        'dice': dice
    }

def decode_sa1b_mask(annotation):
    """Decode SA-1B RLE mask to binary mask."""
    segmentation = annotation['segmentation']
    if isinstance(segmentation, dict):
        mask = mask_utils.decode(segmentation)
    else:
        raise ValueError(f"Unexpected segmentation format: {type(segmentation)}")
    return mask.astype(bool)


def load_sa1b_image_and_annotations(json_path):
    """Load image and annotations from SA-1B format."""
    json_path = Path(json_path)

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Load image
    img_path = json_path.with_suffix('.jpg')
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image, data