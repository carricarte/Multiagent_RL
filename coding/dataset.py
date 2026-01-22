from torch.utils.data import Dataset
from os.path import join, split
from os import listdir
from json import load
import numpy as np
import torch
from pycocotools import mask as mask_utils


def decode_mask_format(rle_dict):
    """
    Decode RLE mask for Dataset.

    Args:
        rle_dict: RLE dictionary with 'counts' and 'size'

    Returns:
        Torch tensor [H, W] - DataLoader will add batch dimension
    """
    # Decode to numpy
    binary_mask = mask_utils.decode(rle_dict)  # [H, W]

    # Convert to torch tensor (no extra dimensions)
    mask_tensor = torch.from_numpy(binary_mask).float()  # [H, W]

    return mask_tensor



def get_embedding(img_name, embb_file_names):
    if not embb_file_names:
        print("No embedding files available")
        return ""

    target_num = int(img_name.split("sa_")[1])

    def get_file_number(file_path):
        _, fname = split(file_path)
        return int(fname.split("sa_")[1].split(".npz")[0])

    # Ensure sorted
    embb_file_names = sorted(embb_file_names, key=get_file_number)

    # Binary search: first file_number >= target_num
    left, right = 0, len(embb_file_names) - 1
    target_idx = None

    while left <= right:
        mid = (left + right) // 2
        mid_num = get_file_number(embb_file_names[mid])

        if mid_num >= target_num:
            target_idx = mid
            right = mid - 1
        else:
            left = mid + 1

    if target_idx is None:
        print(f"Embedding not found (number too large): {img_name}")
        return ""

    try:
        with np.load(embb_file_names[target_idx]) as f:
            return f[img_name]
    except KeyError:
        print(f"Embedding not found in expected file: {img_name}")
        return ""


class MyDataset(Dataset):

    def __init__(self, input_dir):

        self._embedding_files = []
        self._mask_files = []

        [self._embedding_files.append(join(input_dir, emb_file)) for emb_file in listdir(input_dir) if "embedding" in emb_file and '._' not in emb_file and emb_file.endswith(".npz")]
        [self._mask_files.append(join(input_dir, mask_file)) for mask_file in listdir(input_dir) if "challenging_mask" in mask_file and '._' not in mask_file and mask_file.endswith(".json")]

        self._embedding_files.sort()
        self._mask_files.sort()
        self._max_h = 3000
        self._max_w = 3000


        with open(self._mask_files[0], "r") as f:

            self.annotations = load(f)

        for mask_file in self._mask_files[1::]:

            with open(mask_file, "r") as f:

                self.annotations = np.concatenate((self.annotations, load(f)))


    def __len__(self):

        return len(self.annotations)

    def __getitem__(self, idx):

        annotation = self.annotations[idx]
        p_segmentation = annotation['p_segmentation']
        gt_segmentation = annotation['gt_segmentation']

        p_segmentation = decode_mask_format(p_segmentation)
        gt_segmentation = decode_mask_format(gt_segmentation)

        h, w = p_segmentation.shape
        p_padded_segmentation = torch.zeros((self._max_h, self._max_w), dtype=p_segmentation.dtype, device=p_segmentation.device)
        gt_padded_segmentation = torch.zeros((self._max_h, self._max_w), dtype=gt_segmentation.dtype, device=gt_segmentation.device)
        p_padded_segmentation[:h, :w] = p_segmentation
        gt_padded_segmentation[:h, :w] = gt_segmentation

        image_name = annotation['image']
        embedding = torch.tensor(get_embedding(image_name, self._embedding_files)).view(64, 64, 768)

        # return embedding
        return embedding, p_padded_segmentation, gt_padded_segmentation