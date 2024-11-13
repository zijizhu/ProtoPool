"""
Migrated from https://github.com/hqhQAQ/EvalProtoPNet/tree/main
"""

import os
import torch
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations import crop_keypoint_by_coords
from pathlib import Path
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F


def draw_point(img, point, bbox_size=10, color=(0, 0, 255)):
    img[point[1] - bbox_size // 2: point[1] + bbox_size // 2, point[0] - bbox_size // 2: point[0] + bbox_size // 2] = color

    return img

def in_bbox(loc, bbox):
    return loc[0] >= bbox[0] and loc[0] <= bbox[1] and loc[1] >= bbox[2] and loc[1] <= bbox[3]

data_root = 'datasets/CUB_200_2011'

img_txt = os.path.join(data_root, 'images.txt')
cls_txt = os.path.join(data_root, 'image_class_labels.txt')
bbox_txt = os.path.join(data_root, 'bounding_boxes.txt')
train_txt = os.path.join(data_root, 'train_test_split.txt')
part_cls_txt = os.path.join(data_root, 'parts', 'parts.txt')
part_loc_txt = os.path.join(data_root, 'parts', 'part_locs.txt')

# id_to_path: Get the image path of each image according to its image id
id_to_path = {}
with open(img_txt, 'r') as f:
    img_lines = f.readlines()
for img_line in img_lines:
    img_id, img_path = int(img_line.split(' ')[0]), img_line.split(' ')[1][:-1]
    img_folder, img_name = img_path.split('/')[0], img_path.split('/')[1]
    id_to_path[img_id] = (img_folder, img_name)

# id_to_bbox: Get the bounding box annotation (bird part) of each image according to its image id
id_to_bbox = {}
with open(bbox_txt, 'r') as f:
    bbox_lines = f.readlines()
for bbox_line in bbox_lines:
    cts = bbox_line.split(' ')
    img_id, bbox_x, bbox_y, bbox_width, bbox_height = int(cts[0]), int(cts[1].split('.')[0]), int(cts[2].split('.')[0]), int(cts[3].split('.')[0]), int(cts[4].split('.')[0])
    bbox_x2, bbox_y2 = bbox_x + bbox_width, bbox_y + bbox_height
    id_to_bbox[img_id] = (bbox_x, bbox_y, bbox_x2, bbox_y2)

# cls_to_id: Get the image ids of each class
cls_to_id = {}
with open(cls_txt, 'r') as f:
    cls_lines = f.readlines()
for cls_line in cls_lines:
    img_id, cls_id = int(cls_line.split(' ')[0]), int(cls_line.split(' ')[1]) - 1   # 0 -> 199
    if cls_id not in cls_to_id.keys():
        cls_to_id[cls_id] = []
    cls_to_id[cls_id].append(img_id)

# id_to_train: Get the training/test label of each image according to its image id
id_to_train = {}
with open(train_txt, 'r') as f:
    train_lines = f.readlines()
for train_line in train_lines:
    img_id, is_train = int(train_line.split(' ')[0]), int(train_line.split(' ')[1][:-1])
    id_to_train[img_id] = is_train

# part_id_to_part: Get the part name of each object part according to its part id
part_id_to_part = {}
with open(part_cls_txt, 'r') as f:
    part_cls_lines = f.readlines()
for part_cls_line in part_cls_lines:
    id_len = len(part_cls_line.split(' ')[0])
    part_id, part_name = part_cls_line[:id_len], part_cls_line[id_len + 1:]
    part_id_to_part[part_id] = part_name
part_num = len(part_id_to_part.keys())

# id_to_part_loc: Get the part annotations of each image according to its image id
id_to_part_loc = {}
with open(part_loc_txt, 'r') as f:
    part_loc_lines = f.readlines()
for part_loc_line in part_loc_lines:
    content = part_loc_line.split(' ')
    img_id, part_id, loc_x, loc_y, visible = int(content[0]), int(content[1]), int(float(content[2])), int(float(content[3])), int(content[4])
    if img_id not in id_to_part_loc.keys():
        id_to_part_loc[img_id] = []
    if visible == 1:
        id_to_part_loc[img_id].append([part_id, loc_x, loc_y])


class Cub2011Eval(Dataset):
    base_folder = 'test_cropped'

    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, 'cub200_cropped', self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, 'cub200_cropped', self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        img_id = sample.img_id

        if self.transform is not None:
            img = self.transform(img)

        return img, target, img_id

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def preprocess_input_function(x):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    '''
    return preprocess(x, mean=mean, std=std)


def undo_preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y


def undo_preprocess_input_function(x):
    '''
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    '''
    return undo_preprocess(x, mean=mean, std=std)


class CUBEvalDataset(ImageFolder):
    def __init__(self,
                 images_root: str,
                 annotations_root: str,
                 normalization: bool = True,
                 input_size: int = 224):
        transforms = [A.Resize(width=input_size, height=input_size)]
        transforms += [A.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))] if normalization else []

        super().__init__(
            root=images_root,
            transform=A.Compose(
                transforms,
                keypoint_params=A.KeypointParams(
                    format='xy',
                    label_fields=None,
                    remove_invisible=True,
                    angle_in_degrees=True
                )
            ),
            target_transform=None
        )
        self.input_size = input_size
        annotations_root = Path("datasets") / "CUB_200_2011"

        path_df = pd.read_csv(annotations_root / "images.txt", header=None, names=["image_id", "image_path"], sep=" ")
        bbox_df = pd.read_csv(annotations_root / "bounding_boxes.txt", header=None, names=["image_id", "x", "y", "w", "h"], sep=" ")
        self.bbox_df = path_df.merge(bbox_df, on="image_id")
        self.part_loc_df = pd.read_csv(annotations_root / "parts" / "part_locs.txt", header=None, names=["image_id", "part_id", "kp_x", "kp_y", "visible"], sep=" ")
        
        attributes_np = np.loadtxt(annotations_root / "attributes" / "class_attribute_labels_continuous.txt")
        self.attributes = F.normalize(torch.tensor(attributes_np, dtype=torch.float32), p=2, dim=-1)

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im = np.array(Image.open(im_path).convert("RGB"))

        row = self.bbox_df[self.bbox_df["image_path"] == "/".join(Path(im_path).parts[-2:])].iloc[0]
        image_id = row["image_id"]
        bbox_coords = row[["x", "y", "w", "h"]].values.flatten()
        
        mask = self.part_loc_df["image_id"] == image_id
        keypoints = self.part_loc_df[mask][["kp_x", "kp_y"]].values

        keypoints_cropped = [crop_keypoint_by_coords(keypoint=tuple(kp) + (None, None,), crop_coords=bbox_coords[:2]) for kp in keypoints]
        keypoints_cropped = [(np.clip(x, 0, self.input_size), np.clip(y, 0, self.input_size),) for x, y, _, _ in keypoints_cropped]
        
        transformed = self.transform(image=im, keypoints=keypoints_cropped)
        transformed_im, transformed_keypoints = transformed["image"], transformed["keypoints"]
        
        return to_tensor(transformed_im), torch.tensor(transformed_keypoints, dtype=torch.float32), label, self.attributes[label, :], index
    