from collections import defaultdict
from logging import getLogger
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy
from torchvision.ops import box_convert, box_iou

from .utils import Cub2011Eval, mean, std

logger = getLogger(__name__)


@torch.no_grad()
def evaluate_accuracy(net: nn.Module,
                      device: torch.device = torch.device("cpu"),
                      input_size: tuple[int, int] = (224, 224,)):
    normalize = T.Normalize(mean=mean, std=std)
    transform = T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        normalize
    ])

    test_dataset = Cub2011Eval("datasets", train=False, transform=transform)    # CUB test dataset
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=8, pin_memory=True, drop_last=False, shuffle=True)

    net.to(device)
    net.eval()

    mca = MulticlassAccuracy(num_classes=200, average="micro").to(device)
    for b, batch in enumerate(tqdm(test_loader)):
        images, targets, img_ids = tuple(item.to(device=device) for item in batch)
        B, _, INPUT_H, INPUT_W = images.shape
        preds, min_distances, proto_presence = net(images)
        mca(preds, targets)
    acc = mca.compute().item()
    logger.info(f"Eval accuracy: {acc:.4f}")
