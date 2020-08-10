import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from constants import PAD_WIDTH, PAD_HEIGHT
from utils import pad


class PersonDataset(Dataset):
    def __init__(self, images_path, masks_path=None, transforms=None):
        super().__init__()

        ids = []
        for filename in os.listdir(images_path):
            ids.append(Path(filename).name)

        self.ids = ids
        self.images_path = images_path
        self.masks_path = masks_path
        self.transforms = transforms

        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        filename = self.ids[idx]
        image = cv2.cvtColor(cv2.imread(f"{self.images_path}/{filename}"), cv2.COLOR_BGR2RGB)

        if self.masks_path:
            mask = cv2.cvtColor(cv2.imread(f"{self.masks_path}/{filename}"), cv2.COLOR_BGR2RGB)

            if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                image, mask = augmented["image"], augmented["mask"]

            mask = pad(mask, PAD_HEIGHT, PAD_WIDTH)
            image = pad(image, PAD_HEIGHT, PAD_WIDTH)

            mask = torch.from_numpy(mask[:, :, 0:1] // 255).float().permute([2, 0, 1])
            image = self.normalize(torch.from_numpy(image / 255).float().permute([2, 0, 1]))

            return image, mask

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        image = pad(image, PAD_WIDTH, PAD_HEIGHT)
        image = self.normalize(torch.from_numpy(image / 255).float().permute([2, 0, 1]))

        return image
