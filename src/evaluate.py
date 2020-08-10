import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PersonDataset
from model import UnetResnet34
from utils import calculate_iou_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', help='Path to eval images', required=True, type=str)
    parser.add_argument('--masks', help='Path to eval masks', required=True, type=str)
    parser.add_argument('--model', help='Path to model', required=True, type=str)
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training (default: 8)')

    args = parser.parse_args()

    dataset = PersonDataset(
        images_path=args.val_images,
        masks_path=args.val_masks,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)

    device = torch.device("cuda:0")

    unet = UnetResnet34()
    unet.load_state_dict(torch.load(args.model))
    unet.to(device)

    unet.eval()
    predictions, masks = [], []
    for image, mask in tqdm(dataloader):
        prediction = unet(image.to(device))

        prediction = torch.sigmoid(prediction)
        predicted_mask = prediction.squeeze(1)
        masks = masks.squeeze(1)

        predictions.append(predicted_mask.detach().cpu().numpy())
        masks.append(masks.numpy())

    iou_score, threshold = calculate_iou_score(np.concatenate(masks, 0), np.concatenate(predictions, 0))
    print(f"IoU score: {iou_score}, threshold: {threshold}")
