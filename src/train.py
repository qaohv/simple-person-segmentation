import argparse

import albumentations as A
import numpy as np
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from dataset import PersonDataset
from early_stopping import EarlyStopping
from focal_loss import FocalLoss2d
from model import UnetResnet34
from utils import calculate_iou_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-images', help='Path to train images', required=True, type=str)
    parser.add_argument('--train-masks', help='Path to train masks', required=True, type=str)
    parser.add_argument('--val-images', help='Path to val images', required=True, type=str)
    parser.add_argument('--val-masks', help='Path to val masks', required=True, type=str)

    parser.add_argument('--lr', help='SGD learning rate', type=float, default=0.01)
    parser.add_argument('--momentum', help='Momentum for SGD optimizer', type=float, default=0.9)
    parser.add_argument('--weight-decay', help='Weight decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training (default: 8)')

    parser.add_argument('--rop-reduce-factor', help='ROP scheduler reduce factory', type=float, default=0.5)
    parser.add_argument('--rop-patience', help='ROP scheduler patience', type=int, default=6)
    parser.add_argument('--early-stopping-patience', help='Early stopping based on iou score patience', type=int,
                        default=10)
    parser.add_argument('--logdir', help='directory for tensorboard logs', type=str)

    args = parser.parse_args()

    writer = SummaryWriter(args.logdir)

    train = PersonDataset(
        images_path=args.train_images,
        masks_path=args.train_masks,
        transforms=A.Compose([
            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.NoOp(),
            ], p=1),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.07, alpha_affine=120 * 0.03),
                A.ShiftScaleRotate(shift_limit=0.05, rotate_limit=5),
                A.NoOp(),
            ], p=1),
            A.JpegCompression(quality_lower=50, quality_upper=100, p=.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=.5),
        ])
    )

    val = PersonDataset(
        images_path=args.val_images,
        masks_path=args.val_masks,
    )

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True, num_workers=3)

    device = torch.device("cuda:0")
    unet = UnetResnet34().to(device)

    criterion = FocalLoss2d()

    optimizer = torch.optim.SGD(unet.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'max',
                                  factor=args.rop_reduce_factor,
                                  patience=args.rop_patience, verbose=True)
    early_stopping = EarlyStopping(args.early_stopping_patience, mode='max')

    for epoch in range(args.epochs):
        unet.train()
        train_loss = []

        for images, masks in tqdm(train_loader):
            optimizer.zero_grad()

            images, masks = images.to(device), masks.to(device)
            prediction = unet(images)

            predicted_mask = prediction.squeeze(1)
            masks = masks.squeeze(1)
            loss = criterion(predicted_mask, masks)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        unet.eval()
        val_loss, val_predictions, val_masks = [], [], []
        for images, masks in tqdm(val_loader):
            images, masks = images.to(device), masks.to(device)
            prediction = unet(images)

            predicted_mask = prediction.squeeze(1)
            masks = masks.squeeze(1)
            loss = criterion(predicted_mask, masks)  # batch loss

            val_loss.append(loss.item())

            predicted_mask = torch.sigmoid(predicted_mask)
            val_predictions.append(predicted_mask.detach().cpu().numpy())
            val_masks.append(masks.detach().cpu().numpy())

        # calculate validation iou score for
        iou_score, _ = calculate_iou_score(np.concatenate(val_masks, 0), np.concatenate(val_predictions, 0))
        scheduler.step(iou_score)

        writer.add_scalar("Train loss", np.mean(train_loss), epoch)
        writer.add_scalar("Valid loss", np.mean(val_loss), epoch)
        writer.add_scalar("Valid loss", iou_score, epoch)
        print('Epoch {0} finished! train loss: {1:.5f}, '
              'val loss: {2:.5f}, val iou score: {3:.5f}'.format(epoch, np.mean(train_loss),
                                                                 np.mean(val_loss),
                                                                 iou_score))
        if epoch > 10:
            torch.save(unet.state_dict(),
                       "epoch:{}_train-loss:{:.4f}_val-loss:{:.4f}_val-iou:{:.4f}.pth".format(epoch,
                                                                                              np.mean(train_loss),
                                                                                              np.mean(val_loss),
                                                                                              iou_score))
        if early_stopping.early_stop:
            break
