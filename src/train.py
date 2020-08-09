import argparse
import numpy as np

import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from dataset import PersonDataset
from model import UnetResnet34

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-images', type=str, help='Path to train images', required=True)
    parser.add_argument('--train-masks', type=str, help='Path to train masks', required=True)
    parser.add_argument('--val-images', type=str, help='Path to val images', required=True)
    parser.add_argument('--val-masks', type=str, help='Path to val masks', required=True)

    parser.add_argument('--lr', help='SGD learning rate', type=float, default=0.01)
    parser.add_argument('--momentum', help='Momentum for SGD optimizer', type=float, default=0.9)
    parser.add_argument('--weight-decay', help='Weight decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training (default: 8)')

    parser.add_argument('--rop-reduce-factor', help='ROP scheduler reduce factory', type=float, default=0.5)
    parser.add_argument('--rop-patience', help='ROP scheduler patience', type=int, default=6)

    parser.add_argument('--logdir', metavar='logdir', type=str, help='directory for tensorboard logs')

    args = parser.parse_args()

    writer = SummaryWriter(args.logdir)

    # 1. Create dataset
    train = PersonDataset(
        images_path=args.train_images,
        masks_path=args.train_masks,
        transforms=None,
    )

    val = PersonDataset(
        images_path=args.val_images,
        masks_path=args.val_masks,
    )

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=True, num_workers=3)

    device = torch.device("cuda:0")
    unet = UnetResnet34().to(device)

    # TODO: change BCE to Focal Loss
    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.SGD(unet.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                  factor=args.rop_reduce_factor,
                                  patience=args.rop_patience, verbose=True)

    try:
        for epoch in range(args.epochs):
            # train
            unet.train()
            # todo: Add IoU calculation
            train_loss = []

            for images, masks in tqdm(train_loader):
                optimizer.zero_grad()

                images, masks = images.to(device), masks.to(device)
                prediction = unet(images)
                prediction = torch.sigmoid(prediction)

                predicted_mask = prediction.squeeze(1)
                masks = masks.squeeze(1)
                loss = criterion(predicted_mask, masks)
                train_loss.append(loss.item())  # epoch loss

                loss.backward()
                optimizer.step()

            unet.eval()
            val_loss = []
            for images, masks in tqdm(val_loader):
                images, masks = images.to(device), masks.to(device)

                # for basic unet
                prediction = unet(images)
                prediction = torch.sigmoid(prediction)

                predicted_mask = prediction.squeeze(1)
                masks = masks.squeeze(1)
                loss = criterion(predicted_mask, masks)  # batch loss

                val_loss.append(loss.item())

            # ROP
            scheduler.step(np.mean(val_loss))

            writer.add_scalar("Train loss", np.mean(train_loss), epoch)
            writer.add_scalar("Valid loss", np.mean(val_loss), epoch)

            print('Epoch {0} finished! train loss: {1:.5f}, val loss: {2:.5f}'.format(epoch,
                                                                                      np.mean(train_loss),
                                                                                      np.mean(val_loss)))
            # if epoch > 49:
            #     torch.save(unet.state_dict(),
            #                "epoch:{}_train-loss:{:.4f}_val-loss:{:.4f}.pth".format(epoch,
            #                                                                        np.mean(train_loss),
            #                                                                        np.mean(val_loss)))
    except KeyboardInterrupt:
        torch.save(unet.state_dict(), "checkpoint.pth")
