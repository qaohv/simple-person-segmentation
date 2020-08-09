from torch import nn, cat
import torch.nn.functional as F
from torchvision import models


class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, e=None):
        d = F.upsample(x, scale_factor=2, mode='bilinear')
        if e is not None:
            d = cat([d, e], 1)
        #print(d.shape)
        return self.double_conv(d)


class UnetResnet34(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet34(pretrained=True)

        self.encoder1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        self.encoder2 = backbone.layer1
        self.encoder3 = backbone.layer2
        self.encoder4 = backbone.layer3
        self.encoder5 = backbone.layer4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = Decoder(256 + 512, 512, 256)
        self.decoder4 = Decoder(256 + 256, 256, 128)
        self.decoder3 = Decoder(128 + 128, 128, 64)
        self.decoder2 = Decoder(64 + 64, 128, 64)
        self.decoder1 = Decoder(64, 64, 64)

        self.out = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        b = self.bottleneck(e5)

        d5 = self.decoder5(b, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2)

        concatenation = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear'),
            F.upsample(d3, scale_factor=4, mode='bilinear'),
            F.upsample(d4, scale_factor=8, mode='bilinear'),
        ], 1)
        print(concatenation.shape)
        return self.out(F.dropout2d(concatenation, p=.5))


if __name__ == "__main__":
    import torch

    m = UnetResnet34()
    t = torch.randn((1, 3, 320, 320))
    out = m(t)
    print(out.shape)
