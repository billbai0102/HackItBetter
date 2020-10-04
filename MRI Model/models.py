import torch
from torch import nn
from torch import functional as F
from torchsummary import summary
from torchvision.models import resnext50_32x4d

from hyperparameters import *


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding):
        super().__init__()

        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convrelu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvRelu(in_channels, in_channels // 4, 1, 0)

        self.deconv = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                         stride=2, padding=1, output_padding=0)

        self.conv2 = ConvRelu(in_channels // 4, out_channels, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)

        return x

class ResNeXtUNet(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        self.base_model = resnext50_32x4d(pretrained=True)
        self.base_layers = list(self.base_model.children())
        filters = [4 * 64, 4 * 128, 4 * 256, 4 * 512]

        # Down
        self.encoder0 = nn.Sequential(*self.base_layers[:3])
        self.encoder1 = nn.Sequential(*self.base_layers[4])
        self.encoder2 = nn.Sequential(*self.base_layers[5])
        self.encoder3 = nn.Sequential(*self.base_layers[6])
        self.encoder4 = nn.Sequential(*self.base_layers[7])

        # Up
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Final Classifier
        self.last_conv0 = ConvRelu(256, 128, 3, 1)
        self.last_conv1 = nn.Conv2d(128, n_classes, 3, padding=1)

    def forward(self, x):
        # Down
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Up + sc
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        # print(d1.shape)

        # final classifier
        out = self.last_conv0(d1)
        out = self.last_conv1(out)
        out = torch.sigmoid(out)

        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            , nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecodeBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, in_channels//4, kernel_size=1, stride=1, padding=0)
            , nn.ConvTranspose2d(in_channels//4, in_channels//4, kernel_size=4, stride=2, padding=1, output_padding=0)
            , ConvBlock(in_channels//4, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, out_size):
        super(UNet, self).__init__()
        self.resnext = resnext50_32x4d(True)
        self.layers = list(self.resnext.children())

        self.down_0 = nn.Sequential(*self.layers[:3])
        self.down_1 = nn.Sequential(*self.layers[4])
        self.down_2 = nn.Sequential(*self.layers[5])
        self.down_3 = nn.Sequential(*self.layers[6])
        self.down_4 = nn.Sequential(*self.layers[7])

        self.up_4 = DecodeBlock(DECODE_IN // 1, DECODE_IN // 2)
        self.up_3 = DecodeBlock(DECODE_IN // 2, DECODE_IN // 4)
        self.up_2 = DecodeBlock(DECODE_IN // 4, DECODE_IN // 8)
        self.up_1 = DecodeBlock(DECODE_IN // 8, DECODE_IN // 8)

        self.classifier_0 = ConvBlock(DECODE_IN // 8, DECODE_IN // 16, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(DECODE_IN // 16, out_size, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        down_0 = self.down_0(x)
        down_1 = self.down_1(down_0)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)

        up_4 = self.up_4(down_4) + down_3
        up_3 = self.up_3(up_4) + down_2
        up_2 = self.up_2(up_3) + down_1
        up_1 = self.up_1(up_2)

        classifier = self.classifier_0(up_1)
        out = self.out(classifier)

        return torch.sigmoid(out)

class Loss:
    def __init__(self, smooth=1.):
        self.smooth = smooth

    def F1_metric(self, pred, target):
        """
        Calculates F1 score of seg mask
        (2 * overlap) / (total pixels)
        :param pred: prediction mask
        :param target: target mask
        :return:
        """
        overlap = 2. * (pred * target).sum()
        union = pred.sum() + target.sum()

        # return 1 for no tumor detected
        if pred.sum() == 0 and target.sum() == 0:
            return 1.

        return overlap / union

    def F1_score(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Calculates F1 score w/ smoothing of segmentation mask
        1- [(2 * overlap + 1) / (total pixels + 1)]
        :param pred: prediction mask
        :param target: target mask
        :return: F1 score
        """
        overlap = 2. * ((pred * target).sum()) + self.smooth
        union = pred.sum() + target.sum() + self.smooth
        return 1 - (overlap / union)

    def BCEDiceLoss(self, pred: torch.Tensor, target: torch.Tensor):
        """

        :param pred:
        :param target:
        :return:
        """
        f1_score = self.F1_score(pred, target)
        bce = nn.BCELoss()
        loss = bce(pred, target)

        return loss + f1_score


if __name__ == '__main__':
    net = UNet(2).cuda()
    print(net)
    print(summary(net, input_size=(3, 64, 64)))
