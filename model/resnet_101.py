import torch
import torchvision
from torch import nn

class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet101 = torchvision.models.resnet101(pretrained=True)
        for param in resnet101.parameters():
            param.requires_grad = True
        modules = list(resnet101.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.resnet(x)
        out = self.relu(out)
        return out


class Compressor1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        return out

class Compressor2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        return out



# if __name__ == '__main__':
#     x = torch.randn(16, 3, 256, 256)
#
#     e = ResNetEncoder()
#     d = ResNetDecoder()
#
#     y = e(x)
#     z = d(y)
#     print(y.shape)





class ResNetDecoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.ReLU()
        )
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU()
        )
        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU()
        )
        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, stride=1,
                               padding=1),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU()
        )
        self.convTrans5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2,
                               padding=3, output_padding=1),
            nn.Sigmoid()
        )



    def forward(self, x):
        out = self.convTrans1(x)
        out = self.convTrans2(out)
        out = self.convTrans3(out)
        out = self.convTrans4(out)
        out = self.convTrans5(out)
        return out

class ResNetDecoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.ReLU()
        )
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU()
        )
        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU()
        )
        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, stride=1,
                               padding=1),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU()
        )
        self.convTrans5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2,
                               padding=3, output_padding=1),
            nn.Sigmoid()
        )



    def forward(self, x):
        out = self.convTrans1(x)
        out = self.convTrans2(out)
        out = self.convTrans3(out)
        out = self.convTrans4(out)
        out = self.convTrans5(out)
        return out

class ResNetDecoder3(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU()
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.ReLU()
        )
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU()
        )
        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU()
        )
        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, stride=1,
                               padding=1),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU()
        )
        self.convTrans5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2,
                               padding=3, output_padding=1),
            nn.Sigmoid()
        )



    def forward(self, x):
        out = self.convTrans1(x)
        out = self.convTrans2(out)
        out = self.convTrans3(out)
        out = self.convTrans4(out)
        out = self.convTrans5(out)
        return out



class ExitGate1(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.pool = nn.AvgPool2d((8, 8))
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, 128, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(64, 1, bias=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.relu(self.bn0(self.pool(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        out = self.linear(x)
        out = self.sigmoid(out)

        return out

class GatingAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = ResNetEncoder()
        self.compressor1 = Compressor1()
        self.compressor2 = Compressor2()
        self.decoder1 = ResNetDecoder1()
        self.decoder2 = ResNetDecoder2()
        self.decoder3 = ResNetDecoder3()
        self.exitgate1 = ExitGate1(2048)


    def forward(self, x):
        bottleneck1 = self.encoder1(x)
        bottleneck2 = self.compressor1(bottleneck1)
        # bottleneck3 = self.compressor2(bottleneck1)
        out1 = self.decoder1(bottleneck1)
        out2 = self.decoder2(bottleneck2)
        # out3 = self.decoder3(bottleneck3)
        # exit1 = self.exitgate1(bottleneck1)
        return out1, out2