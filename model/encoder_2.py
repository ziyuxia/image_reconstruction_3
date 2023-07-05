import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class encoder1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()


    def forward(self, x):
        # input(3, 256, 256)
        x = self.relu(self.e11(x))
        x = self.relu(self.e12(x))
        x = self.pool1(x)

        # (64, 128, 128)
        x = self.relu(self.e21(x))
        x = self.relu(self.e22(x))
        x = self.pool2(x)

        # (128, 64, 64)
        x = self.relu(self.e31(x))
        x = self.relu(self.e32(x))
        x = self.pool3(x)
        # (256, 32, 32)
        x = self.relu(self.e41(x))
        x = self.relu(self.e42(x))
        x = self.pool4(x)
        # (512, 16, 16)
        x = self.relu(self.e51(x))
        x = self.relu(self.e52(x))
        x = self.pool5(x)
        # (1024, 8, 8)

        return x

# class encoder2(torch.nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.backbone = torchvision.models.vgg13()
#
#     def forward(self, x):
#         x = self.backbone(x)
#         return x

class decoder1(nn.Module):
    def __init__(self):
        super().__init__()

        self.unpool5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.d51 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.d52 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.unpool4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.d41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.unpool3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.d31 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.unpool2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.d21 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.unpool1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.d11 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(64, 3, kernel_size=3, padding=1)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.unpool5(x)
        x = self.relu(self.d51(x))
        x = self.relu(self.d52(x))

        x = self.unpool4(x)
        x = self.relu(self.d41(x))
        x = self.relu(self.d42(x))

        x = self.unpool3(x)
        x = self.relu(self.d31(x))
        x = self.relu(self.d32(x))

        x = self.unpool2(x)
        x = self.relu(self.d21(x))
        x = self.relu(self.d22(x))

        x = self.unpool1(x)
        x = self.relu(self.d11(x))
        x = self.sigmoid(self.d12(x))
        return x


class GatingAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = encoder1()
        self.decoder1 = decoder1()

    def forward(self, x):
        bottleneck = self.encoder1(x)
        # print(bottleneck.size())
        out = self.decoder1(bottleneck)

        return out