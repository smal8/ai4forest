'''
Purpose of this python file:
- Define the Unet 3D Architecture
- one layer is removed so that an image with 11 time dimensions can be processed
'''

import torch
from torch import nn
import torch.nn.functional as F
import pdb
import time

# Define a class for a double convolutional block
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, reduce=True):
        super(DoubleConv, self).__init__()
        # Define convolutional layers with batch normalization and ReLU activation
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(3,3,3), padding=1),
            nn.GroupNorm(32,out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=(3,3,3), padding=(0,1,1)) if reduce else nn.Conv3d(out_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.GroupNorm(32,out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.conv(x)
        return x

# Define a class for the upsampling operation
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        # Define the upsampling operation using transposed convolution
        self.up_scale = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x1, x2):
        # Upsample the input tensor
        x2 = self.up_scale(x2)

        # Calculate the difference in dimensions between x1 and x2
        diffZ = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        diffX = x1.size()[4] - x2.size()[4]

        # Pad x2 to match the dimensions of x1
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        # Concatenate x1 and x2 along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return x


# Define a class for the downsampling operation
class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        # Define a max pooling layer and a double convolutional block
        self.pool = nn.MaxPool3d((1,2,2), stride=(1,2,2), padding=0)
        self.conv = DoubleConv(in_ch, out_ch, reduce=True)

    def forward(self, x):
        # Forward pass through the max pooling and convolutional layers
        x = self.conv(self.pool(x))
        return x

# Define a class for the upsampled layer
class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpLayer, self).__init__()
        # Define the upsampling and double convolutional blocks
        self.up = Up(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch, reduce=False)

    def forward(self, x1, x2):
        # Perform the upsampling operation
        a = self.up(x1, x2)
        # Forward pass through the convolutional layers
        x = self.conv(a)
        return x

class ReduceTimeLayer(nn.Module):
    def __init__(self):
        super(ReduceTimeLayer, self).__init__()
        self.conv1 = nn.Conv3d(12, 24, kernel_size=(5,3,3), padding=(0,1,1))
        self.conv2 = nn.Conv3d(24, 36, kernel_size=(5,3,3), padding=(0,1,1))
        self.conv3 = nn.Conv3d(36, 36, kernel_size=(4,3,3), padding=(0,1,1))
        self.GroupNorm24 = nn.GroupNorm(24, 24)
        self.GroupNorm36 = nn.GroupNorm(36, 36)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.GroupNorm24(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.GroupNorm36(x)
        x = self.ReLU(x)
        x = self.conv3(x)
        x = self.GroupNorm36(x)
        x = self.ReLU(x)
        return x


# Define the U-Net architecture
class UNet(nn.Module):
    def __init__(self, dimensions=7):
        super(UNet, self).__init__()
        # Define the contracting (downsampling) path of the U-Net
        self.conv1 = DoubleConv(36, 64) # statt 1 haben wir hier 12 Input Channels
        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        self.down3 = DownLayer(256, 512)
        self.down4 = DownLayer(512, 1024)

        # Define the expanding (upsampling) path of the U-Net
        self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(512, 256)
        self.up3 = UpLayer(256, 128)
        self.up4 = UpLayer(128, 64)

        # Define the final convolutional layer
        # in unserem Fall ist dimensions = 1, weil wir pro Pixel eine Baumhöhe ausgeben
        self.last_conv = nn.Conv3d(64, dimensions, 1)
        self.x1_conv = nn.Conv3d(64,64, kernel_size=(10,1,1), padding=(0,0,0))
        self.x2_conv = nn.Conv3d(128,128, kernel_size=(8,1,1), padding=(0,0,0))
        self.x3_conv = nn.Conv3d(256,256, kernel_size=(6,1,1), padding=(0,0,0))
        self.x4_conv = nn.Conv3d(512,512, kernel_size=(4,1,1), padding=(0,0,0))
        self.x5_conv = nn.Conv3d(1024,1024, kernel_size=(2,1,1), padding=(0,0,0))

        self.ReduceTimeLayer = ReduceTimeLayer()

    def forward(self, x):
        print(0)
        time.sleep(10)
        print(1)
        x = x.permute(0,2,1,3,4).reshape(x.shape[0]*x.shape[2]//12,12,12,512,512).permute(0,2,1,3,4)
        print(2)
        time.sleep(10)
        x = self.ReduceTimeLayer(x)
        time.sleep(3)
        print(3)
        x = x.permute(0,2,1,3,4).reshape(x.shape[0]//7,7,36,512,512).permute(0,2,1,3,4)
        time.sleep(3)
        print(4)

        # Forward pass through the U-Net architecture
        x1 = self.conv1(x)
        x1_agg = self.x1_conv(x1)
        time.sleep(3)
        print(5)

        x2 = self.down1(x1)
        x2_agg = self.x2_conv(x2)
        time.sleep(3)
        print(6)

        x3 = self.down2(x2)
        x3_agg = self.x3_conv(x3)
        time.sleep(3)
        print(7)

        x4 = self.down3(x3)
        x4_agg = self.x4_conv(x4)
        time.sleep(3)
        print(8)

        x5 = self.down4(x4)
        x5_agg = self.x5_conv(x5)
        time.sleep(3)
        print(9)

        x1_up = self.up1(x4_agg, x5_agg)

        x2_up = self.up2(x3_agg, x1_up)

        x3_up = self.up3(x2_agg, x2_up)

        x4_up = self.up4(x1_agg, x3_up)


        output = self.last_conv(x4_up)

        # Eine Dimension muss weg, damit das Format mit den Labeldaten übereinstimmt
        output_tensor = torch.squeeze(output, dim=1)
        return output_tensor
