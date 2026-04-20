import torch
import torch.nn as nn
import torch.nn.functional as F

# src: Prof. Chen unet_example.py
class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2

    This is the basic unit used in both the encoder and decoder.
    Padding=1 keeps H and W the same after each 3x3 convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling with maxpool then DoubleConv.
    Reduces H and W by 2 using MaxPool2d, then expands feature channels with DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """Upscaling then DoubleConv.

    Two upsampling options:
    - bilinear: use F.interpolate (fast, memory-light). We follow with a 1x1 conv to
      reduce channels of the skip-connection input if needed.
    - transposed conv: learnable upsampling that can also learn to reduce checkerboards.

    After upsampling, we concatenate the skip feature from the encoder (same spatial size)
    along the channel dimension, then apply DoubleConv to fuse them.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            # When concatenating, channels become (in_channels from up) + (skip channels).
            # We don't know skip channels here, so we'll reduce upsampled channels by half
            # so that after concat the DoubleConv sees roughly a manageable number.
            # Conventionally in U-Net: in_channels is 2 * out_channels at each up step.
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            in_channels = in_channels // 2  # after reduce
        else:
            # Transposed convolution doubles spatial resolution and reduces channels.
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.reduce = nn.Identity()
            in_channels = in_channels // 2

        # DoubleConv will receive concatenated [up(x), skip] -> channels = in_channels + skip_channels
        self.conv = DoubleConv(in_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)            # upsample decoder feature
        x = self.reduce(x)        # ensure channel alignment with design

        # In rare cases, due to odd sizes, upsampled tensor may be off by 1px.
        # We pad/crop to match spatial size of the skip tensor exactly.
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)

        # Concatenate along channel dimension: [N, C_up, H, W] + [N, C_skip, H, W] -> [N, C_up+C_skip, H, W]
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """Final 1x1 convolution to map features -> class logits per pixel."""
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetPP(nn.Module):
    """
    U-Net++ (Nested U-Net)
    Builds on the UNet architecture. It adds convolution layers 
    that produce skip connections between the encoder and decoder
    """

    def __init__(self, in_channels=3, num_classes=1, base_channels=64, bilinear=True):
        super().__init__()
        nb_filter = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]

        # encoder: these will be used for downsampling
        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.down1 = Down(nb_filter[0], nb_filter[1])
        self.down2 = Down(nb_filter[1], nb_filter[2])
        self.down3 = Down(nb_filter[2], nb_filter[3])
        self.down4 = Down(nb_filter[3], nb_filter[4])

        # first skip layer found in Unet
        self.up0_1 = Up(nb_filter[1], nb_filter[0], bilinear)
        self.up1_1 = Up(nb_filter[2], nb_filter[1], bilinear)
        self.up2_1 = Up(nb_filter[3], nb_filter[2], bilinear)
        self.up3_1 = Up(nb_filter[4], nb_filter[3], bilinear)

        # the additional skip connections added to Unet; this is what distinguishes
        # the Unet++ architecture from Unet
        self.conv0_2 = DoubleConv(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        # decoder: this'll be used for the final output
        self.outc = OutConv(nb_filter[0], num_classes)

    def forward(self, x):
        # downsampiing (encoder)
        x0_0 = self.conv0_0(x)
        x1_0 = self.down1(x0_0)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)
        x4_0 = self.down4(x3_0)

        # first skip layer found in Unet
        x0_1 = self.up0_1(x1_0, x0_0)
        x1_1 = self.up1_1(x2_0, x1_0)
        x2_1 = self.up2_1(x3_0, x2_0)
        x3_1 = self.up3_1(x4_0, x3_0)

        # the additional skip connections added to Unet; this is what distinguishes
        # the Unet++ architecture from Unet
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, size=x0_0.shape[-2:], \
        mode='bilinear', align_corners=True)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, size=x1_0.shape[-2:], \
        mode='bilinear', align_corners=True)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, size=x2_0.shape[-2:], \
        mode='bilinear', align_corners=True)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, size=x0_0.shape[-2:], \
        mode='bilinear', align_corners=True)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, size=x1_0.shape[-2:], \
        mode='bilinear', align_corners=True)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, size=x0_0.shape[-2:], \
        mode='bilinear', align_corners=True)], dim=1))

        # upsampling (decoder)
        logits = self.outc(x0_4)
        return logits
