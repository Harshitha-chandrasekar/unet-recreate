import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list = [64, 128, 256, 512]):
        """
        Args:
            in_channels: Number of channels in the input image (3 for RGB).
            out_channels: Number of target segmentation classes (1 for binary mask).
            features: Feature channel scaling sizes across encoder stages.
        """
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Contracting Path)
        curr_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_channels, feature))
            curr_channels = feature

        # Bottleneck (Bridge between Encoder & Decoder)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (Expansive Path)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 1x1 Convolution to map features to target class count
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Contracting Path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for decoder matching
        skip_connections = skip_connections[::-1]

        # Expansive Path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # ConvTranspose2d upsampling
            skip_connection = skip_connections[idx // 2]

            # Spatial size padding check if dimensions differ slightly due to odd resolutions
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            # Concatenate along channel dimension (Skip Connection)
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)  # DoubleConv

        return self.final_conv(x)


if __name__ == "__main__":
    # Test tensor shape check
    x = torch.randn((2, 3, 128, 128))
    model = UNet(in_channels=3, out_channels=1)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    assert preds.shape == (2, 1, 128, 128), "Shape mismatch in U-Net output!"
    print("U-Net architecture verification passed!")