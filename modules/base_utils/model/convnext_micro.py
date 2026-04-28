import torch
import torch.nn as nn
import torch.nn.functional as F

class CNBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)
        return x + shortcut


class ConvNeXtMicro(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        depths = [1, 1, 1, 1]
        dims   = [32, 64, 96, 128]
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample = nn.Sequential(
                nn.BatchNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[CNBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])
        x = self.norm(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = ConvNeXtMicro(num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.3f}M")