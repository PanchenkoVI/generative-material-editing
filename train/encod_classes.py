import torch 

class ResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm1 = torch.nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm2 = torch.nn.GroupNorm(min(8, out_ch), out_ch)
        self.act   = torch.nn.SiLU()
        self.skip  = torch.nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else torch.nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


def build_spatial_cond_encoder(device, dtype): 
    return torch.nn.Sequential(
        torch.nn.Conv2d(2, 32, 3, 1, 1),
        ResBlock(32, 32),
        torch.nn.Conv2d(32, 64, 4, 2, 1),
        ResBlock(64, 64),
        torch.nn.Conv2d(64, 32, 4, 2, 1),
        ResBlock(32, 32),
        torch.nn.GroupNorm(8, 32),
    ).to(device).to(dtype)


class CondProj(torch.nn.Module):
    def __init__(self, in_dim, bottleneck=256, out_dim=384, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, bottleneck),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(bottleneck, out_dim),
        )

    def forward(self, x):
        return self.net(x)