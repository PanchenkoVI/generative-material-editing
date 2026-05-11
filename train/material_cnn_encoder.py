import torch

class MaterialCNNEncoder(torch.nn.Module): 
    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.feat_dim = feat_dim
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3,   32, 3, stride=2, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32,  64, 3, stride=2, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(64,  128, 3, stride=2, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(128, feat_dim, 3, stride=2, padding=1), torch.nn.GELU(),
        )
        self.norm_global = torch.nn.LayerNorm(feat_dim)
        self.norm_patch  = torch.nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x) 
        global_vec    = self.norm_global(feat.mean(dim=[2, 3])) 
        patch_tokens = self.norm_patch(feat.flatten(2).permute(0, 2, 1)) 
        return global_vec, patch_tokens 
