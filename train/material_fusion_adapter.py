import torch

class MaterialFusionAdapter(torch.nn.Module): 
    def __init__(self, spatial_dim: int = 128, mat_dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.film_proj = torch.nn.Linear(mat_dim, spatial_dim)
        self.norm_s  = torch.nn.LayerNorm(spatial_dim)
        self.norm_m  = torch.nn.LayerNorm(mat_dim)
        self.proj_m  = torch.nn.Linear(mat_dim, spatial_dim)
        self.attn    = torch.nn.MultiheadAttention(
            spatial_dim, num_heads=num_heads, batch_first=True
        )
        self.ff = torch.nn.Sequential(
            torch.nn.LayerNorm(spatial_dim),
            torch.nn.Linear(spatial_dim, spatial_dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(spatial_dim * 4, spatial_dim),
        )

        torch.nn.init.xavier_uniform_(self.attn.out_proj.weight, gain=0.5)
        torch.nn.init.zeros_(self.attn.out_proj.bias)

        self.film_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.attn_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, spatial_tokens: torch.Tensor,
                global_vec: torch.Tensor,
                patch_tokens: torch.Tensor) -> torch.Tensor:
        film_bias = self.film_proj(global_vec).unsqueeze(1)
        x = spatial_tokens +  self.film_scale * film_bias
        kv  = self.proj_m(self.norm_m(patch_tokens))
        q   = self.norm_s(x)
        out, _ = self.attn(q, kv, kv)
        x = x + self.attn_scale * out
        x = x + self.ff(x)
        return x