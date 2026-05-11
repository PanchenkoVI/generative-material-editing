from diffusers.pipelines import FluxFillPipeline
import lightning as L
from peft import LoraConfig, get_peft_model_state_dict
import prodigyopt
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import os
from pathlib import Path
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from .transformer import tranformer_forward 
from .pipeline_tools import encode_images, prepare_text_input, build_spatial_cond_encoder
from .encod_classes import CondProj
from .material_fusion_adapter import MaterialFusionAdapter
from .material_cnn_encoder import MaterialCNNEncoder

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False 

def load_xlabs_image_encoder(device, dtype):
    repo = "openai/clip-vit-large-patch14"
    encoder = CLIPVisionModelWithProjection.from_pretrained(repo, torch_dtype=dtype).to(device)
    processor = CLIPImageProcessor.from_pretrained(repo)
    encoder.requires_grad_(False).eval()
    print(f"CLIP image_encoder загружен из {repo}")
    return encoder, processor


def load_xlabs_ip_proj(xlabs_ckpt_path, device, dtype):
    proj = torch.nn.Linear(768, 16 * 4096, bias=True).to(device=device, dtype=dtype)
    norm = torch.nn.LayerNorm(4096).to(device=device, dtype=dtype)

    if xlabs_ckpt_path and Path(xlabs_ckpt_path).exists():
        ckpt = load_file(xlabs_ckpt_path)
        for attr, key in [("weight", "ip_adapter_proj_model.proj.weight"),
                          ("bias", "ip_adapter_proj_model.proj.bias")]:
            if key in ckpt:
                getattr(proj, attr).data.copy_(ckpt[key].to(device=device, dtype=dtype))
        for attr, key in [("weight", "ip_adapter_proj_model.norm.weight"),
                          ("bias", "ip_adapter_proj_model.norm.bias")]:
            if key in ckpt and ckpt[key].shape == (4096,):
                getattr(norm, attr).data.copy_(ckpt[key].to(device=device, dtype=dtype))
        print(f"ip_proj загружен из {xlabs_ckpt_path}")
    else:
        print("[WARNING] xlabs_ckpt_path не задан — ip_proj случайные веса")

    proj.requires_grad_(False)
    norm.requires_grad_(False)
    return proj, norm


def get_ip_tokens(ip_proj, ip_norm, clip_pooled):
    B = clip_pooled.shape[0]
    out = ip_proj(clip_pooled).view(B, 16, 4096)
    return ip_norm(out)


class FluxIPAttnProcessor(torch.nn.Module):
    def __init__(self, dim: int, ip_dim: int = 4096):
        super().__init__()
        self.ip_to_k = torch.nn.Linear(ip_dim, dim, bias=True)
        self.ip_to_v = torch.nn.Linear(ip_dim, dim, bias=True)
        self.register_buffer("ip_scale", torch.tensor(1.0))
        self._ip_tokens = None
        torch.nn.init.normal_(self.ip_to_k.weight, std=0.02)
        torch.nn.init.zeros_(self.ip_to_k.bias)
        torch.nn.init.normal_(self.ip_to_v.weight, std=0.02)
        torch.nn.init.zeros_(self.ip_to_v.bias)

    def forward(self, attn, hidden_states, encoder_hidden_states=None,
                attention_mask=None, image_rotary_emb=None, ip_tokens=None, **kwargs):
        current_ip = ip_tokens if ip_tokens is not None else self._ip_tokens
        B, seq, _ = hidden_states.shape
        inner_dim = attn.to_q.out_features
        head_dim = inner_dim // attn.heads

        q = attn.to_q(hidden_states).view(B, -1, attn.heads, head_dim).transpose(1, 2)
        k = attn.to_k(hidden_states).view(B, -1, attn.heads, head_dim).transpose(1, 2)
        v = attn.to_v(hidden_states).view(B, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: k = attn.norm_k(k)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, -1, inner_dim).to(q.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if current_ip is not None:
            scale = torch.tanh(self.ip_scale) * 2.0
            ip_k = self.ip_to_k(current_ip).view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ip_v = self.ip_to_v(current_ip).view(B, -1, attn.heads, head_dim).transpose(1, 2)
            ip_out = F.scaled_dot_product_attention(q, ip_k, ip_v, dropout_p=0.0)
            ip_out = ip_out.transpose(1, 2).reshape(B, -1, inner_dim).to(q.dtype)
            out = out + scale * ip_out

        return out


def install_ip_processors(transformer, ip_dim=4096, device=None, dtype=None):
    processors = torch.nn.ModuleList()
    for _, module in transformer.named_modules():
        if isinstance(module, Attention):
            proc = FluxIPAttnProcessor(dim=module.to_q.out_features, ip_dim=ip_dim)
            if device: proc = proc.to(device=device, dtype=dtype)
            proc.requires_grad_(False)
            module.processor = proc
            processors.append(proc)
    print(f"IP процессоры установлены: {len(processors)}")
    return processors


def load_xlabs_ip_weights(transformer, xlabs_ckpt_path, device, dtype):
    if not xlabs_ckpt_path or not Path(xlabs_ckpt_path).exists():
        print("[WARNING] xlabs_ckpt_path не задан")
        return
    ckpt = load_file(xlabs_ckpt_path)
    weight_map = {}
    for key, val in ckpt.items():
        if not key.startswith("double_blocks."): continue
        parts = key.split(".")
        try:
            n = int(parts[1])
        except (ValueError, IndexError):
            continue

        wm = weight_map.setdefault(n, {})

        if "k_proj.weight" in key: 
            wm["k_w"] = val
        elif "k_proj.bias" in key: 
            wm["k_b"] = val
        elif "v_proj.weight" in key: 
            wm["v_w"] = val
        elif "v_proj.bias" in key: 
            wm["v_b"] = val

    tb_attn = {}
    for name, module in transformer.named_modules():
        if isinstance(module, Attention) and isinstance(module.processor, FluxIPAttnProcessor):
            if name.startswith("transformer_blocks."):
                try: tb_attn[int(name.split(".")[1])] = module
                except: pass

    loaded = 0
    for n, wm in weight_map.items():
        if n not in tb_attn: continue
        proc = tb_attn[n].processor
        ok = True
        for linear, wk, bk in [(proc.ip_to_k, "k_w", "k_b"), (proc.ip_to_v, "v_w", "v_b")]:
            if wk in wm:
                w = wm[wk].to(device=device, dtype=dtype)
                if linear.weight.shape == w.shape:
                    linear.weight.data.copy_(w)
                else:
                    ok = False
            if bk in wm and linear.bias is not None:
                linear.bias.data.copy_(wm[bk].to(device=device, dtype=dtype))
        if ok: loaded += 1
    print(f"XLabs IP weights загружены: {loaded}/{len(weight_map)} блоков")


def set_ip_tokens(transformer, ip_tokens):
    for module in transformer.modules():
        if isinstance(module, Attention) and isinstance(module.processor, FluxIPAttnProcessor):
            module.processor._ip_tokens = ip_tokens


def to_display(t):
    t = t.float()
    if t.min() < -0.01: 
        return ((t + 1.0) / 2.0).clamp(0, 1)
    if t.max() > 1.01:
        return (t / 255.0).clamp(0, 1)
    return t.clamp(0, 1)


class OminiModelFIll(L.LightningModule): 
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        reuse_lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        val_batch: dict = None,
        val_every_n_steps: int = 125,
        train_every_n_steps: int = 50,
        val_num_steps: int = 40,
        val_guidance: float = 30.0,
        ip_dim: int = 4096,
        ip_num_tokens: int = 16,
        xlabs_ckpt_path: str = None,
        cond_scale: float = 1.0,
        mask_loss_weight: float = 2.0,
        lpips_weight: float = 1.0,
        mat_feat_dim: int = 512,
    ):
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.val_batch = val_batch
        self.val_every_n_steps = val_every_n_steps
        self.val_num_steps = val_num_steps
        self.val_guidance = val_guidance
        self.cond_scale = cond_scale
        self.mask_loss_weight = mask_loss_weight
        self.lpips_weight = lpips_weight
        self.ip_num_tokens = ip_num_tokens
        self.ip_dim = ip_dim
        self.model_dtype = dtype
        self.vae_scale_factor = 8
        self.mat_feat_dim = mat_feat_dim
        self._val_iteration = 0
        self._last_val_step = -1
 
        self.flux_pipe = FluxFillPipeline.from_pretrained(
            flux_pipe_id, torch_dtype=dtype,
            device_map="balanced", low_cpu_mem_usage=True, offload_state_dict=True,
        )
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        enc_device = next(self.transformer.parameters()).device
        self.lora_layers = self.init_lora(lora_path, lora_config)

        # Spatial encoder: edges + mask → spatial tokens
        self.spatial_cond_encoder = build_spatial_cond_encoder(enc_device, dtype)

        # Material CNN encoder: basecolor → глобальный вектор материала
        self.material_cnn_encoder = MaterialCNNEncoder(feat_dim=mat_feat_dim).to(enc_device).to(dtype)

        # Fusion adapter: cross-attention spatial ← material 
        self.material_fusion_adapter = MaterialFusionAdapter(
            spatial_dim=128, mat_dim=mat_feat_dim, num_heads=8
        ).to(enc_device).to(dtype)

        self.cond_proj = None

        # XLabs CLIP (frozen)
        self.ip_image_encoder, self.ip_image_processor = load_xlabs_image_encoder(enc_device, dtype)
        self.ip_proj_in, self.ip_norm = load_xlabs_ip_proj(xlabs_ckpt_path, enc_device, dtype)

        # IP attention processors (frozen)
        self.ip_processors = install_ip_processors(
            self.transformer, ip_dim=self.ip_dim, device=enc_device, dtype=dtype
        )
        load_xlabs_ip_weights(self.transformer, xlabs_ckpt_path, enc_device, dtype)

        self.lpips_fn = (
            LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).eval()
            if HAS_LPIPS else None
        )
        if self.lpips_fn:
            self.lpips_fn.requires_grad_(False)

        if reuse_lora_path:
            self._load_checkpoint(reuse_lora_path)

    def init_lora(self, lora_path, lora_config):
        assert lora_path or lora_config
        if lora_path: raise NotImplementedError
        self.transformer.add_adapter(LoraConfig(**lora_config))
        return list(filter(lambda p: p.requires_grad, self.transformer.parameters()))

    def configure_optimizers(self):
        self.transformer.requires_grad_(False) 
        self.trainable_params = (
            list(self.lora_layers) +
            list(self.spatial_cond_encoder.parameters()) +
            list(self.material_cnn_encoder.parameters()) +
            list(self.material_fusion_adapter.parameters())
        )
        for p in self.trainable_params:
            p.requires_grad_(True)

        opt_config = self.optimizer_config
        if opt_config["type"] == "AdamW":
            return torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            return prodigyopt.Prodigy(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "SGD":
            return torch.optim.SGD(self.trainable_params, **opt_config["params"])
        raise NotImplementedError

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)

    def _load_checkpoint(self, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        lora_p = ckpt_dir / "pytorch_lora_weights.safetensors"
        if lora_p.exists():
            sd = load_file(str(lora_p))
            sd = {k.replace("lora_A", "lora_A.default").replace("lora_B", "lora_B.default")
                   .replace("transformer.", ""): v for k, v in sd.items()}
            self.transformer.load_state_dict(sd, strict=False)
            print(f"LoRA загружена: {lora_p}")
        for attr, fname in [
            ("spatial_cond_encoder", "spatial_cond_encoder.safetensors"),
            ("material_cnn_encoder","material_cnn_encoder.safetensors"),
            ("material_fusion_adapter", "material_fusion_adapter.safetensors"),
            ("cond_proj", "cond_proj.safetensors"),
        ]:
            p = ckpt_dir / fname
            obj = getattr(self, attr, None)
            if p.exists() and obj is not None:
                obj.load_state_dict(load_file(str(p)), strict=False)
                print(f"{attr} загружен: {p}") 

        cond_path = ckpt_dir / "cond_proj.safetensors"
        if cond_path.exists(): 
            if self.cond_proj is not None:
                self.cond_proj.load_state_dict(load_file(str(cond_path)), strict=False)
                print(f"cond_proj weights loaded from {cond_path}")
            else: 
                self._pending_cond_proj = load_file(str(cond_path))
                print(f"Pending cond_proj weights for later init")

    def save_checkpoint(self, path):
        os.makedirs(path, exist_ok=True)
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        for attr, fname in [
            ("spatial_cond_encoder", "spatial_cond_encoder.safetensors"),
            ("material_cnn_encoder", "material_cnn_encoder.safetensors"),
            ("material_fusion_adapter", "material_fusion_adapter.safetensors"),
            ("cond_proj", "cond_proj.safetensors"),
        ]:
            obj = getattr(self, attr, None)
            if obj is not None:
                save_file({k: v.contiguous() for k, v in obj.state_dict().items()},
                          os.path.join(path, fname))
        print(f"Чекпоинт сохранён: {path}")

    def save_lora(self, path): self.save_checkpoint(path)

    def on_train_epoch_end(self):
        sp = getattr(self.trainer, "default_root_dir", "./output")
        rn = getattr(self.trainer, "run_name", "run")
        self.save_checkpoint(f"{sp}/{rn}/epoch_{self.current_epoch:04d}")

    def _save_step_checkpoint(self, step):
        sp = getattr(self.trainer, "default_root_dir", "./output")
        rn = getattr(self.trainer, "run_name", "run")
        self.save_checkpoint(f"{sp}/{rn}/step_{step:06d}")

    def _vae_encode_rgb(self, images_rgb):
        vd, vt = (next(self.flux_pipe.vae.parameters()).device,
                  next(self.flux_pipe.vae.parameters()).dtype)
        lat = self.flux_pipe.vae.encode(images_rgb.to(device=vd, dtype=vt)).latent_dist.sample()
        lat = (lat - self.flux_pipe.vae.config.shift_factor) * self.flux_pipe.vae.config.scaling_factor
        tok = self.flux_pipe._pack_latents(lat, *lat.shape)
        cond_lat = F.interpolate(lat, size=(64, 64), mode="nearest")
        cond_tok = self.flux_pipe._pack_latents(cond_lat, *cond_lat.shape)
        return tok, cond_tok, lat

    def _pack_mask_full(self, mask, B, H, W):
        sf = self.vae_scale_factor
        m = mask.view(B, H // sf, sf, W // sf, sf).permute(0, 2, 4, 1, 3).reshape(B, sf*sf, H//sf, W//sf)
        return self.flux_pipe._pack_latents(m, B, sf*sf, H//sf, W//sf)

    def _pack_mask_cond(self, mask, B):
        sf = self.vae_scale_factor
        cm = F.interpolate(mask, size=(64, 64), mode="nearest").repeat(1, sf*sf, 1, 1)
        return self.flux_pipe._pack_latents(cm, B, sf*sf, 64, 64)

    @torch.no_grad()
    def _encode_material_clip(self, material_rgb):
        mat = ((material_rgb.float() + 1.0) / 2.0).clamp(0, 1)
        mat = F.interpolate(mat, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=mat.device, dtype=mat.dtype).view(1,3,1,1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=mat.device, dtype=mat.dtype).view(1,3,1,1)
        mat = (mat - mean) / std
        ed = next(self.ip_image_encoder.parameters()).device
        return self.ip_image_encoder(pixel_values=mat.to(device=ed, dtype=self.model_dtype)).image_embeds

    def _get_ip_tokens(self, material_rgb):
        return get_ip_tokens(self.ip_proj_in, self.ip_norm, self._encode_material_clip(material_rgb))

    def _build_conditioning_patch(self, materials, edges, mask_proc, B, H, W, device):
        edges_full = F.interpolate(edges, size=(H, W), mode="bilinear", align_corners=False)
        cond_input = torch.cat([edges_full, mask_proc], dim=1)
        cond_feat = self.spatial_cond_encoder(cond_input)
        cond_feat = F.interpolate(cond_feat, size=(64, 64), mode="bilinear")
        cond_latents = self.flux_pipe._pack_latents(cond_feat, B, cond_feat.shape[1], 64, 64)
        mat_input = F.interpolate(materials[:, :3], size=(256, 256), mode="bilinear", align_corners=False)
        global_vec, patch_tokens = self.material_cnn_encoder(mat_input)
        cond_latents = self.material_fusion_adapter(cond_latents, global_vec, patch_tokens)
        return cond_latents * self.cond_scale

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        if torch.isnan(loss):
            print(f"NaN at step {self.global_step}")
            return None

        self.log_loss = (loss.item() if not hasattr(self, "log_loss") else self.log_loss * 0.95 + loss.item() * 0.05)
        self.log("train/loss", self.log_loss, prog_bar=False, on_step=True)
        self.log("train/loss_sd", self.res["loss_sd"].detach(), prog_bar=False, on_step=True)
        self.log("train/loss_mask", self.res["loss_mask"].detach(),prog_bar=False, on_step=True)
        self.log("train/loss_lpips", self.res["loss_lpips"].detach(), prog_bar=False, on_step=True)
        self.log("train/t", self.last_t, prog_bar=False, on_step=True)

        si = getattr(self.trainer, "save_interval", None)
        if si and (self.global_step + 1) % si == 0:
            self._save_step_checkpoint(self.global_step)

        if ((self.global_step + 1) % self.val_every_n_steps == 0 and self._last_val_step != self.global_step):
            self._last_val_step = self.global_step
            self._run_val_visualization()
            print("\ntrain/loss", self.log_loss, "train/loss_sd", self.res["loss_sd"].detach(), "train/loss_mask",  self.res["loss_mask"].detach(), "train/loss_lpips", self.res["loss_lpips"].detach())
            if hasattr(self, "_last_val_lpips"):
                self.log("val/lpips", self._last_val_lpips, prog_bar=False, on_step=True)
        return loss

    def step(self, batch):
        device = next(self.transformer.parameters()).device
        imgs = batch["image"]
        ref_imgs = batch["ref_image"]
        materials = batch["condition"].to(device, dtype=self.model_dtype)
        edges = batch["edges"].to(device, dtype=self.model_dtype)
        hints = batch["hint"]
        prompts = batch["description"]

        mask_image = hints[:, 0]
        B, _, H, W = imgs.shape

        mask_proc = self.flux_pipe.mask_processor.preprocess(mask_image, height=H, width=W) 
        mask_proc = (mask_proc > 0.5).to(device=device, dtype=self.model_dtype) 

        ip_tokens = self._get_ip_tokens(materials[:, :3])

        # Conditioning (обучаемый путь с материалом) 
        cond_latents = self._build_conditioning_patch(materials, edges, mask_proc, B, H, W, device)
        condition_ids = self.flux_pipe._prepare_latent_image_ids(B, 32, 32, device, self.model_dtype)

        with torch.no_grad():
            x_0, img_ids = encode_images(self.flux_pipe, imgs)
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(self.flux_pipe, prompts)
            t = torch.rand((B,), device=device)
            x_1 = torch.randn_like(x_0)
            x_t = ((1 - t.view(B,1,1)) * x_0 + t.view(B,1,1) * x_1).to(self.model_dtype)

        # Masked inpainting input
        ref_proc = self.flux_pipe.image_processor.preprocess(ref_imgs, height=H, width=W).to(device=device, dtype=self.model_dtype)
        imgs_proc = self.flux_pipe.image_processor.preprocess(imgs, height=H, width=W).to(device=device, dtype=self.model_dtype) 
        masked_rgb = ref_proc * (1 - mask_proc)

        masked_latents, cond_masked_latents, _ = self._vae_encode_rgb(masked_rgb)
        masked_latents = torch.cat([masked_latents, self._pack_mask_full(mask_proc, B, H, W)], dim=-1)
        cond_masked_latents = torch.cat([cond_masked_latents, self._pack_mask_cond(mask_proc, B)], dim=-1)

        # Combined condition
        combined_cond = torch.cat((cond_latents, cond_masked_latents), dim=2)
        if self.cond_proj is None:
            in_dim = combined_cond.shape[-1]
            self.cond_proj = CondProj(in_dim=in_dim, bottleneck=384, out_dim=384, dropout=0.1)
            self.cond_proj = self.cond_proj.to(device=device, dtype=self.model_dtype)
            opt = self.optimizers()
            if opt is not None:
                opt.add_param_group({"params": list(self.cond_proj.parameters())})
                self.trainable_params += list(self.cond_proj.parameters())
            print(f"cond_proj инициализирован: in_dim={in_dim}")

        if hasattr(self, '_pending_cond_proj') and self._pending_cond_proj is not None:
            self.cond_proj.load_state_dict(self._pending_cond_proj, strict=False)
            delattr(self, '_pending_cond_proj')
            print("cond_proj weights restored from pending")

        combined_cond = self.cond_proj(combined_cond)
        condition_type_ids = torch.full((B, 1), 11, device=device, dtype=torch.long)
        guidance = torch.ones_like(t) if self.transformer.config.guidance_embeds else None

        set_ip_tokens(self.transformer, ip_tokens)
        pred = tranformer_forward(
            self.transformer,
            model_config=self.model_config,
            condition_latents=combined_cond,
            condition_ids=condition_ids,
            condition_type_ids=condition_type_ids,
            hidden_states=torch.cat((x_t, masked_latents), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        set_ip_tokens(self.transformer, None)

        target = x_1 - x_0
        loss_sd = F.mse_loss(pred, target)

        sf = self.vae_scale_factor
        ph, pw = H // sf // 2, W // sf // 2
        mask_lat = F.interpolate(mask_proc, size=(ph, pw), mode="nearest")
        mask_lat = mask_lat.view(B, 1, ph * pw).permute(0, 2, 1)
        loss_mask = ((pred - target) ** 2 * mask_lat).mean()
        loss = loss_sd + self.mask_loss_weight * loss_mask

        lpips_loss = torch.tensor(0.0, device=device)
        loss_odm = torch.tensor(0.0, device=device)

        if self.lpips_fn is not None and self.lpips_weight > 0:
            try:
                x0h = x_1 - pred
                ld = self.flux_pipe._unpack_latents(x0h, H, W, sf)
                ld = (ld / self.flux_pipe.vae.config.scaling_factor) + self.flux_pipe.vae.config.shift_factor
                vd, vt = (next(self.flux_pipe.vae.parameters()).device, next(self.flux_pipe.vae.parameters()).dtype)
                img_p = self.flux_pipe.vae.decode(ld.to(device=vd, dtype=vt))[0]
                p01 = (img_p / 2 + 0.5).clamp(0, 1)
                t01 = (imgs_proc.to(img_p.device) / 2 + 0.5).clamp(0, 1)
                lpips_loss = self.lpips_fn(p01, t01).mean()
                loss = loss + self.lpips_weight * lpips_loss
            except RuntimeError:
                pass

        self.last_t = t.mean().item()
        self.res = {
            "loss": loss.detach(), "loss_sd": loss_sd.detach(),
            "loss_mask": loss_mask.detach(), "loss_lpips": lpips_loss.detach(),
            "loss_odm": loss_odm.detach(),
        }
        return loss

    @torch.no_grad()
    def _run_val_visualization(self):
        self._val_iteration += 1
        if self.val_batch is None or self.cond_proj is None:
            return
        tb = None
        for lg in (self.loggers or []):
            if hasattr(lg, "experiment") and hasattr(lg.experiment, "add_image"):
                tb = lg.experiment; break

        step = self._val_iteration
        device = next(self.transformer.parameters()).device
        self.transformer.eval()

        batch = self.val_batch
        imgs = batch["image"].to(device, dtype=self.model_dtype)
        ref_imgs = batch["ref_image"].to(device, dtype=self.model_dtype)
        materials = batch["condition"].to(device, dtype=self.model_dtype)
        edges_raw = batch["edges"].to(device, dtype=self.model_dtype)
        mask_image = batch["hint"][:, 0].to(device, dtype=self.model_dtype)
        prompts = batch["description"]
        B, _, H, W = imgs.shape
 
        mask_proc = self.flux_pipe.mask_processor.preprocess(mask_image, height=H, width=W)
        mask_proc = (mask_proc > 0.5).to(device=device, dtype=self.model_dtype) 
        ip_tokens = self._get_ip_tokens(materials[:, :3])
        cond_latents = self._build_conditioning_patch(materials, edges_raw, mask_proc, B, H, W, device)
        condition_ids = self.flux_pipe._prepare_latent_image_ids(B, 32, 32, device, self.model_dtype)

        x_0, img_ids = encode_images(self.flux_pipe, imgs)
        prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(self.flux_pipe, prompts)

        ref_proc = self.flux_pipe.image_processor.preprocess(ref_imgs, height=H, width=W).to(device=device, dtype=self.model_dtype)
        masked_rgb = ref_proc * (1 - mask_proc)
        masked_latents, cond_masked_latents, _ = self._vae_encode_rgb(masked_rgb)
        masked_latents = torch.cat([masked_latents, self._pack_mask_full(mask_proc, B, H, W)], dim=-1)
        cond_masked_latents = torch.cat([cond_masked_latents, self._pack_mask_cond(mask_proc, B)], dim=-1)
        combined_cond = self.cond_proj(torch.cat((cond_latents, cond_masked_latents), dim=2))

        condition_type_ids = torch.full((B, 1), 11, device=device, dtype=torch.long)
        sf = self.vae_scale_factor
        image_seq_len = (H // sf) * (W // sf)
        mu = calculate_shift(
            image_seq_len,
            self.flux_pipe.scheduler.config.base_image_seq_len,
            self.flux_pipe.scheduler.config.max_image_seq_len,
            self.flux_pipe.scheduler.config.base_shift,
            self.flux_pipe.scheduler.config.max_shift,
        )
        self.flux_pipe.scheduler.set_timesteps(self.val_num_steps, device=device, mu=mu)
        latents = torch.randn_like(x_0)
        guidance = torch.ones(B, device=device) if self.transformer.config.guidance_embeds else None

        for t_val in self.flux_pipe.scheduler.timesteps:
            t_norm = t_val.expand(B).to(device=device, dtype=self.model_dtype) / 1000.0
            hidden = torch.cat((latents.to(self.model_dtype), masked_latents), dim=-1)
            set_ip_tokens(self.transformer, ip_tokens)
            pred_out = tranformer_forward(
                self.transformer, model_config=self.model_config,
                condition_latents=combined_cond, condition_ids=condition_ids,
                condition_type_ids=condition_type_ids, hidden_states=hidden,
                timestep=t_norm, guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids, img_ids=img_ids,
                joint_attention_kwargs=None, return_dict=False,
            )[0]
            set_ip_tokens(self.transformer, None)
            latents = self.flux_pipe.scheduler.step(pred_out, t_val, latents).prev_sample

        vd, vt = (next(self.flux_pipe.vae.parameters()).device, next(self.flux_pipe.vae.parameters()).dtype)
        ld = self.flux_pipe._unpack_latents(latents, H, W, sf)
        ld = (ld / self.flux_pipe.vae.config.scaling_factor) + self.flux_pipe.vae.config.shift_factor
        pred_images = self.flux_pipe.vae.decode(ld.to(device=vd, dtype=vt), return_dict=False)[0]

        pred_disp = to_display(pred_images)
        gt_disp = to_display(imgs)
        ref_disp = to_display(ref_imgs)
        mat_disp = to_display(materials[:, :3])
        edge_disp = to_display(edges_raw[:, :1].repeat(1, 3, 1, 1))

        if self.lpips_fn is not None:
            self._last_val_lpips = self.lpips_fn(pred_disp, gt_disp.to(pred_disp.device)).item()
            print(f"  [val {step}] LPIPS: {self._last_val_lpips:.4f}")
            if tb: tb.add_scalar("val/lpips", self._last_val_lpips, step)

        if tb is not None:
            grid = torch.cat([mat_disp.cpu(), edge_disp.cpu(), ref_disp.cpu(), pred_disp.cpu(), gt_disp.cpu()], dim=0)
            grid_img = vutils.make_grid(grid, nrow=B, padding=4, normalize=False)
            log_dir  = getattr(self.trainer.logger, "log_dir", getattr(self.trainer, "default_root_dir", "."))
            save_dir = Path(log_dir) / "val_images"
            save_dir.mkdir(exist_ok=True)
            vutils.save_image(grid_img, save_dir / f"step_{step:06d}.png")
            tb.add_image("val/mat_edges_ref_pred_gt", grid_img, step)
            print(f"  [val {step}] → {save_dir}/step_{step:06d}.png")
        self.transformer.train()