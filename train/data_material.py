import os
import random
import numpy as np
from PIL import Image 
from torch.utils.data import Dataset
import torchvision.transforms as T

class MaterialDataSet(Dataset):
    def __init__(
        self,
        root_dir: str,
        condition_type: str = "material_transfer",
        material_channel: str = "basecolor",   # какая карта материала (basecolor, diffuse, specular, ...)
        use_edges: bool = True,                # добавлять ли edges.png как 4-й канал
        image_size: int = 512,
        drop_condition_prob: float = 0.0,      # вероятность заменить condition на ноль (для robustness)
        drop_text_prob: float = 0.0,           # вероятность сделать промпт пустым
    ):
        self.root_dir = root_dir
        self.condition_type = condition_type
        self.material_channel = material_channel
        self.use_edges = use_edges
        self.image_size = (image_size, image_size)
        self.drop_condition_prob = drop_condition_prob
        self.drop_text_prob = drop_text_prob
 
        self.samples = []
        for name in sorted(os.listdir(root_dir)):
            if name.startswith("sample_") and os.path.isdir(os.path.join(root_dir, name)):
                self.samples.append(name)
        if not self.samples:
            raise RuntimeError(f"No sample_* folders found in {root_dir}")
        print(f"MaterialDataSet loaded: {len(self.samples)} samples")

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path, as_rgb=True, normalize=True):
        """Загружает изображение, ресайзит, возвращает numpy array (H,W,C) в [0,1]."""

        img = Image.open(path).convert("RGB" if as_rgb else "L")
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        if normalize:
            img_np = np.array(img).astype(np.float32) / 255.0
        else:
            img_np = np.array(img).astype(np.float32)
        return img_np

    def _generate_description(self, has_edges):
        """Формирует текстовый промпт на основе типа материала и наличия границ."""

        if self.material_channel == "basecolor":
            mat_name = "base color" 
        else:
            mat_name = self.material_channel
        base = f"high-quality transfer of {mat_name} material texture"
        if has_edges:
            base += " with sharp structural edges"
        else:
            base += " without explicit edges"
        return base
            
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        sample_path = os.path.join(self.root_dir, sample_name)
 
        input_path  = os.path.join(sample_path, "input_new.png")
        target_path = os.path.join(sample_path, "target.png")
        mask_path   = os.path.join(sample_path, "mask.png")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing input_new.png in {sample_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Missing target.png in {sample_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing mask.png in {sample_path}")
 
        image     = self._load_image(target_path, as_rgb=True)  
        ref_image = self._load_image(input_path,  as_rgb=True)   
 
        hint = self._load_image(mask_path, as_rgb=False)
        if hint.ndim == 2:
            hint = hint[..., np.newaxis]
 
        material_path = os.path.join(sample_path, f"{self.material_channel}.png")
        if not os.path.exists(material_path):
            material_path = os.path.join(sample_path, "basecolor.png")
            if not os.path.exists(material_path):
                raise FileNotFoundError(f"Missing material map in {sample_path}")

        material_rgb = self._load_image(material_path, as_rgb=True)
 
        if self.use_edges:
            edges_path = os.path.join(sample_path, "sobel.png")

            if os.path.exists(edges_path):
                edges = self._load_image(edges_path, as_rgb=False)
                if edges.ndim == 2:
                    edges = edges[..., np.newaxis]
            else:
                edges = np.zeros(self.image_size + (1,), dtype=np.float32)

            has_edges = True
        else:
            edges = np.zeros(self.image_size + (1,), dtype=np.float32)
            has_edges = False
 
        condition = material_rgb
 
        if random.random() < self.drop_condition_prob:
            condition = np.zeros_like(condition)
 
        description = self._generate_description(has_edges)
        if random.random() < self.drop_text_prob:
            description = ""
            
        image_tensor     = self.to_tensor(image)
        ref_tensor       = self.to_tensor(ref_image)
        condition_tensor = self.to_tensor(condition)  
        edges_tensor     = self.to_tensor(edges) 
        hint_tensor      = self.to_tensor(hint)
        
        return {
            "image": image_tensor,
            "ref_image": ref_tensor,
            "condition": condition_tensor,
            "edges": edges_tensor,
            "hint": hint_tensor,
            "condition_type": self.condition_type,
            "description": description, 
            "position_delta": np.array([0, 0], dtype=np.float32),
            "n_lines": 0,
            "gly_line": [],
            "language": [],
            "positions": [],
            "texts": [],
            "attnmask": np.zeros(self.image_size, dtype=np.uint8),
        }