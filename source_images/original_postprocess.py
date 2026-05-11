import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil
import json

def process_one_image(image_path: Path, output_sample_dir: Path,
                      min_contrast: float = 0.0) -> bool:
    img = Image.open(image_path).convert('RGBA')
    img_rgb = np.array(img.convert('RGB'))
    alpha = np.array(img.split()[-1])
    mask = (alpha > 0).astype(np.uint8) * 255
    mask_bool = mask > 0

    if np.count_nonzero(mask_bool) == 0:
        print(f"  ⚠️ {image_path.name}: пустая маска, пропускаем")
        return False 
    Image.fromarray(mask).save(output_sample_dir / "mask.png")

    # Фильтр контраста
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    std_luma = np.std(gray[mask_bool])
    if std_luma < min_contrast:
        print(f"  ⚠️ {image_path.name}: контраст {std_luma:.1f} ниже {min_contrast}, пропускаем")
        return False

    gray_on_gray = np.where(mask_bool, gray, 128).astype(np.uint8)
    blur = cv2.GaussianBlur(gray_on_gray, (3, 3), 0)

    # ---------- Edge Drawing  ---------- 
    if not hasattr(cv2, 'ximgproc') or not hasattr(cv2.ximgproc, 'createEdgeDrawing'):
        raise ImportError("Edge Drawing требует opencv-contrib-python. Установите: pip install opencv-contrib-python")
    
    ed = cv2.ximgproc.createEdgeDrawing() 
    ed.detectEdges(blur)
    edges_ed = ed.getEdgeImage() 
    kernel = np.ones((2, 2), np.uint8)
    edges_ed = cv2.dilate(edges_ed, kernel, iterations=1) 
    edges_ed_masked = cv2.bitwise_and(edges_ed, edges_ed, mask=mask)
    Image.fromarray(edges_ed_masked).save(output_sample_dir / "canny.png") 
    edges = edges_ed

    # -----------------------------------------------

    # --- SOBEL   ---
    sobelx = cv2.Sobel(gray_on_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_on_gray, cv2.CV_32F, 0, 1, ksize=3)

    grad = np.sqrt(sobelx**2 + sobely**2)
    grad = grad / (grad.max() + 1e-6)
    grad = grad ** 0.5
    grad = np.clip(grad * 255, 0, 255).astype(np.uint8)
    sobel_masked = cv2.bitwise_and(grad, grad, mask=mask)
    Image.fromarray(sobel_masked).save(output_sample_dir / "sobel.png")

    # Комбинируем ED и Sobel
    edges = cv2.max(edges, grad)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
    Image.fromarray(edges_masked).save(output_sample_dir / "edges.png")

    # --- saturation   ---
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    sat_min = sat[mask_bool].min()
    sat_max = sat[mask_bool].max()
    if sat_max > sat_min:
        sat_norm = (sat - sat_min) / (sat_max - sat_min)
    else:
        sat_norm = np.zeros_like(sat)
    val_min = val[mask_bool].min()
    val_max = val[mask_bool].max()
    if val_max > val_min:
        val_norm = (val - val_min) / (val_max - val_min)
    else:
        val_norm = np.zeros_like(val)
    sat_vis = 0.7 * sat_norm + 0.3 * val_norm
    sat_vis = np.clip(sat_vis * 255, 0, 255).astype(np.uint8)
    sat_vis = cv2.medianBlur(sat_vis, 3)
    sat_vis[~mask_bool] = 0
    Image.fromarray(sat_vis).save(output_sample_dir / "saturation.png")

    # --- meta (без изменений) ---
    meta = {
        "source_file": image_path.name,
        "has_normals": False,
        "contrast_std": float(std_luma),
        "saturation_mean": float(np.mean(sat[mask_bool])) if mask_bool.any() else 0.0,
        "saturation_min": float(sat_min),
        "saturation_max": float(sat_max),
    }
    with open(output_sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return True 


def main(): 
    input_root = Path("training_assets/original_renders")
    output_dir = Path("training_assets/processed_renders") 

    output_dir.mkdir(exist_ok=True)

    MIN_CONTRAST = 0.0   # можно поднять, если нужно фильтровать

    image_extensions = {".png", ".jpg", ".jpeg"} 
    items = []  
    for model_folder in input_root.iterdir():
        if not model_folder.is_dir():
            continue
        for file_path in model_folder.iterdir():
            if file_path.suffix.lower() in image_extensions:
                items.append((model_folder.name, file_path))

    if not items:
        print(f"Нет изображений в подпапках {input_root}")
        return

    print(f"Найдено {len(items)} изображений")
    succeeded = 0

    for model_name, img_file in tqdm(items, desc="Обработка"): 
        sample_name = f"{model_name}_{img_file.stem}"
        sample_dir = output_dir / sample_name
        sample_dir.mkdir(exist_ok=True)

        if process_one_image(img_file, sample_dir, MIN_CONTRAST):
            succeeded += 1
        else:
            shutil.rmtree(sample_dir, ignore_errors=True)

    print(f"\nГотово. Сохранено сэмплов: {succeeded} из {len(items)}")
    if MIN_CONTRAST > 0:
        print(f"Порог контраста: >= {MIN_CONTRAST}")

if __name__ == "__main__":
    main()