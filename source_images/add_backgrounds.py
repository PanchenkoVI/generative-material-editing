from pathlib import Path
from PIL import Image

backgrounds_dir = Path("./training_assets/Backgrounds")         # папки с фонами 
renders_root = Path("./training_assets/renders_output")         # папки с материалами
original_root = Path("./training_assets/original_renders")      # оригиналы (прозрачные PNG) 

original_names = [d.name for d in original_root.iterdir() if d.is_dir()]

def find_original_dir(mat_folder_name: str) -> Path | None:
    """По имени папки материала ищет соответствующую папку оригинала."""

    for orig in original_names:
        if mat_folder_name.endswith(orig):
            return original_root / orig
    parts = mat_folder_name.split('_')
    for i in range(len(parts) - 1, 0, -1):
        candidate = '_'.join(parts[i-1:])
        if candidate in original_names:
            return original_root / candidate
    return None

# Загружаем все фоны
all_backgrounds = sorted(backgrounds_dir.glob("*.png")) + sorted(backgrounds_dir.glob("*.jpg"))
if not all_backgrounds:
    raise RuntimeError(f"Нет фонов в {backgrounds_dir}")

def overlay_background(input_png, bg_path, output_png):
    obj = Image.open(input_png).convert("RGBA")
    bg = Image.open(bg_path).convert("RGBA")
    bg = bg.resize(obj.size, Image.Resampling.LANCZOS)
    composite = Image.alpha_composite(bg, obj)
    composite.save(output_png)

# Проходим по каждой папке материала
for mat_folder in renders_root.glob("*"):
    if not mat_folder.is_dir():
        continue
    print(f"Обрабатываю: {mat_folder.name}")

    orig_dir = find_original_dir(mat_folder.name)
    if orig_dir is None:
        print(f"  ⚠️ Папка оригинала не найдена для {mat_folder.name}. Пропускаю.")
        continue

    # Уникальный сдвиг для этой папки (хеш от имени)
    offset = hash(mat_folder.name) % len(all_backgrounds)

    for png_path in sorted(mat_folder.glob("view_*.png")):
        if "_bg" in png_path.stem:
            continue

        view_num = int(png_path.stem.split("_")[-1])
        bg_index = (view_num + offset) % len(all_backgrounds)
        bg_path = all_backgrounds[bg_index]

        # 1. Рендер с материалом
        overlay_background(png_path, bg_path, png_path)
        print(f"    {png_path.name} -> фон #{bg_index} (материал)")

        # 2. Оригинал
        original_png = orig_dir / f"view_{view_num:03d}.png"
        if not original_png.exists():
            print(f"    ⚠️ Пропущен {original_png}: файл не найден")
            continue

        bg_output = mat_folder / f"view_{view_num:03d}_bg.png"
        overlay_background(original_png, bg_path, bg_output)
        print(f"    {bg_output.name} -> фон #{bg_index} (оригинал)")