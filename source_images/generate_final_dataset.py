import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

def resize_image_to_512(image_path: Path, output_path: Path):
    """
    Изменяет размер изображения до 512×512 пикселей и сохраняет результат. 
    """ 

    with Image.open(image_path) as img:
        if img.size == (512, 512):
            img.save(output_path)
            return
        img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
        img_resized.save(output_path)

def find_material_folder(render_folder_name: str, materials_root: Path) -> Path:
    """
    Находит подпапку с материалом, чьё имя является префиксом имени папки рендера. 
    """ 

    available = [d for d in materials_root.iterdir() if d.is_dir()]
    available.sort(key=lambda p: len(p.name), reverse=True)
    for mat_dir in available:
        if render_folder_name.startswith(mat_dir.name):
            return mat_dir
    return None

def extract_model_name(render_folder_name: str, material_name: str) -> str:
    """
    Убирает префикс материала из имени папки рендера, возвращает название модели.
    Удаляет указанное имя материала и следующий за ним символ подчёркивания (если есть). 
    """ 

    if render_folder_name.startswith(material_name):
        rest = render_folder_name[len(material_name):]
        if rest.startswith('_'):
            rest = rest[1:]
        return rest
    return ""

def generate_texture_and_lines(input_path: Path, mask_path: Path, edges_path: Path,
                               basecolor_path: Path, output_dir: Path,
                               tile_size: int = 256, repeats_per_side: int = 4,
                               thin_line_threshold: int = 180, thick_line_threshold: int = 210):
    """
    Создаёт несколько вариантов наложения текстуры и линий для обучающих семплов.

    На основе исходного изображения (input), маски (mask), карты краёв (edges) и
    текстуры материала (basecolor) генерирует:
      - input_new.png     : текстура + тонкие линии (thin_line_threshold)
      - input_new2.png    : текстура + толстые линии (thick_line_threshold)
      - edges_new2.png    : бинарная маска толстых линий (белые на чёрном)
      - edges_hint1.png   : 0.7*grad + 0.3*material (grayscale)
      - edges_hint2.png   : grad * (0.5 + 0.5*material) (grayscale)

    Текстура формируется путём тайлинга (мозаичного повторения) basecolor с размерами
    tile_size и repeats_per_side.

    Args:
        input_path (Path): Путь к изображению input.png (оригинал с фоном).
        mask_path (Path): Путь к маске mask.png (белый – объект, чёрный – фон).
        edges_path (Path): Путь к edges.png (цветная карта краёв, где линии яркие).
        basecolor_path (Path): Путь к basecolor.png (цветной материал).
        output_dir (Path): Директория для сохранения сгенерированных файлов.
        tile_size (int): Размер стороны квадратного тайла для повторения текстуры.
        repeats_per_side (int): Количество повторений тайла по горизонтали и вертикали
                                для создания базового блока.
        thin_line_threshold (int): Порог яркости инвертированных краёв для тонких линий.
        thick_line_threshold (int): Порог яркости инвертированных краёв для толстых линий.
    """
    
    input_img = Image.open(input_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L")
    edges_img = Image.open(edges_path).convert("RGB")
    basecolor_orig = Image.open(basecolor_path).convert("RGB")

    input_np = np.array(input_img)
    mask_np = np.array(mask_img)
    edges_np = np.array(edges_img)

    h, w = input_np.shape[:2]

    # --- Мозаичная текстура из basecolor ---
    basecolor_tile = basecolor_orig.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    basecolor_tile_np = np.array(basecolor_tile)
    total_size = tile_size * repeats_per_side
    basecolor_big = np.tile(basecolor_tile_np, (repeats_per_side, repeats_per_side, 1))
    repeat_y = int(np.ceil(h / total_size))
    repeat_x = int(np.ceil(w / total_size))
    tiled_texture = np.tile(basecolor_big, (repeat_y, repeat_x, 1))
    tiled_texture = tiled_texture[:h, :w, :]
    object_mask = mask_np > 128
    textured = input_np.copy()
    textured[object_mask] = tiled_texture[object_mask]

    # --- Линии из edges.png (инвертируем) ---
    edges_inv = 255 - edges_np
    brightness = edges_inv.mean(axis=-1)   # яркость (0..255)
    thin_lines = (brightness <= thin_line_threshold) & object_mask
    thick_lines = (brightness <= thick_line_threshold) & object_mask
    result_thin = textured.copy()
    result_thin[thin_lines] = [0, 0, 0]
    result_thick = textured.copy()
    result_thick[thick_lines] = [0, 0, 0]
    edges_thick_mask = np.zeros((h, w), dtype=np.uint8)
    edges_thick_mask[thick_lines] = 255

    Image.fromarray(result_thin).save(output_dir / "input_new.png")
    Image.fromarray(result_thick).save(output_dir / "input_new2.png")
    Image.fromarray(edges_thick_mask).save(output_dir / "edges_new2.png")

    sobel_path = output_dir / "sobel.png"
    basecolor_path_local = output_dir / "basecolor.png"
    if sobel_path.exists() and basecolor_path_local.exists():
        # Загружаем sobel (градиент) как grayscale
        sobel_img = Image.open(sobel_path).convert("L")
        grad_np = np.array(sobel_img, dtype=np.float32) / 255.0   # [0,1]
        # Загружаем basecolor, переводим в серый
        basecolor_gray = Image.open(basecolor_path_local).convert("L")
        material_np = np.array(basecolor_gray, dtype=np.float32) / 255.0
        # Вариант D1: 0.7*grad + 0.3*material
        hint1 = 0.7 * grad_np + 0.3 * material_np
        hint1 = np.clip(hint1 * 255, 0, 255).astype(np.uint8)
        hint1[~object_mask] = 0
        # Вариант D2: grad * (0.5 + 0.5*material)
        hint2 = grad_np * (0.5 + 0.5 * material_np)
        hint2 = np.clip(hint2 * 255, 0, 255).astype(np.uint8)
        hint2[~object_mask] = 0 
        Image.fromarray(hint1).save(output_dir / "edges_hint1.png")
        Image.fromarray(hint2).save(output_dir / "edges_hint2.png")
    else: 
        print(f"  ⚠️ {sobel_path} или {basecolor_path_local} не найдены, hint-версии не созданы")

def main():
    """
    Собирает финальный датасет для обучения из рендеров, обработанных данных и материалов,
    а также генерирует дополнительные варианты текстур и линий.

    Ожидается структура папок:
        training_assets/
            renders_output_train/           # папки вида <material>_<model>_view_<num>
                <folder>/
                    view_XXX.png
                    view_XXX_bg.png
            processed_renders_train/        # результат работы внешнего скрипта
                <model>_view_<num>/
                    mask.png, edges.png, saturation.png, sobel.png, canny.png
            MatSynth_few/                   # материалы (из MatSynth или выбранные)

            final_dataset_train/            # на выходе папки sample_XXXXXX (создаются)

    Алгоритм:
        1. Находит все папки рендеров.
        2. Для каждой папки определяет соответствующий материал (по префиксу).
        3. Извлекает имя модели.
        4. Для каждого ракурса проверяет наличие view_XXX.png и *_bg.png, а также папки
           processed_renders.
        5. Создаёт папку sample_XXXXXX и копирует/генерирует все необходимые файлы:
           - input.png, target.png, basecolor.png
           - файлы из processed_renders (mask, edges, saturation, sobel, canny)
           - дополнительные варианты через generate_texture_and_lines.
        6. Переход к следующему ракурсу.

    При ошибках (отсутствие файлов, невозможность создать папку) пропускает некорректные
    семплы и продолжает обработку. В конце выводится статистика.

    Настройки (TILE_SIZE, REPEATS_PER_SIDE, пороги) задаются в теле main().
    """ 
    # ===== НАСТРОЙКИ (меняйте под свои эксперименты) =====
    TILE_SIZE = 256               # размер тайла текстуры
    REPEATS_PER_SIDE = 4          # повторений (4x4 = 16 тайлов, итого 1024x1024, обрежется)
    THIN_LINE_THRESHOLD = 180     # порог для тонких линий (input_new.png)
    THICK_LINE_THRESHOLD = 210    # порог для толстых линий (input_new2.png, edges_new2.png)
    # =====================================================
 
    renders_root = Path("training_assets/renders_output")
    processed_root = Path("training_assets/processed_renders")
    materials_root = Path("training_assets/MatSynth")

    output_root = Path("training_assets/final_dataset_train")

    output_root.mkdir(parents=True, exist_ok=True)

    for p in [renders_root, processed_root, materials_root]:
        if not p.exists():
            print(f"Ошибка: папка не существует {p}")
            return

    render_folders = [d for d in renders_root.iterdir() if d.is_dir()]
    print(f"Найдено папок с рендерами: {len(render_folders)}")

    sample_counter = 0
    errors = []

    for render_folder in tqdm(render_folders, desc="Обработка"):
        material_dir = find_material_folder(render_folder.name, materials_root)
        if material_dir is None:
            print(f"⚠️ Не найден материал для {render_folder.name}. Пропускаем.")
            continue
        basecolor_path = material_dir / "basecolor.png"
        if not basecolor_path.exists():
            print(f"⚠️ Нет basecolor.png в {material_dir}. Пропускаем {render_folder.name}.")
            continue

        material_name = material_dir.name
        model = extract_model_name(render_folder.name, material_name)
        if not model:
            print(f"⚠️ Не удалось извлечь модель из {render_folder.name} для материала {material_name}. Пропускаем.")
            continue

        view_files = list(render_folder.glob("view_*.png"))
        view_nums = set()
        for f in view_files:
            stem = f.stem
            if stem.startswith("view_"):
                num_part = stem.split("_")[-1]
                if num_part.isdigit():
                    view_nums.add(int(num_part))
        view_nums = sorted(view_nums)

        for view_num in view_nums:
            view_with_mat = render_folder / f"view_{view_num:03d}.png"
            view_original_bg = render_folder / f"view_{view_num:03d}_bg.png"
            if not view_with_mat.exists() or not view_original_bg.exists():
                errors.append(f"Пропущен ракурс {view_num:03d} в {render_folder.name}: отсутствует один из файлов")
                continue

            proc_folder = processed_root / f"{model}_view_{view_num:03d}"
            if not proc_folder.exists():
                errors.append(f"Нет папки processed_renders/{proc_folder.name} для {render_folder.name}")
                continue

            sample_dir = output_root / f"sample_{sample_counter:06d}"
            sample_dir.mkdir(parents=True, exist_ok=False)

            try:
                for needed in ["mask.png", "edges.png", "saturation.png", "sobel.png", "canny.png"]:
                    src = proc_folder / needed
                    if src.exists():
                        shutil.copy2(src, sample_dir / needed)
                    else:
                        print(f"  ⚠️ Отсутствует {needed} в {proc_folder}")

                shutil.copy2(view_original_bg, sample_dir / "input.png")
                shutil.copy2(view_with_mat, sample_dir / "target.png")
                resize_image_to_512(basecolor_path, sample_dir / "basecolor.png")

                generate_texture_and_lines(
                    input_path=sample_dir / "input.png",
                    mask_path=sample_dir / "mask.png",
                    edges_path=sample_dir / "edges.png",
                    basecolor_path=sample_dir / "basecolor.png",
                    output_dir=sample_dir,
                    tile_size=TILE_SIZE,
                    repeats_per_side=REPEATS_PER_SIDE,
                    thin_line_threshold=THIN_LINE_THRESHOLD,
                    thick_line_threshold=THICK_LINE_THRESHOLD
                )

                sample_counter += 1

            except Exception as e:
                print(f"❌ Ошибка при обработке {render_folder.name} view_{view_num:03d}: {e}")
                shutil.rmtree(sample_dir, ignore_errors=True)
                errors.append(str(e))

    print(f"\n✅ Готово. Создано сэмплов: {sample_counter}")
    if errors:
        print(f"⚠️ Возникли ошибки ({len(errors)}):")
        for err in errors[:10]:
            print(f"   {err}")
        if len(errors) > 10:
            print(f"   ... и ещё {len(errors)-10}")

if __name__ == "__main__":
    main()