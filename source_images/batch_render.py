import subprocess
from pathlib import Path

num_views = 10 # Количество ракурсов для каждого объекта и каждого материала

blender_exe = "/Applications/Blender.app/Contents/MacOS/blender"

materials_root = Path("training_assets/MatSynth")
models_root = Path("training_assets/Models")
output_root = Path("training_assets/renders_output")
backgrounds_dir = Path("training_assets/Backgrounds")

script_path = Path("source_images/render_objects.py").absolute()

material_dirs = [d for d in materials_root.iterdir() if d.is_dir()]
model_files = list(models_root.glob("*.obj"))

print(f"Найдено материалов: {len(material_dirs)}")
print(f"Найдено моделей: {len(model_files)}")
print(f"Фоны: {backgrounds_dir}")

# Рендерим оригиналы (без материалов)
print("\n=== Рендеринг оригиналов ===")
for model_file in model_files:
    cmd = [
        blender_exe, "-b", "-P", str(script_path), "--",
        "SKIP", str(model_file), str(output_root), str(num_views),
        "--bg_dir", str(backgrounds_dir)
    ]
    subprocess.run(cmd)

# Рендерим с материалами
print("\n=== Рендеринг с материалами ===")
for mat_dir in material_dirs:
    for model_file in model_files:
        out_dir = output_root / f"{mat_dir.name}_{model_file.stem}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            blender_exe, "-b", "-P", str(script_path), "--",
            str(mat_dir), str(model_file), str(out_dir), str(num_views),
            "--skip_original",
            "--bg_dir", str(backgrounds_dir)
        ]
        subprocess.run(cmd)