import argparse
from datasets import load_dataset
from pathlib import Path

def extract_type(name):
    """Извлекает обобщённый тип из имени материала..."""

    parts = name.split('_') 
    meaningful_parts = [p for p in parts if not p.isdigit()]
    if len(meaningful_parts) >= 2:
        return '_'.join(meaningful_parts[:2])
    elif meaningful_parts:
        return '_'.join(meaningful_parts)
    else:
        return name 

def download_few_materials(num_materials=10, output_dir="../training_assets/MatSynth"):
    """Скачивает материалы, выбирая не более одного на каждый тип."""

    ds = load_dataset("gvecchio/MatSynth", split="train", streaming=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    collected = 0
    used_types = set()
    
    for item in ds:
        if collected >= num_materials:
            break
        name = item["name"]
        mat_type = extract_type(name)
        if mat_type in used_types:
            continue
        
        print(f"Сохранение материала {name} (тип: {mat_type})...")
        mat_dir = output_dir / name
        mat_dir.mkdir(exist_ok=True)
        for key in ["basecolor", "normal", "roughness", "metallic", "displacement", "diffuse", "specular"]:
            if key in item and item[key] is not None:
                img = item[key]
                img.save(mat_dir / f"{key}.png")
        used_types.add(mat_type)
        collected += 1
    
    print(f"Сохранено {collected} материалов (уникальных типов) в {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download distinct material types from MatSynth.")
    parser.add_argument("num", type=int, nargs="?", default=10, help="Number of materials to download (default: 10)")
    args = parser.parse_args()
    
    download_few_materials(num_materials=args.num)