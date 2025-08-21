import os

# 📂 ruta absoluta al dataset
dataset_root = r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset"

# --- PARTE 1: revisar y corregir labels ---
EXPECTED_COLS = 68  # YOLOv8 espera 68 columnas (1 clase + 21 keypoints * 3 + extras si aplica)

corrupt_files = []

for root, dirs, files in os.walk(dataset_root):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                lines = f.readlines()
            new_lines = []
            fixed = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) != EXPECTED_COLS:
                    corrupt_files.append(path)
                    if len(parts) > EXPECTED_COLS:
                        parts = parts[:EXPECTED_COLS]  # recorta si hay columnas de más
                        fixed = True
                    elif len(parts) < EXPECTED_COLS:
                        parts += ["0.0"] * (EXPECTED_COLS - len(parts))  # rellena con 0.0 si faltan
                        fixed = True
                new_lines.append(" ".join(parts))
            if fixed:
                with open(path, "w") as f:
                    f.write("\n".join(new_lines))

print(f"archivos corruptos detectados y corregidos: {len(corrupt_files)}")
for f in corrupt_files:
    print(f)
print("✅ revisión y corrección completa de labels")

# --- PARTE 2: renombrar imágenes y labels ---
for root, dirs, files in os.walk(dataset_root):
    for file in files:
        old_path = os.path.join(root, file)
        # reemplazar espacios y guiones raros por guion bajo
        new_name = file.replace(" ", "_").replace("-", "_")
        new_name = new_name.replace("(", "").replace(")", "")
        new_path = os.path.join(root, new_name)
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"renombrado: {old_path} → {new_path}")

print("✅ renombrado de imágenes y labels completado")
