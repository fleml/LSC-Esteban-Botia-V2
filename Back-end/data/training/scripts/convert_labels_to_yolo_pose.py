import os
import shutil

# rutas absolutas según lo que me pasaste
IMAGES_DIR = r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\images"
LABELS_DIR = r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\labels"

BACKUP_DIR = LABELS_DIR + "_raw_backup"
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

NUM_KPTS = 21
VIS_VALUE = 2  # visibilidad fija para YOLO-Pose

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def find_image_for_label(label_path):
    rel = os.path.relpath(label_path, LABELS_DIR)
    base, _ = os.path.splitext(rel)
    for ext in IMG_EXTS:
        cand = os.path.join(IMAGES_DIR, base + ext)
        if os.path.exists(cand):
            return cand
    return None

def convert_label_file(src_txt, dst_txt):
    with open(src_txt, "r", encoding="utf-8") as f:
        line = f.read().strip()
    if not line:
        return False

    parts = line.split()
    cls = int(parts[0])
    vals = list(map(float, parts[1:]))

    if len(vals) != 3 * NUM_KPTS:
        print(f"⚠️ {src_txt}: se esperaban {3*NUM_KPTS} valores y hay {len(vals)}.")
        return False

    kpts = []
    xs, ys = [], []
    for i in range(0, len(vals), 3):
        x = min(max(vals[i],   0.0), 1.0)
        y = min(max(vals[i+1], 0.0), 1.0)
        # ignoramos vals[i+2] (vis original)
        kpts.append((x, y))
        xs.append(x); ys.append(y)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = max(1e-6, x_max - x_min)
    h = max(1e-6, y_max - y_min)
    xc = x_min + w / 2
    yc = y_min + h / 2

    out = [str(cls), f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for (x, y) in kpts:
        out += [f"{x:.6f}", f"{y:.6f}", str(VIS_VALUE)]

    ensure_dir(os.path.dirname(dst_txt))
    with open(dst_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(out) + "\n")
    return True

def main():
    if not os.path.exists(BACKUP_DIR):
        print(f"📦 Respaldo en: {BACKUP_DIR}")
        shutil.copytree(LABELS_DIR, BACKUP_DIR)
    else:
        print(f"ℹ️ Respaldo ya existe: {BACKUP_DIR}")

    total, ok = 0, 0
    for root, _, files in os.walk(LABELS_DIR):
        for fn in files:
            if not fn.lower().endswith(".txt"):
                continue
            total += 1
            src_txt = os.path.join(root, fn)

            if not find_image_for_label(src_txt):
                print(f"⚠️ Sin imagen correspondiente para: {src_txt}")
                continue

            if convert_label_file(src_txt, src_txt):
                ok += 1

    print(f"✅ Conversión YOLO-Pose: {ok}/{total} labels convertidos.")

if __name__ == "__main__":
    main()
