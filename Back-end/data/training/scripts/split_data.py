import os
import random
import shutil

# 📁 rutas base
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset"))
IMAGES_SRC = os.path.join(DATASET_DIR, "images")
LABELS_SRC = os.path.join(DATASET_DIR, "labels")

# ⚖️ porcentajes
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 42
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def is_image(fn):
    return os.path.splitext(fn)[1].lower() in IMG_EXTS

def main():
    random.seed(SEED)

    # recorrer todas las carpetas derecha/izquierda + A/B/C
    for side in ["derecha", "izquierda"]:
        for letter in ["A", "B", "C"]:
            folder_img = os.path.join(IMAGES_SRC, side, letter)
            folder_lbl = os.path.join(LABELS_SRC, side, letter)
            if not os.path.exists(folder_img):
                continue

            imgs = [f for f in os.listdir(folder_img) if is_image(f)]
            random.shuffle(imgs)
            n = len(imgs)
            n_train = int(n * SPLITS["train"])
            n_val = int(n * SPLITS["val"])
            n_test = n - n_train - n_val

            split_dict = {
                "train": imgs[:n_train],
                "val": imgs[n_train:n_train+n_val],
                "test": imgs[n_train+n_val:]
            }

            for split, files in split_dict.items():
                for f in files:
                    src_img = os.path.join(folder_img, f)
                    src_lbl = os.path.join(folder_lbl, os.path.splitext(f)[0]+".txt")

                    # crear carpetas destino
                    dst_img = os.path.join(IMAGES_SRC, split, side, letter, f)
                    dst_lbl = os.path.join(LABELS_SRC, split, side, letter, os.path.splitext(f)[0]+".txt")
                    ensure_dir(os.path.dirname(dst_img))
                    ensure_dir(os.path.dirname(dst_lbl))

                    # copiar archivos
                    shutil.copy2(src_img, dst_img)
                    if os.path.exists(src_lbl):
                        shutil.copy2(src_lbl, dst_lbl)
                    else:
                        print(f"⚠️ No se encontró label para: {side}/{letter}/{f}")

    print("✅ Split completado.")

if __name__ == "__main__":
    main()
