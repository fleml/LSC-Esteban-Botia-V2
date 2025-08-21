# 📌 augmentation_faltantes.py
# Genera data augmentation SOLO para las imágenes faltantes
# Soporta labels con "_" y "__"
# Esteban Botia - 2025

import cv2
import numpy as np
import random
from pathlib import Path

# =========================
# CONFIGURACIÓN
# =========================
IMAGES_MISSING = [
    r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\images\train\derecha\B\B_DERECHA_1.jpg",
    r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\images\train\derecha\B\B_DERECHA_2.jpg",
    r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\images\train\derecha\B\B_DERECHA_3.jpg",
    r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\images\train\derecha\C\C_DERECHA_1.jpg",
    r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\images\train\derecha\C\C_DERECHA_2.jpg",
    r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\images\train\derecha\C\C_DERECHA_3.jpg",
    r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\images\train\izquierda\C\C_IZQUIERDA_1.jpg",
    r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\images\train\izquierda\C\C_IZQUIERDA_2.jpg",
    r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\images\train\izquierda\C\C_IZQUIERDA_3.jpg",
]

AUG_PER_IMAGE = 3

# =========================
# FUNCIONES
# =========================

def rotate_image_and_keypoints(image, keypoints, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    rotated_kpts = []
    for i in range(0, len(keypoints), 3):
        x = keypoints[i] * w
        y = keypoints[i+1] * h
        v = keypoints[i+2]
        if v > 0:
            coords = np.dot(M, np.array([x, y, 1]))
            nx = coords[0] / w
            ny = coords[1] / h
            rotated_kpts.extend([max(0,min(1,nx)), max(0,min(1,ny)), v])
        else:
            rotated_kpts.extend([keypoints[i], keypoints[i+1], v])
    return rotated_img, rotated_kpts


def flip_image_and_keypoints(image, keypoints):
    flipped_img = cv2.flip(image, 1)
    h, w = image.shape[:2]
    flipped_kpts = []
    for i in range(0, len(keypoints), 3):
        x = keypoints[i]
        y = keypoints[i+1]
        v = keypoints[i+2]
        if v > 0:
            flipped_kpts.extend([1 - x, y, v])
        else:
            flipped_kpts.extend([x, y, v])
    return flipped_img, flipped_kpts


def find_label_path(img_path: Path) -> Path | None:
    """
    Busca el label asociado a la imagen, probando con "_" y "__"
    """
    lbl_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")

    if lbl_path.exists():
        return lbl_path

    # probar reemplazando "_" por "__"
    lbl_double = Path(str(lbl_path).replace("_", "__"))
    if lbl_double.exists():
        return lbl_double

    return None


def process_image(img_path):
    img_path = Path(img_path)
    lbl_path = find_label_path(img_path)

    dataset_type = "train" if "train" in str(img_path) else "val"

    if lbl_path is None or not lbl_path.exists():
        print(f"❌ Label no encontrado para {img_path.name} ({dataset_type})")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ No se pudo leer la imagen {img_path}")
        return

    with open(lbl_path, "r") as f:
        label_data = f.readline().strip().split()

    if len(label_data) < 4 + 21 * 3:
        print(f"❌ Label inválido en {lbl_path}")
        return

    class_id = int(label_data[0])
    bbox = list(map(float, label_data[1:5]))
    keypoints = list(map(float, label_data[5:]))

    for aug_idx in range(AUG_PER_IMAGE):
        aug_img_name = f"{img_path.stem}_aug{aug_idx}.jpg"
        aug_lbl_name = f"{img_path.stem}_aug{aug_idx}.txt"
        aug_img_path = img_path.parent / aug_img_name
        aug_lbl_path = lbl_path.parent / aug_lbl_name

        if aug_img_path.exists() and aug_lbl_path.exists():
            print(f"⏩ Ya existe {aug_img_name}, skip.")
            continue

        aug_type = random.choice(["rotate", "flip"])
        if aug_type == "rotate":
            angle = random.uniform(-15, 15)
            aug_img, aug_kpts = rotate_image_and_keypoints(img, keypoints, angle)
        else:
            aug_img, aug_kpts = flip_image_and_keypoints(img, keypoints)

        cv2.imwrite(str(aug_img_path), aug_img)

        aug_line = [str(class_id)] + [f"{x:.6f}" for x in bbox] + [f"{x:.6f}" for x in aug_kpts]
        with open(aug_lbl_path, "w") as f:
            f.write(" ".join(aug_line) + "\n")

        print(f"✅ Generado {aug_img_name} ({dataset_type})")


if __name__ == "__main__":
    print("🚀 Iniciando augmentation solo para las imágenes faltantes...")
    for path in IMAGES_MISSING:
        process_image(path)
    print("🎯 Finalizado.")
