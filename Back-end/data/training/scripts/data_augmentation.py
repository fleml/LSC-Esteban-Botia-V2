# 📌 data_augmentation.py (versión corregida)
# Genera imágenes augmentadas físicamente con sus labels correspondientes
# Compatible con dataset YOLO Pose (21 keypoints)
# Esteban Botia - 2025 (fix by ChatGPT)

import os
import cv2
import glob
import random
import numpy as np
from pathlib import Path

# =========================
# CONFIGURACIÓN DE RUTAS
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]  # Back-End/data
IMAGES_DIR = BASE_DIR / "dataset/images"
LABELS_DIR = BASE_DIR / "dataset/labels"

TRAIN_IMG = IMAGES_DIR / "train"
VAL_IMG = IMAGES_DIR / "val"
TRAIN_LBL = LABELS_DIR / "train"
VAL_LBL = LABELS_DIR / "val"

# Cuántas imágenes nuevas quieres por cada imagen original
AUG_PER_IMAGE = 3

# =========================
# FUNCIONES DE AUGMENTATION
# =========================

def clip_coord(val):
    """Forzar coordenadas dentro de [0,1]."""
    return min(max(val, 0.0), 1.0)

def recalc_bbox_from_keypoints(kpts):
    """Recalcula el bbox (x_center, y_center, w, h) a partir de los keypoints visibles."""
    xs, ys = [], []
    for i in range(0, len(kpts), 3):
        x, y, v = kpts[i], kpts[i+1], kpts[i+2]
        if v > 0:
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return [0.5, 0.5, 1.0, 1.0]  # bbox dummy, evitar crash
    x_min, x_max = max(min(xs), 0.0), min(max(xs), 1.0)
    y_min, y_max = max(min(ys), 0.0), min(max(ys), 1.0)
    w = x_max - x_min
    h = y_max - y_min
    x_c = x_min + w / 2
    y_c = y_min + h / 2
    return [x_c, y_c, w, h]

def rotate_image_and_keypoints(image, keypoints, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # rotar imagen
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # rotar keypoints
    rotated_kpts = []
    for i in range(0, len(keypoints), 3):
        x = keypoints[i] * w
        y = keypoints[i+1] * h
        v = keypoints[i+2]
        if v > 0:
            coords = np.dot(M, np.array([x, y, 1]))
            nx = clip_coord(coords[0] / w)
            ny = clip_coord(coords[1] / h)
            rotated_kpts.extend([nx, ny, v])
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
            flipped_kpts.extend([clip_coord(1 - x), clip_coord(y), v])
        else:
            flipped_kpts.extend([x, y, v])
    return flipped_img, flipped_kpts

# =========================
# PROCESAR DATASET
# =========================

def process_folder(img_folder, lbl_folder):
    img_paths = glob.glob(str(img_folder / "**" / "*.jpg"), recursive=True) + \
                glob.glob(str(img_folder / "**" / "*.png"), recursive=True)

    if not img_paths:
        print(f"⚠️ No se encontraron imágenes en {img_folder}")
        return

    for img_path in img_paths:
        img_name = Path(img_path).stem
        lbl_path = lbl_folder / Path(img_path).relative_to(img_folder)
        lbl_path = lbl_path.with_suffix(".txt")

        if not lbl_path.exists():
            print(f"⚠️ No se encontró label para {img_name}")
            continue

        # leer imagen y label
        img = cv2.imread(img_path)
        with open(lbl_path, "r") as f:
            label_data = f.readline().strip().split()
        
        if len(label_data) < 4 + 21 * 3:
            print(f"❌ Label inválido en {lbl_path}")
            continue

        class_id = int(label_data[0])
        bbox = list(map(float, label_data[1:5]))
        keypoints = list(map(float, label_data[5:]))

        for aug_idx in range(AUG_PER_IMAGE):
            aug_type = random.choice(["rotate", "flip"])
            
            if aug_type == "rotate":
                angle = random.uniform(-15, 15)
                aug_img, aug_kpts = rotate_image_and_keypoints(img, keypoints, angle)
            else:
                aug_img, aug_kpts = flip_image_and_keypoints(img, keypoints)

            # recalcular bbox desde los keypoints válidos
            aug_bbox = recalc_bbox_from_keypoints(aug_kpts)

            # guardar imagen augmentada
            aug_img_name = f"{img_name}_aug{aug_idx}.jpg"
            aug_img_path = Path(img_path).parent / aug_img_name
            cv2.imwrite(str(aug_img_path), aug_img)

            # guardar label augmentado
            aug_lbl_path = Path(lbl_path).parent / f"{img_name}_aug{aug_idx}.txt"
            with open(aug_lbl_path, "w") as f:
                aug_line = [str(class_id)] + [f"{x:.6f}" for x in aug_bbox] + [f"{x:.6f}" for x in aug_kpts]
                f.write(" ".join(aug_line) + "\n")

    print(f"✅ Augmentation completado en {img_folder}")

if __name__ == "__main__":
    print("🚀 Iniciando data augmentation físico para YOLO Pose...")
    process_folder(TRAIN_IMG, TRAIN_LBL)
    process_folder(VAL_IMG, VAL_LBL)
    print("🎯 Finalizado. Dataset aumentado listo.")
