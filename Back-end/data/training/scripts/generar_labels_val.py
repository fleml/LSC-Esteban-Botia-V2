import cv2
import os
import numpy as np

# Rutas reales del dataset para VAL
images_root = r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\images\val"
labels_root = r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\labels\val"

# Mapeo de carpetas a class_id
class_map = {
    "A_DERECHA": 0,
    "B_DERECHA": 1,
    "C_DERECHA": 2,
    "A_IZQUIERDA": 3,
    "B_IZQUIERDA": 4,
    "C_IZQUIERDA": 5
}

def interpolate_points(points, target_count):
    """Interpola puntos para alcanzar el número objetivo."""
    if len(points) == target_count:
        return points
    interpolated = []
    for i in range(len(points)):
        interpolated.append(points[i])
        next_idx = (i + 1) % len(points)
        if len(interpolated) < target_count:
            mid_x = (points[i][0] + points[next_idx][0]) / 2
            mid_y = (points[i][1] + points[next_idx][1]) / 2
            interpolated.append((mid_x, mid_y))
        if len(interpolated) >= target_count:
            break
    return interpolated[:target_count]

# Procesar todas las imágenes de VAL
for subdir, _, files in os.walk(images_root):
    for file in files:
        if not file.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(subdir, file)
        rel_path = os.path.relpath(img_path, images_root)
        parts = rel_path.split(os.sep)

        if len(parts) < 3:
            continue

        mano = parts[0].strip().upper()        # derecha / izquierda
        letra = parts[1].strip().upper()       # A / B / C
        class_name = f"{letra}_{mano}"

        class_id = class_map.get(class_name)
        if class_id is None:
            print(f"⚠️ Clase no reconocida: {class_name}")
            continue

        label_path = os.path.join(labels_root, mano.lower(), letra, os.path.splitext(file)[0] + ".txt")

        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ No se pudo leer: {img_path}")
            continue

        h, w, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"⚠️ No se detectó mano: {img_path}")
            continue

        c = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(c, returnPoints=True)

        # Eliminar puntos duplicados del hull
        unique_points = []
        for p in hull[:, 0, :]:
            pt = tuple(p)
            if pt not in unique_points:
                unique_points.append(pt)

        # Asegurar 21 puntos exactos
        if len(unique_points) > 21:
            indices = np.linspace(0, len(unique_points) - 1, 21, dtype=int)
            sampled_points = [unique_points[i] for i in indices]
        elif len(unique_points) < 21:
            sampled_points = interpolate_points(unique_points, 21)
        else:
            sampled_points = unique_points

        # Normalizar y guardar
        keypoints = []
        for x, y in sampled_points:
            keypoints.extend([x / w, y / h, 1])

        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w") as f:
            f.write(f"{class_id} " + " ".join(map(str, keypoints)) + "\n")

        print(f"✅ Label generado: {label_path}")

print("\n🎯 Proceso finalizado: Todos los labels de VAL fueron generados con 21 keypoints únicos")
