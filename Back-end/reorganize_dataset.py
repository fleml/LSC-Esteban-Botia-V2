import os
import shutil

# 🔹 ruta base del dataset
base_path = r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\data\dataset\images"

for split in ["train", "val"]:
    split_path = os.path.join(base_path, split)
    
    for side in ["derecha", "izquierda"]:
        side_path = os.path.join(split_path, side)
        if not os.path.exists(side_path):
            continue
        
        for cls in os.listdir(side_path):
            cls_path = os.path.join(side_path, cls)
            if not os.path.isdir(cls_path):
                continue
            
            dest_cls_path = os.path.join(split_path, cls)
            os.makedirs(dest_cls_path, exist_ok=True)
            
            for img_name in os.listdir(cls_path):
                src = os.path.join(cls_path, img_name)
                dest = os.path.join(dest_cls_path, img_name)
                # si ya existe no sobreescribe
                if not os.path.exists(dest):
                    shutil.move(src, dest)
                else:
                    print(f"⚠️ Imagen duplicada ignorada: {img_name}")
        
        # borrar carpeta side si quedó vacía
        shutil.rmtree(side_path, ignore_errors=True)

print("✅ Dataset reordenado correctamente, ahora solo quedan 3 clases por split.")
