import os
from ultralytics import YOLO

# -----------------------
# Rutas según tu estructura
# -----------------------
# ruta original de tu best.pt
model_path = r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\runs\pose\train16\weights\best.pt"

# carpeta de salida para ONNX
onnx_dir = os.path.join(os.path.dirname(__file__), "onnx")
os.makedirs(onnx_dir, exist_ok=True)
onnx_path = os.path.join(onnx_dir, "bestEsteban.onnx")

# -----------------------
# Cargar modelo YOLO
# -----------------------
try:
    model = YOLO(model_path)
    print(f"✅ modelo cargado correctamente desde:\n{model_path}")
except Exception as e:
    raise RuntimeError(f"❌ error cargando el modelo: {e}")

# -----------------------
# Exportar a ONNX
# -----------------------
try:
    model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False,
        device="cpu",         # 👈 forzamos a CPU porque no tienes CUDA
        project=onnx_dir,     # manda el archivo a la carpeta "onnx"
        name="",              # evita crear subcarpeta extra
    )
    print(f"✅ modelo exportado a ONNX correctamente en:\n{onnx_path}")
except Exception as e:
    raise RuntimeError(f"❌ error exportando a ONNX: {e}")
