from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "lsc-pose", "weights", "bestEsteban.pt")

if __name__ == "__main__":
    # cargar el modelo entrenado
    model = YOLO(MODEL_PATH)

    # exportar a onnx como pose
    model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False,
        device=0,    # usa 0 si tienes GPU, -1 para CPU
    )
