import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# cargar modelo YOLO keypoints
model = YOLO("app/weights/yolov8n-pose.pt")

def run_inference(image_bytes):
    # escribir la imagen temporalmente
    with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        # leer con opencv
        img = cv2.imread(tmp.name)
    
    # hacer inferencia
    results = model.predict(img, imgsz=640, conf=0.5)
    
    # parsear resultado a algo sencillo
    output = []
    for r in results:
        for kps in r.keypoints:
            output.append(kps.tolist())  # lista de keypoints
    
    return output
