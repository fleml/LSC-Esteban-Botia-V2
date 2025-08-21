import os
import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\Juan Esteban Botia\Desktop\PRACTICA - ADSO - 2025\LSC Esteban Botia V2\Back-end\runs\classify\train3\weights\best.pt"
IMG_SIZE = 224
CONF_THRESH = 0.05  # bajo para no perder A

app = FastAPI(title="API LSC Clasify", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")

model = YOLO(MODEL_PATH)
CLASS_NAMES = model.model.names
print(f"✅ Modelo cargado: {MODEL_PATH}")

def read_image_bytes_to_bgr(image_bytes: bytes):
    file = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode devolvió None")
    return img

@app.get("/")
def root():
    return {"ok": True, "classes": CLASS_NAMES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = read_image_bytes_to_bgr(image_bytes)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        results = model.predict(img, imgsz=IMG_SIZE, verbose=False)
        best_class, best_conf = None, 0.0

        # lógica igual para todas las letras A/B/C
        if results and hasattr(results[0], "probs") and results[0].probs is not None:
            probs = results[0].probs
            cls_id = probs.top1
            best_conf = float(probs.top1conf.item())
            if best_conf >= CONF_THRESH:
                best_class = CLASS_NAMES[cls_id]

        if best_class is None and hasattr(results[0], "boxes"):
            boxes = results[0].boxes
            if boxes.cls is not None and len(boxes.cls) > 0:
                cls_id = int(boxes.cls[0])
                best_conf = float(boxes.conf[0])
                if best_conf >= CONF_THRESH:
                    best_class = CLASS_NAMES[cls_id]

        return JSONResponse({"class_name": best_class, "confidence": best_conf})

    except Exception as e:
        print("❌ Error en /predict:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
