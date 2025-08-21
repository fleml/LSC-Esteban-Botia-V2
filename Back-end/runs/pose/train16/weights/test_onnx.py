import onnxruntime as ort
import numpy as np

# ruta al modelo exportado
onnx_path = "best.onnx"

# crear sesión
session = ort.InferenceSession(onnx_path)

# ver nombres de entrada y salida
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"✅ input: {input_name}, output: {output_name}")

# dummy input (imagen simulada 640x640)
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

# correr inferencia
outputs = session.run([output_name], {input_name: dummy_input})
print("✅ inferencia lista, shape salida:", outputs[0].shape)
