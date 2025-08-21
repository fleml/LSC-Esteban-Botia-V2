// src/App.jsx
import React from "react";
import VideoCapture from "./components/VideoCapture";

export default function App() {
  return (
    <div style={{ padding: 16 }}>
      <h2 style={{ marginBottom: 8 }}>lsc — demo en vivo (onnxruntime-web, yolov8-pose)</h2>
      <VideoCapture />
    </div>
  );
}
