import React, { useEffect, useRef, useState } from "react";

const API_URL = "http://localhost:8000/predict";
const SEND_EVERY_MS = 150; // más rápido para no perder movimientos

export default function VideoCapture() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const offscreenRef = useRef(null);
    const sendingRef = useRef(false);
    const [status, setStatus] = useState("inicializando cámara...");
    const lastDetectedRef = useRef({ class_name: null, timestamp: 0 });
    const detectedHistory = useRef([]);

    useEffect(() => {
        (async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "user" },
                    audio: false,
                });
                videoRef.current.srcObject = stream;
                videoRef.current.onloadedmetadata = async () => {
                    await videoRef.current.play();
                    const off = document.createElement("canvas");
                    off.width = videoRef.current.videoWidth;
                    off.height = videoRef.current.videoHeight;
                    offscreenRef.current = off;
                    setStatus("cámara lista ✨");
                    startLoops();
                };
            } catch (e) {
                console.error(e);
                setStatus("no se pudo abrir la cámara 😢");
            }
        })();

        return () => {
            const s = videoRef.current?.srcObject;
            if (s) s.getTracks().forEach((t) => t.stop());
        };
    }, []);

    const startLoops = () => {
        const drawLoop = () => {
            drawFrame();
            requestAnimationFrame(drawLoop);
        };
        requestAnimationFrame(drawLoop);

        setInterval(() => {
            if (!sendingRef.current) sendFrame();
        }, SEND_EVERY_MS);
    };

    const captureFrameBlob = async () => {
        const video = videoRef.current;
        const off = offscreenRef.current;
        if (!video || !off) return null;
        const octx = off.getContext("2d");
        octx.drawImage(video, 0, 0, off.width, off.height);
        return await new Promise((resolve) => off.toBlob(resolve, "image/jpeg", 1.0));
    };

    const sendFrame = async () => {
        try {
            const blob = await captureFrameBlob();
            if (!blob) return;

            sendingRef.current = true;
            setStatus("analizando…");

            const form = new FormData();
            form.append("file", blob, "frame.jpg");

            const res = await fetch(API_URL, { method: "POST", body: form });
            const json = await res.json();
            const now = Date.now();

            if (json.error) {
                console.error("❌ /predict error:", json.error);
                setStatus("error en /predict");
            } else if (json.class_name) {
                // guardar en historial para suavizado
                detectedHistory.current.push(json.class_name);
                if (detectedHistory.current.length > 5) detectedHistory.current.shift();

                // mayoría simple (para todas las letras A/B/C)
                const counts = {};
                detectedHistory.current.forEach(c => counts[c] = (counts[c]||0)+1);
                const majorityClass = Object.entries(counts).sort((a,b)=>b[1]-a[1])[0][0];

                lastDetectedRef.current = { class_name: majorityClass, timestamp: now };
                setStatus(`letra detectada: ${majorityClass} (${(json.confidence*100).toFixed(1)}%)`);
            } else {
                // mantener última detección 1s
                if (now - lastDetectedRef.current.timestamp < 1000 && lastDetectedRef.current.class_name) {
                    setStatus(`letra detectada: ${lastDetectedRef.current.class_name}`);
                } else {
                    setStatus("esperando seña…");
                }
            }

        } catch (e) {
            console.error("❌ error /predict:", e);
            setStatus("error en /predict");
        } finally {
            sendingRef.current = false;
        }
    };

    const drawFrame = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    };

    return (
        <div style={{ position: "relative", width: "fit-content" }}>
            <video ref={videoRef} autoPlay playsInline muted style={{ display: "none" }} />
            <canvas ref={canvasRef} width={640} height={480} style={{ border: "1px solid #222", borderRadius: 8 }} />
            <div style={{ marginTop: 8, fontFamily: "monospace", color: "#888" }}>
                {status}
            </div>
        </div>
    );
}
