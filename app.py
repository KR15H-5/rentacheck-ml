from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import numpy as np
import cv2
import os
import torch

app = FastAPI()

# Security fix for PyTorch 2.6+
torch.serialization.add_safe_globals([YOLO])

# Load model
model_path = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(model_path).to('cpu')

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Only images allowed")
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = model(img, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })
        
        return {"detections": detections}

    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")