from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import cv2
import numpy as np
import logging

app = FastAPI()
model = YOLO("/app/best.pt")  # Load model

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        # Verify image
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Only images allowed")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run detection
        results = model(img, verbose=False)
        
        # Format response
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()  # [x1,y1,x2,y2]
                })
        
        return {"detections": detections}

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(500, "Processing failed")