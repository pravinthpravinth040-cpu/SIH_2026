from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from inference import predict

app = FastAPI(
    title="Sentinel-1 SAR Oil Spill Classifier API",
    description="Binary image classifier API for detecting oil spills in SAR images",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Oil Spill Detection API is running. Use POST /predict to upload an image."}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        allowed_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        if file.filename and not any(file.filename.lower().endswith(ext) for ext in allowed_exts):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
            
    contents = await file.read()
    try:
        result = predict(contents)
        return JSONResponse(content={
            "filename": file.filename,
            "oil_detected": result["oil_detected"],
            "confidence": round(result["confidence"], 4),
            "raw_score": round(result["raw_score"], 4)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
