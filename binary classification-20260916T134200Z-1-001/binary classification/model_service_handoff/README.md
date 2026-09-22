# Oil Spill Classification Model Microservice

Clean, self-contained microservice for binary oil spill detection in Sentinel-1 SAR satellite images.

---

## Service Overview
- **Model**: ResNet18 Binary Classifier
- **Framework**: PyTorch + FastAPI + Uvicorn
- **Default Port**: `8000`

---

## Quick Start (Setup & Run in 3 Commands)

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Venv & Install Dependencies
#### On Windows (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
#### On Linux / macOS:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start Service (One Command)
```bash
python app.py
```
*The service is now running on `http://localhost:8000`.*

---

## Automated Verification Test

Once the service is started, run the automated test suite in another terminal:
```bash
python test_service.py
```
*Outputs `ALL 3 TESTS PASSED SUCCESSFULLY!` upon clean verification.*

---

## API Endpoints

### 1. `GET /health`
- **Purpose**: Service health check
- **Response**: `200 OK` -> `{"status": "healthy", "service": "oil-spill-detector"}`

### 2. `POST /predict`
- **Purpose**: Classify an uploaded SAR image patch
- **Content-Type**: `multipart/form-data`
- **Form Field**: `file` (Image file `.jpg`, `.png`, `.tif`)
- **Response**:
```json
{
  "filename": "oil_sample_1.jpg",
  "oil_detected": true,
  "confidence": 1.0,
  "raw_score": 1.0
}
```

---

## Tested Curl Commands

### Health Check:
```bash
curl -X GET http://localhost:8000/health
```

### Predict Sample Image:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample_images/oil_sample_1.jpg"
```

---

## Directory Structure
```
model_service_handoff/
├── app.py                  # FastAPI REST web server
├── inference.py            # Preprocessing & PyTorch model loader/predictor
├── model.py                # ResNet18 binary classifier definition
├── requirements.txt        # Pinned dependencies for serving
├── test_service.py         # Automated verification script
├── API_CONTRACT.md         # Full API specification & schemas
├── README.md               # Quickstart guide
├── checkpoints/
│   └── best_model.pth      # Trained PyTorch model weights (~44MB)
└── sample_images/
    ├── oil_sample_1.jpg    # Sample test image (Known Oil Spill)
    └── clean_sample_1.jpg  # Sample test image (Known Clean Sea)
```
