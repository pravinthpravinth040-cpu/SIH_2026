# Oil Spill Classification Model Service Handoff

Standalone, lightweight FastAPI backend deployment package for Sentinel-1 SAR Oil Spill Classification.

## Quick Start Instructions

### 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run API Server
Start the FastAPI server using Uvicorn:
```bash
python app.py
```
The server will start at `http://localhost:8000`. Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### 3. Run Automated Handoff Tests
Run `test_service.py` to launch an in-process test server, send sample image payloads, and verify API responses:
```bash
python test_service.py
```

---

## Directory Layout
```
model_service_handoff/
├── app.py                 # FastAPI service endpoints
├── inference.py           # Core PyTorch inference engine
├── model.py               # ResNet-18 model architecture
├── requirements.txt       # Trimmed dependency specification
├── API_CONTRACT.md        # API request/response specification
├── README.md              # Deployment guide
├── test_service.py        # Automated service integration test script
├── checkpoints/           # Trained weights
│   └── best_model.pth     # ResNet-18 binary classification weights
└── sample_images/         # Test sample images
    ├── sample_no_oil_1.jpg
    ├── sample_no_oil_2.jpg
    └── sample_oil_1.jpg
```
