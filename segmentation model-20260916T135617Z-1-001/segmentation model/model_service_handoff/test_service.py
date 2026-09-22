import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

from app import app
from inference import predict

def run_service_tests():
    print("=" * 70, flush=True)
    print("STARTING MODEL SERVICE HANDOFF INTEGRATION TESTS", flush=True)
    print("=" * 70, flush=True)
    
    base_dir = Path(__file__).parent
    sample_dir = base_dir / "sample_images"
    sample_images = sorted(list(sample_dir.glob("*.jpg")))
    
    if not sample_images:
        print("[FAIL] No sample images found in sample_images/ directory.", flush=True)
        sys.exit(1)
        
    print(f"Found {len(sample_images)} sample images for testing.\n", flush=True)
    
    # 1. Direct Python Inference Test
    print("--- 1. Testing Core Python Inference (inference.py) ---", flush=True)
    for img_path in sample_images:
        res = predict(img_path)
        print(f"File: {img_path.name:<22} | Oil Detected: {str(res['oil_detected']):<5} | Confidence: {res['confidence']:.4f} | Raw Score: {res['raw_score']:.4f}", flush=True)
        assert "oil_detected" in res and "confidence" in res and "raw_score" in res, "Missing required keys in predict response!"
    print("[PASS] Direct Python inference test succeeded.\n", flush=True)
    
    # 2. FastAPI Endpoint Integration Test
    print("--- 2. Testing FastAPI REST Endpoint (POST /predict) ---", flush=True)
    client = TestClient(app)
    
    # Test Root Health Endpoint
    root_res = client.get("/")
    assert root_res.status_code == 200, f"Root health check failed with status {root_res.status_code}"
    print(f"Health Check GET / -> {root_res.json()}")
    
    # Test Upload Endpoints
    for img_path in sample_images:
        with open(img_path, "rb") as f:
            response = client.post("/predict", files={"file": (img_path.name, f, "image/jpeg")})
            
        assert response.status_code == 200, f"Predict endpoint failed for {img_path.name} with status {response.status_code}: {response.text}"
        data = response.json()
        print(f"POST /predict [{data['filename']}]: Status 200 | Oil Detected: {data['oil_detected']} | Confidence: {data['confidence']:.4f}")
        
        assert "oil_detected" in data
        assert "confidence" in data
        assert "raw_score" in data
        assert isinstance(data["oil_detected"], bool)
        assert 0.0 <= data["confidence"] <= 1.0
        
    print("[PASS] FastAPI REST Endpoint integration test succeeded.\n")
    
    print("=" * 70)
    print("ALL SERVICE HANDOFF TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_service_tests()
