import os
import sys
import time
from pathlib import Path
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")

def run_tests():
    print("=" * 60)
    print("OIL SPILL MODEL SERVICE - AUTOMATED HANDOFF TEST SUITE")
    print("=" * 60)
    print(f"Target API Server: {API_URL}\n")
    
    passed_tests = 0
    total_tests = 3
    
    # Test 1: Health Check Endpoint
    print("[TEST 1/3] Testing GET /health...")
    try:
        r_health = requests.get(f"{API_URL}/health", timeout=5)
        if r_health.status_code == 200 and r_health.json().get("status") == "healthy":
            print(f"  --> PASS: Server is healthy ({r_health.json()})")
            passed_tests += 1
        else:
            print(f"  --> FAIL: Health check returned HTTP {r_health.status_code}: {r_health.text}")
    except Exception as e:
        print(f"  --> FAIL: Could not connect to {API_URL}/health. Error: {e}")
        print("      Make sure app.py is running on port 8000!")
        
    base_dir = Path(__file__).parent
    
    # Test 2: Predict Endpoint on Known Oil Sample Image
    print("\n[TEST 2/3] Testing POST /predict with known Oil Spill image...")
    oil_img_path = base_dir / "sample_images" / "oil_sample_1.jpg"
    if not oil_img_path.exists():
        print(f"  --> FAIL: Sample image file missing at {oil_img_path}")
    else:
        try:
            with open(oil_img_path, "rb") as f:
                files = {"file": ("oil_sample_1.jpg", f, "image/jpeg")}
                r_oil = requests.post(f"{API_URL}/predict", files=files, timeout=5)
                
            if r_oil.status_code == 200:
                data = r_oil.json()
                print(f"  --> Response: {data}")
                if data.get("oil_detected") is True:
                    print(f"  --> PASS: Correctly identified Oil Spill (Confidence: {data.get('confidence')*100:.2f}%)")
                    passed_tests += 1
                else:
                    print(f"  --> WARNING/FAIL: Expected oil_detected=True, got {data.get('oil_detected')}")
            else:
                print(f"  --> FAIL: Endpoint returned HTTP {r_oil.status_code}: {r_oil.text}")
        except Exception as e:
            print(f"  --> FAIL: Exception calling /predict: {e}")
            
    # Test 3: Predict Endpoint on Known Clean (No-Oil) Sample Image
    print("\n[TEST 3/3] Testing POST /predict with known Clean (No-Oil) image...")
    clean_img_path = base_dir / "sample_images" / "clean_sample_1.jpg"
    if not clean_img_path.exists():
        print(f"  --> FAIL: Sample image file missing at {clean_img_path}")
    else:
        try:
            with open(clean_img_path, "rb") as f:
                files = {"file": ("clean_sample_1.jpg", f, "image/jpeg")}
                r_clean = requests.post(f"{API_URL}/predict", files=files, timeout=5)
                
            if r_clean.status_code == 200:
                data = r_clean.json()
                print(f"  --> Response: {data}")
                if data.get("oil_detected") is False:
                    print(f"  --> PASS: Correctly identified Clean / No-Oil (Confidence: {data.get('confidence')*100:.2f}%)")
                    passed_tests += 1
                else:
                    print(f"  --> WARNING/FAIL: Expected oil_detected=False, got {data.get('oil_detected')}")
            else:
                print(f"  --> FAIL: Endpoint returned HTTP {r_clean.status_code}: {r_clean.text}")
        except Exception as e:
            print(f"  --> FAIL: Exception calling /predict: {e}")
            
    print("\n" + "=" * 60)
    if passed_tests == total_tests:
        print(f"ALL {total_tests} TESTS PASSED SUCCESSFULLY! Service is fully operational.")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"TEST SUITE FAILED: Passed {passed_tests}/{total_tests} tests.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
