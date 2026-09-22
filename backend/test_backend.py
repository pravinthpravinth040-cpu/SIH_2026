import io
import json
from pathlib import Path
from fastapi.testclient import TestClient
from main import app
from database import get_db, init_db
from sample_data import seed_initial_datasets_if_empty

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_SAMPLE = BASE_DIR / "binary classification-20260916T134200Z-1-001" / "binary classification" / "model_service_handoff" / "sample_images" / "oil_sample_1.jpg"

client = TestClient(app)

def setup_db():
    init_db()
    db = next(get_db())
    seed_initial_datasets_if_empty(db)
    db.close()

def test_dashboard_statistics():
    print("Testing GET /api/dashboard/statistics...")
    response = client.get("/api/dashboard/statistics")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "online"
    assert data["total_spills_detected"] >= 4
    assert data["total_satellite_images"] >= 4
    assert "active_spill" in data
    assert data["active_spill"] is not None
    assert len(data["active_spill"]["nearby_vessels"]) > 0
    print("  -> Passed dashboard statistics test.")

def test_satellite_datasets():
    print("Testing GET /api/satellite/datasets...")
    response = client.get("/api/satellite/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert len(data["datasets"]) >= 4
    print("  -> Passed satellite datasets test.")

def test_ais_datasets():
    print("Testing GET /api/ais/datasets...")
    response = client.get("/api/ais/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert len(data["datasets"]) >= 1
    print("  -> Passed AIS datasets test.")

def test_spill_list_and_detail():
    print("Testing GET /api/spills and GET /api/spills/{id}...")
    response = client.get("/api/spills")
    assert response.status_code == 200
    spills = response.json()["spills"]
    assert len(spills) >= 4
    
    first_id = spills[0]["spill_id"]
    detail_res = client.get(f"/api/spills/{first_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["spill_id"] == first_id
    assert "polygon_geojson" in detail
    assert "weather_info" in detail
    assert len(detail["nearby_vessels"]) > 0
    print("  -> Passed spill list and detail test.")

def test_history_filtering():
    print("Testing GET /api/history with filters...")
    # Filter by spill_id
    response = client.get("/api/history?spill_id=OG-2026-0825-001")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    assert data["history"][0]["spill_id"] == "OG-2026-0825-001"

    # Filter by confidence
    response2 = client.get("/api/history?confidence_min=93.0")
    assert response2.status_code == 200
    for item in response2.json()["history"]:
        assert item["confidence"] >= 93.0
    print("  -> Passed history filtering test.")

def test_upload_satellite():
    print("Testing POST /api/upload/satellite...")
    import numpy as np
    import cv2
    img = np.full((150, 150), 180, dtype=np.uint8)
    cv2.circle(img, (75, 75), 25, 40, -1) # Dark spot
    _, encoded = cv2.imencode(".png", img)

    file_bytes = encoded.tobytes()
    files = {"file": ("test_sar.png", file_bytes, "image/png")}
    data = {"dataset_name": "Test Sentinel-1 SAR Upload", "lat": 12.5, "lon": 74.2}

    response = client.post("/api/upload/satellite", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert "spill_id" in res
    assert len(res["attributed_vessels"]) > 0
    print("  -> Passed satellite image upload and processing test.")

def test_upload_ais_csv():
    print("Testing POST /api/upload/ais...")
    csv_content = """MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName,VesselType
412999001,2026-08-25T11:00:00,11.24,72.46,14.0,45.0,45.0,MV TEST TANKER,Crude Oil Tanker
412999001,2026-08-25T11:15:00,11.23,72.45,13.8,45.0,45.0,MV TEST TANKER,Crude Oil Tanker
412999001,2026-08-25T11:30:00,11.22,72.44,14.2,45.0,45.0,MV TEST TANKER,Crude Oil Tanker
"""
    files = {"file": ("test_ais.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    data = {"dataset_name": "Test MarineCadastre Ingestion"}

    response = client.post("/api/upload/ais", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["record_count"] == 3
    print("  -> Passed AIS CSV ingestion test.")

def test_health_and_detection_routes():
    print("Testing /api/health and detection endpoints...")
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    assert MODEL_SAMPLE.exists(), f"Missing model sample image: {MODEL_SAMPLE}"
    with MODEL_SAMPLE.open("rb") as f:
        classify_response = client.post("/api/detection/classify", files={"file": (MODEL_SAMPLE.name, f.read(), "image/jpeg")})
    assert classify_response.status_code == 200, classify_response.text
    classify_data = classify_response.json()
    assert "is_oil_spill" in classify_data
    assert "confidence" in classify_data

    with MODEL_SAMPLE.open("rb") as f:
        segment_response = client.post("/api/detection/segment", files={"file": (MODEL_SAMPLE.name, f.read(), "image/jpeg")})
    assert segment_response.status_code == 200, segment_response.text
    segment_data = segment_response.json()
    assert "mask_url" in segment_data or "warning" in segment_data

    with MODEL_SAMPLE.open("rb") as f:
        run_response = client.post("/api/detection/run", files={"file": (MODEL_SAMPLE.name, f.read(), "image/jpeg")})
    assert run_response.status_code == 200, run_response.text
    run_data = run_response.json()
    assert run_data["status"] in {"completed", "processed", "completed_with_warnings"}
    assert "classification" in run_data
    print("  -> Passed health and detection pipeline routes test.")

def test_api_gateway_and_v1_endpoints():
    print("Testing API Gateway /api/v1/ endpoints...")
    # Test v1 spills
    resp = client.get("/api/v1/spills", headers={"X-API-Key": "XzQgiffxY0uAsFgL6df2fcgerAmdEqg8VHafPr0U"})
    assert resp.status_code == 200
    assert "history" in resp.json()

    # Test v1 ports
    resp_ports = client.get("/api/v1/ports")
    assert resp_ports.status_code == 200
    assert "ports" in resp_ports.json()

    # Test v1 alerts
    resp_alerts = client.get("/api/v1/alerts")
    assert resp_alerts.status_code == 200
    assert "alerts" in resp_alerts.json()
    print("  -> Passed API Gateway & /api/v1/ endpoints test.")

def test_webhooks_and_orchestrator():
    print("Testing Webhooks Ingress & 13-Agent Orchestrator Pipeline...")
    # 1. Satellite Webhook
    sat_payload = {
        "event": "new_satellite_scene",
        "timestamp": "2026-08-28T10:30:00Z",
        "sensor": "Sentinel-1 SAR",
        "area": {"min_lon": 72.5, "min_lat": 18.5, "max_lon": 73.5, "max_lat": 19.5},
        "scene_id": "SCENE_TEST_001"
    }
    resp_sat = client.post("/api/v1/webhooks/satellite", json=sat_payload)
    assert resp_sat.status_code == 200
    sat_data = resp_sat.json()
    assert sat_data["status"] == "ACCEPTED_AND_PROCESSED"
    assert "pipeline_job_id" in sat_data
    assert "estimated_origin" in sat_data

    # 2. AIS Webhook
    ais_payload = {"event": "ais_stream", "vessels": [{"mmsi": "538008123", "lat": 18.9, "lon": 72.8}]}
    resp_ais = client.post("/api/v1/webhooks/ais", json=ais_payload)
    assert resp_ais.status_code == 200
    assert resp_ais.json()["status"] == "ACCEPTED"

    # 3. Orchestrator logs
    resp_logs = client.get("/api/v1/orchestrator/logs")
    assert resp_logs.status_code == 200
    assert resp_logs.json()["total_logs"] > 0
    print("  -> Passed Webhooks Ingress & 13-Agent Orchestrator test.")

if __name__ == "__main__":
    setup_db()
    print("\n=======================================================")
    print("  RUNNING COMPLETE BACKEND TEST SUITE")
    print("=======================================================\n")
    test_dashboard_statistics()
    test_satellite_datasets()
    test_ais_datasets()
    test_spill_list_and_detail()
    test_history_filtering()
    test_upload_satellite()
    test_upload_ais_csv()
    test_health_and_detection_routes()
    test_api_gateway_and_v1_endpoints()
    test_webhooks_and_orchestrator()
    print("\n>>> ALL 10 BACKEND & ARCHITECTURE TEST CASES PASSED PERFECTLY! <<<\n")

