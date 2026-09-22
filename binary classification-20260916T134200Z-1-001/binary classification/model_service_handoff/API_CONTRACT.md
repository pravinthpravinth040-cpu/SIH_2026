# API Contract: Sentinel-1 SAR Oil Spill Classification Service

This document defines the HTTP API specification for integrating the oil spill detection machine learning model into your backend services.

## Overview
- **Service Name**: Oil Spill Detector API
- **Protocol**: HTTP/REST
- **Default Base URL**: `http://localhost:8000`
- **Data Transport**: `multipart/form-data` for image uploads, `application/json` for responses.

---

## Endpoints

### 1. Health Check
Checks if the model service is online and ready for inference.

- **Method**: `GET`
- **Path**: `/health`
- **Headers**: None
- **Response**: `200 OK`

#### Response Schema (`application/json`)
```json
{
  "status": "healthy",
  "service": "oil-spill-detector"
}
```

---

### 2. Predict Oil Spill
Accepts a binary Sentinel-1 SAR satellite patch image and returns whether an oil spill was detected, along with a confidence percentage.

- **Method**: `POST`
- **Path**: `/predict`
- **Content-Type**: `multipart/form-data`
- **Form Parameters**:
  - `file` *(required)*: Binary image file (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`).

#### Response Schema (`application/json`)
```json
{
  "filename": "string",
  "oil_detected": "boolean",
  "confidence": "float (0.0 to 1.0)",
  "raw_score": "float (0.0 to 1.0)"
}
```

#### Field Descriptions
| Field | Type | Description |
| :--- | :--- | :--- |
| `filename` | String | Name of the uploaded file. |
| `oil_detected` | Boolean | `true` if oil spill is detected, `false` if clean sea surface. |
| `confidence` | Float | Model certainty percentage (e.g. `0.9953` = 99.53% confident). |
| `raw_score` | Float | Raw sigmoid output probability (`>= 0.5` triggers `oil_detected = true`). |

#### Example Response (Oil Spill Detected)
```json
{
  "filename": "oil_sample_1.jpg",
  "oil_detected": true,
  "confidence": 1.0,
  "raw_score": 1.0
}
```

#### Example Response (Clean / No Oil)
```json
{
  "filename": "clean_sample_1.jpg",
  "oil_detected": false,
  "confidence": 1.0,
  "raw_score": 0.0
}
```

---

## HTTP Status Codes

| Code | Status | Meaning |
| :--- | :--- | :--- |
| `200` | OK | Successful inference. |
| `400` | Bad Request | Uploaded file is empty or not a valid image format. |
| `500` | Internal Error | Exception occurred during PyTorch inference. |

---

## Curl Request Examples

### Health Check:
```bash
curl -X GET http://localhost:8000/health
```

### Predict Image:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_images/oil_sample_1.jpg"
```
