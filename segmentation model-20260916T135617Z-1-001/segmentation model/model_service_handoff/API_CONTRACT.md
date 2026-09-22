# API Contract: Sentinel-1 SAR Oil Spill Classification Service

## Overview
This REST API service processes uploaded SAR satellite images or patches and performs binary classification to detect the presence of oil spills.

- **Base URL**: `http://localhost:8000`
- **Protocol**: HTTP/REST
- **Content-Type**: `multipart/form-data` for file uploads, `application/json` for responses

---

## Endpoints

### 1. Health Check
`GET /`

#### Response
- **Status Code**: `200 OK`
- **Body**:
```json
{
  "status": "ok",
  "message": "Oil Spill Detection API is running. Use POST /predict to upload an image."
}
```

---

### 2. Predict Oil Spill
`POST /predict`

#### Request
- **Headers**: `Content-Type: multipart/form-data`
- **Form Data**:
  - `file`: Image file (`.jpg`, `.png`, `.bmp`, `.tif`, `.tiff`)

#### Success Response (`200 OK`)
```json
{
  "filename": "sample_oil_1.jpg",
  "oil_detected": true,
  "confidence": 0.9845,
  "raw_score": 0.9845
}
```

#### Field Descriptions
| Field | Type | Description |
| :--- | :--- | :--- |
| `filename` | string | Name of uploaded image file |
| `oil_detected` | boolean | `true` if oil spill is detected ($\text{probability} \ge 0.5$), `false` otherwise |
| `confidence` | float | Classification confidence score $[0.5, 1.0]$ |
| `raw_score` | float | Raw sigmoid output probability $[0.0, 1.0]$ |

#### Error Responses
- **`400 Bad Request`**: Uploaded file is not an image or unsupported format.
```json
{
  "detail": "Uploaded file must be an image."
}
```
- **`500 Internal Server Error`**: Inference engine processing failure.
```json
{
  "detail": "Inference error: <description>"
}
```
