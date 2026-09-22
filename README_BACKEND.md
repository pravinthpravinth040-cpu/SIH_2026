# Backend Setup and Configuration

## 1. Installation

```bash
cd backend
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Environment variables

Copy the example file and fill in actual values:

```bash
copy ..\.env.example ..\.env
```

Required backend values:

- `SUPABASE_URL`: full project URL, such as `https://<project-ref>.supabase.co`
- `SUPABASE_SERVICE_ROLE_KEY`: service role key from Supabase project settings
- `SUPABASE_ANON_KEY`: public key for browser-only access if used directly from frontend
- `DATABASE_URL`: local SQLite or hosted database connection string

> The host value `db.iuwwzrgvjywbukgugmsq.supabase.co` is a Postgres hostname used for direct database access only. It must not be exposed in the browser or frontend JavaScript.

## 3. Supabase setup

1. Create a Supabase project.
2. Copy the Project URL and the service role key from the dashboard.
3. Add them to `.env` only on the backend server.
4. Run the schema migration in `supabase_schema.sql`.
5. Create buckets:
   - `satellite-images`
   - `segmentation-masks`
   - `detection-overlays`

## 4. Database migration

```bash
psql "$DATABASE_URL" -f supabase_schema.sql
```

If you are using the Supabase SQL editor, paste the contents of `supabase_schema.sql` and run it.

## 5. Model setup

The binary classification model is already provided in the model handoff folder:

- `binary classification-20260916T134200Z-1-001/binary classification/model_service_handoff/checkpoints/best_model.pth`

The segmentation folder already contains the corresponding service handoff and weights:

- `segmentation model-20260916T135617Z-1-001/segmentation model/model_service_handoff/checkpoints/best_model.pth`

The backend loads these inference modules from disk and keeps them in memory once at startup. The code also supports CUDA when available and falls back to CPU otherwise.

## 6. Local development

```bash
cd backend
python main.py
```

Or use Uvicorn:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs:

- http://localhost:8000/docs
- http://localhost:8000/redoc

## 7. Production deployment

Deploy the backend to a cloud runtime that supports Python/FastAPI, then keep the frontend on GitHub Pages and point it to the backend URL with `API_BASE_URL`.

Example:

```js
const API_BASE_URL = "https://your-backend-domain.example.com";
```

## 8. API endpoints

- `GET /api/health`
- `POST /api/detection/classify`
- `POST /api/detection/segment`
- `POST /api/detection/run`
- `GET /api/detections`
- `GET /api/detections/{id}`
- `GET /api/images`
- `GET /api/processing-jobs`
- `GET /api/vessels`
- `GET /api/vessel-correlations/{detection_id}`
- `DELETE /api/detections/{id}`

## 9. Frontend configuration

The browser should use the backend over HTTPS. GitHub Pages is static hosting and should not host the Python service directly.

```js
const API_BASE_URL = "http://localhost:8000"; // local dev
// or
const API_BASE_URL = "https://your-backend-domain.example.com"; // production
```

Keep the service-role key out of the browser at all times.

## 10. External API integration

External API calls are made only by `backend/api_service.py`. Configure the
provider URL and secret in the backend `.env` file:

```env
EXTERNAL_API_URL=https://provider.example/api/data
EXTERNAL_API_KEY=your_api_key_here
EXTERNAL_API_AUTH_HEADER=Authorization
EXTERNAL_API_AUTH_SCHEME=Bearer
EXTERNAL_API_TIMEOUT_SECONDS=10
```

The adapter validates HTTP status, timeout, JSON shape, and never includes the
credential in responses or logs. External API failure is recorded as
`status: unavailable` while ML detection continues. Successful responses are
cached for ten minutes per coordinate in `external_api_data`.

Additional endpoints:

- `GET /api/detections`
- `GET /api/detections/{id}`
- `GET /api/external-data`
- `GET /api/external-data/{id}`

The exact provider URL, authentication header, and response schema must be
filled in from the external API documentation before live external data can be
retrieved. The current adapter stores the provider JSON in `data` without
inventing fields.
