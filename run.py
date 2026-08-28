import os
import sys
import uvicorn

if __name__ == "__main__":
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
    sys.path.insert(0, backend_dir)
    uvicorn.run("main:app", app_dir=backend_dir, host="127.0.0.1", port=8000, reload=True)
