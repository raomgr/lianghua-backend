from pathlib import Path

import uvicorn

BACKEND_DIR = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
