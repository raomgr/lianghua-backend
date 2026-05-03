from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

BACKEND_DIR = Path(__file__).resolve().parent


def main(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(BACKEND_DIR)] if reload else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the A-Share Quant backend server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local development.")
    args = parser.parse_args()
    main(host=args.host, port=args.port, reload=args.reload)
