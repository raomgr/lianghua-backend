from __future__ import annotations

import argparse

from app.tasks.train_model import main as train_main
from app.tasks.update_data import main as sync_main
from run import main as run_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Management commands for the A-Share Quant backend.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start the backend API service.")
    run_parser.add_argument("--host", default="127.0.0.1")
    run_parser.add_argument("--port", type=int, default=8000)
    run_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local development.")

    subparsers.add_parser("sync", help="Sync market data into the local database.")
    subparsers.add_parser("train", help="Train the local baseline model and store predictions.")
    subparsers.add_parser("refresh", help="Run sync first, then train the local baseline model.")

    args = parser.parse_args()

    if args.command == "run":
        run_main(host=args.host, port=args.port, reload=args.reload)
        return

    if args.command == "sync":
        sync_main()
        return

    if args.command == "train":
        train_main()
        return

    if args.command == "refresh":
        sync_main()
        train_main()


if __name__ == "__main__":
    main()
