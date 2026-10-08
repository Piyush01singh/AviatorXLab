"""Entry point to launch the Aviator ML Research Dashboard.

Usage:
    python run_app.py [--host HOST] [--port PORT] [--reload]
"""

import argparse
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Aviator ML Research Web Application")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable automatic code reload")
    args = parser.parse_args()

    print("=" * 70)
    print("✈️  AVIATOR ML RESEARCH DASHBOARD")
    print(f"Server starting at: http://{args.host}:{args.port}")
    print("=" * 70)

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
