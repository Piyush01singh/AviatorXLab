"""FastAPI application serving the Aviator ML Research Live Dashboard & CSV API."""

import io
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

from src.api.inference import (
    load_benchmark_comparison_table,
    process_round_records,
    registry,
)
from src.collector.live_manager import live_manager
from src.data.loader import load_csv
from src.utils.logging import get_logger

logger = get_logger("api_server")

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm model registry and manage background workers."""
    try:
        registry.load_all_models()
        logger.info("Server startup: ML models ready.")
    except Exception as e:
        logger.warning(f"Could not pre-warm models: {e}")
    yield
    try:
        await live_manager.stop_current_stream()
        logger.info("Server shutdown: Live streams stopped.")
    except Exception as e:
        logger.warning(f"Error stopping live stream on shutdown: {e}")


app = FastAPI(
    title="Aviator ML Research API",
    description="Empirical machine learning and predictability research platform for crash games",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    """Enforce no-cache headers for static files and frontend index during active research."""
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path in ("/", "/bridge", "/bridge.html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/api/health")
async def health_check() -> JSONResponse:
    """Check API and model status."""
    return JSONResponse(
        content={
            "status": "ready",
            "models_loaded": registry._is_loaded,
            "models_count": 6,
            "live_connected": live_manager.is_connected,
            "live_stage": live_manager.current_stage,
            "bridge_connected": live_manager.bridge_connected,
        }
    )


# ==============================================================================
# LIVE TELEMETRY & WEBSOCKET ENDPOINTS
# ==============================================================================
@app.get("/api/live/state")
async def get_live_state() -> JSONResponse:
    """Get the latest real-time Aviator research snapshot."""
    snapshot = live_manager.get_state_snapshot()
    return JSONResponse(content=snapshot)


@app.post("/api/live/connect")
async def connect_live(request: Request) -> JSONResponse:
    """Start or attach to a live Aviator telemetry source.

    Request body (optional):
        source: 'sample_stream' | 'websocket' | 'browser_observer' (default: 'sample_stream')
        url: WebSocket URL if source == 'websocket'
        interval: seconds per round in replay mode (default: 3.0)
    """
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    source = body.get("source", "browser_observer")
    interval = float(body.get("interval", 3.0))
    ws_url = body.get("url")

    if source == "sample_stream":
        await live_manager.start_sample_stream(interval_sec=interval)
    elif source == "websocket":
        if not ws_url:
            raise HTTPException(status_code=400, detail="Missing 'url' parameter for websocket source.")
        await live_manager.connect_external_websocket(ws_url)
    elif source == "browser_observer":
        await live_manager.stop_current_stream()
        live_manager._user_disconnected = False
        live_manager.is_connected = True
        live_manager.source_type = "browser_observer"
        await live_manager.broadcast_state()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported source: {source}")

    return JSONResponse(content=live_manager.get_state_snapshot())


@app.post("/api/live/disconnect")
async def disconnect_live() -> JSONResponse:
    """Disconnect active live stream."""
    await live_manager.stop_current_stream()
    return JSONResponse(content=live_manager.get_state_snapshot())


@app.get("/api/live/observer.js")
async def get_observer_script() -> FileResponse:
    """Serve the read-only browser observer snippet for easy copy and DevTools execution."""
    script_path = Path(__file__).parent.parent / "collector" / "observer_snippet.js"
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Observer script not found.")
    return FileResponse(
        str(script_path),
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.post("/api/live/ingest")
async def ingest_live_event(request: Request) -> JSONResponse:
    """Ingest raw live telemetry frames from the browser observer snippet."""
    try:
        body = await request.json()
        raw_frame = body.get("raw_frame", body)
        await live_manager.ingest_raw(raw_frame)
        return JSONResponse(content={"status": "ingested"})
    except Exception as e:
        logger.error(f"Error ingesting raw frame: {e}")
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})


@app.websocket("/api/live/ws")
async def websocket_live_stream(websocket: WebSocket):
    """Real-time push WebSocket for live rounds, pre-round predictions, and metrics."""
    await websocket.accept()
    queue = live_manager.subscribe()
    try:
        # Push initial snapshot immediately
        initial_snapshot = live_manager.get_state_snapshot()
        await websocket.send_json(initial_snapshot)

        while True:
            # Wait for next state update
            snapshot = await queue.get()
            await websocket.send_json(snapshot)
    except WebSocketDisconnect:
        logger.debug("Live WebSocket client disconnected.")
    except Exception as e:
        logger.debug(f"Live WebSocket error: {e}")
    finally:
        live_manager.unsubscribe(queue)


@app.websocket("/api/live/bridge-ws")
async def websocket_bridge_stream(websocket: WebSocket):
    """Receive live telemetry frames directly from browser bridge companion window."""
    await websocket.accept()
    live_manager.bridge_connected = True
    await live_manager.broadcast_state()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                import json
                payload = json.loads(data)
                raw_frame = payload.get("raw_frame", data)
            except Exception:
                raw_frame = data
            await live_manager.ingest_raw(raw_frame)
    except WebSocketDisconnect:
        logger.debug("Bridge WebSocket disconnected.")
    except Exception as e:
        logger.debug(f"Bridge WebSocket error: {e}")



# ==============================================================================
# CSV FALLBACK & BENCHMARK ENDPOINTS
# ==============================================================================
@app.post("/api/upload")
async def upload_rounds(file: UploadFile = File(...)) -> JSONResponse:
    """Upload and process a CSV file of Aviator round records (Fallback Mode)."""
    if not file.filename.lower().endswith(".csv"):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Only CSV (.csv) files are supported.",
                "errors": [f"Uploaded file '{file.filename}' is not a CSV."],
            },
        )

    try:
        content = await file.read()
        decoded = content.decode("utf-8", errors="replace")
        df_raw = pd.read_csv(io.StringIO(decoded))
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Failed to parse CSV file: {str(e)}",
                "errors": [str(e)],
            },
        )

    result = process_round_records(df_raw, filename=file.filename)
    if result.get("status") == "error":
        return JSONResponse(status_code=422, content=result)

    return JSONResponse(content=result)


@app.get("/api/sample")
async def get_sample_data(
    source: str = Query("synthetic", pattern="^(synthetic|real)$")
) -> JSONResponse:
    """Load and process authentic sample crash dataset for 1-click verification."""
    if source == "real":
        file_path = Path("data/raw/real_aviator_historical.csv")
        filename = "real_aviator_historical_sample.csv"
    else:
        file_path = Path("data/raw/synthetic_sample.csv")
        filename = "synthetic_sample_verification.csv"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample file {file_path} not found on server.")

    try:
        # Load a representative sample (250 rounds)
        df_full = load_csv(str(file_path))
        df_sample = df_full.head(250).copy()
        result = process_round_records(df_sample, filename=filename)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing sample data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmark")
async def get_benchmark_table() -> JSONResponse:
    """Return model evaluation comparison benchmark metrics."""
    table = load_benchmark_comparison_table()
    return JSONResponse(content={"benchmark": table})


# Mount static directory for frontend
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the single-page frontend application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_file))


@app.get("/bridge")
@app.get("/bridge.html")
async def serve_bridge() -> FileResponse:
    """Serve the lightweight telemetry bridge companion window."""
    bridge_file = STATIC_DIR / "bridge.html"
    if not bridge_file.exists():
        raise HTTPException(status_code=404, detail="Bridge page not found.")
    return FileResponse(
        str(bridge_file),
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )

