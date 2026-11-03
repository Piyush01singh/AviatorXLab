"""FastAPI application serving the Aviator ML Research Live Dashboard & CSV API."""

import io
import json
from pathlib import Path
from typing import Any, Dict, Optional
import uuid
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
    safe_snapshot = json.loads(json.dumps(snapshot, default=str))
    return JSONResponse(content=safe_snapshot)


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
# NEXT ROUND PREDICTION ENDPOINT (Requirement 4)
# ==============================================================================
@app.get("/api/prediction/next-round")
async def get_next_round_prediction() -> JSONResponse:
    """Return dedicated Next Round Prediction snapshot for the upcoming round.

    Provides target round ID, WAITING/FLYING/unavailable phase, champion model name,
    checkpoint version, 5 threshold probabilities, generation timestamp, verification status,
    game source status, last event, and forecast freshness.
    If unverified or post-flying, displays exactly 'No valid pre-round prediction available'.
    """
    snap = live_manager.get_next_round_prediction_snapshot()
    safe_snap = json.loads(json.dumps(snap, default=str))
    return JSONResponse(content=safe_snap)


# ==============================================================================
# IMMUTABLE LEDGER & ADAPTIVE RESEARCH ENDPOINTS
# ==============================================================================
@app.get("/api/live/ledger")
async def get_live_ledger() -> JSONResponse:
    """View immutable pre-round forecasts, resolutions, and cryptographic integrity audit."""
    from src.collector.ledger import ledger
    return JSONResponse(content={
        "integrity": ledger.verify_ledger_integrity(),
        "recent_entries": ledger.get_recent_entries(20),
    })


@app.get("/api/live/shadow-learner")
async def get_shadow_learner_status() -> JSONResponse:
    """View candidate adaptive ensemble weights and out-of-sample loss tracking."""
    from src.models.shadow_learner import shadow_learner
    return JSONResponse(content=shadow_learner.get_candidate_status())


@app.get("/api/collection/setup-info")
async def get_collection_setup_info() -> JSONResponse:
    """Return connection status and default parameters for Overnight Session Setup dialog."""
    from src.collector.collector import collector
    return JSONResponse(content={
        "default_duration_hours": 6.0,
        "default_target_rounds": None,
        "source_locked": "browser_observer",
        "is_collecting": collector.is_active,
        "session_status": collector.session_status,
        "session_id": collector.session_id,
        "bridge_connected": live_manager.bridge_connected,
        "observer_installed": live_manager.observer_installed,
        "observer_status": live_manager.observer_status,
        "observer_host": live_manager.observer_host,
        "total_authentic_rounds": len(collector.collected_rounds),
        "stopping_summary": collector.stopping_condition_desc,
    })


@app.post("/api/collection/start")
async def start_collection(request: Request) -> JSONResponse:
    """Start an overnight or scheduled genuine Stake telemetry collection session."""
    from src.collector.collector import collector
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    # Source validation: MUST be browser_observer
    req_source = body.get("source", "browser_observer")
    if req_source != "browser_observer":
        raise HTTPException(
            status_code=400,
            detail="Source is locked to genuine 'browser_observer' Stake telemetry; sample_stream and synthetic sources are forbidden."
        )

    # Multiple session prevention
    if collector.is_active or collector.session_status in ("COLLECTING", "FINALIZING"):
        raise HTTPException(
            status_code=409,
            detail=f"A collection session ('{collector.session_id}') is already active. Multiple simultaneous sessions are not permitted."
        )

    dur_hours = body.get("duration_hours", 6.0)
    if dur_hours is not None:
        try:
            dur_hours = float(dur_hours)
            if dur_hours <= 0:
                raise HTTPException(status_code=400, detail="Collection duration in hours must be greater than 0.")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid duration_hours provided.")

    tgt_rounds = body.get("target_rounds")
    if tgt_rounds is not None and tgt_rounds != "":
        try:
            tgt_rounds = int(tgt_rounds)
            if tgt_rounds <= 0:
                raise HTTPException(status_code=400, detail="Maximum completed-round limit must be an integer >= 1.")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid target_rounds provided.")
    else:
        tgt_rounds = None

    try:
        res = collector.start_session(
            duration_hours=dur_hours,
            duration_seconds=body.get("duration_seconds"),
            target_rounds=tgt_rounds,
            schedule_end_time=body.get("schedule_end_time"),
            source="browser_observer",
            auto_train=body.get("auto_train", False),
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    await live_manager.broadcast_state()
    return JSONResponse(content=res)


@app.post("/api/collection/stop")
async def stop_collection() -> JSONResponse:
    """Stop active collection session without resetting accumulated authentic rounds."""
    from src.collector.collector import collector
    res = collector.stop_session()
    await live_manager.broadcast_state()
    return JSONResponse(content=res)


@app.get("/api/collection/status")
async def get_collection_status() -> JSONResponse:
    """Return genuine Stake collection progress, round count, and dataset hash."""
    from src.collector.collector import collector
    return JSONResponse(content=collector.get_session_progress())


@app.get("/api/training/retrain-preview")
@app.post("/api/training/retrain-preview")
async def get_retrain_preview() -> JSONResponse:
    """Provide complete dataset manifest, split, model, and safeguard preview for Retraining Confirmation Dialog."""
    from src.collector.collector import collector
    from src.data.dataset_aggregator import dataset_aggregator

    df_agg, manifest = dataset_aggregator.aggregate_and_manifest()
    summary = manifest.get("summary", {})
    final_rows = summary.get("final_training_row_count", len(df_agg))
    dataset_hash = manifest.get("corpus_sha256", "empty")

    token = uuid.uuid4().hex
    is_insufficient = (final_rows < 120)

    return JSONResponse(content={
        "session_id": collector.session_id or "aggregated_corpus",
        "dataset_file": "Aggregated Multi-Session Corpus",
        "dataset_sha256": dataset_hash,
        "collection_time_range": summary.get("date_range", {
            "start": collector.session_start_time_iso or "N/A",
            "end": collector.last_valid_round_timestamp or "N/A",
        }),
        "authentic_completed_rounds": collector.session_rounds_collected,
        "total_authentic_rounds": len(collector.collected_rounds),
        "eligible_training_examples": final_rows,
        "record_counts": {
            "accepted": summary.get("total_valid_rows", final_rows),
            "incomplete": summary.get("total_rejected_rows", 0),
            "rejected": summary.get("total_rejected_rows", 0),
            "duplicate": summary.get("duplicates_removed_count", 0),
            "conflicts": summary.get("conflicts_detected_count", 0),
        },
        "manifest": manifest,
        "dataset_manifest": manifest,
        "input_files": manifest.get("files", []),
        "manifest_summary": summary,
        "final_training_row_count": final_rows,
        "historical_data": {
            "source": "data/raw/real_aviator_historical.csv",
            "provenance": "Provably Fair Stake Historical Archive",
            "row_count": 15962,
        },
        "models_to_retrain": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "MLP",
            "LSTM",
            "GRU",
        ],
        "split_plan": "Chronological 4-Way Split: 60% Train, 15% Calibration, 15% Validation, 10% Untouched Future Test",
        "calibration_methods": [
            "Platt Sigmoid Scaling (Separate Calibration Split)",
            "Isotonic Regression (Separate Calibration Split)",
        ],
        "validation_strategy": "Out-of-sample multi-threshold Brier score and Log Loss on untouched test partition",
        "promotion_safeguards": (
            "Paired loss difference significance (Holm-corrected alpha < 0.05), "
            "Brier & Log Loss margin thresholds, strict probability monotonicity P(1.5x)≥...≥P(10x), "
            "minimum sample gate (N≥120)."
        ),
        "insufficient_sample_warning": is_insufficient,
        "confirmation_token": token,
        "collection_state": collector.session_status,
    })


@app.post("/api/training/overnight/start")
async def trigger_overnight_training(request: Request) -> JSONResponse:
    """Trigger full six-model candidate retraining, calibration, and conditional promotion."""
    from src.collector.collector import collector
    from src.training.overnight_pipeline import overnight_pipeline
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    # Explicit confirmation requirement
    confirmed = body.get("confirmed", False)
    if not confirmed and not body.get("confirmation_token"):
        raise HTTPException(
            status_code=400,
            detail="Explicit confirmation required to start model retraining. Review dataset preview and confirm."
        )

    # Block training if overnight dataset validation failed
    if collector.session_status == "FAILED":
        raise HTTPException(
            status_code=422,
            detail="Retraining blocked: overnight dataset validation failed critical integrity checks."
        )

    # Idempotent job execution guard
    if overnight_pipeline.is_training:
        return JSONResponse(content={
            "status": "running",
            "job_id": overnight_pipeline.active_job_id,
            "message": "Retraining job is already running.",
            "progress": overnight_pipeline.job_progress,
        })

    min_test = int(body.get("min_test_rounds", 40))
    res = overnight_pipeline.run_pipeline(
        min_test_rounds=min_test,
        confirmed=True,
        force_promotion=False,
    )
    registry._is_loaded = False
    registry.load_all_models()
    await live_manager.broadcast_state()
    return JSONResponse(content=res)


@app.get("/api/training/overnight/status")
async def get_overnight_training_status() -> JSONResponse:
    """Return training job progress, active champion version, rollback version, and collection progress."""
    from src.collector.collector import collector
    from src.training.overnight_pipeline import overnight_pipeline
    app_file = Path("models/approved_version.json")
    meta = {}
    if app_file.exists():
        try:
            with open(app_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass
    return JSONResponse(content={
        "approved_version": meta,
        "collection_status": collector.get_session_progress(),
        "training_job": overnight_pipeline.get_job_status(),
    })


@app.post("/api/training/rollback")
async def rollback_model() -> JSONResponse:
    """Roll back active model checkpoint to the previous approved configuration."""
    from src.training.overnight_pipeline import overnight_pipeline
    res = overnight_pipeline.rollback()
    registry._is_loaded = False
    registry.load_all_models()
    await live_manager.broadcast_state()
    return JSONResponse(content=res)


@app.get("/api/research/statistical-signal")
async def get_statistical_signal() -> JSONResponse:
    """Run autocorrelation, serial dependence, and Provably Fair distribution audits."""
    from src.evaluation.statistical_signal import run_autocorrelation_audit, run_distribution_fairness_audit
    mults = [r["multiplier"] for r in live_manager.history_records]
    if len(mults) < 30:
        p = Path("data/raw/real_aviator_historical.csv")
        if p.exists():
            df = load_csv(str(p))
            mults = df["multiplier"].tolist()
    autocorr = run_autocorrelation_audit(mults, max_lags=20)
    fairness = run_distribution_fairness_audit(mults, rtp=0.97)
    return JSONResponse(content={
        "autocorrelation_audit": autocorr,
        "distribution_fairness_audit": fairness,
    })



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

