"""Local HTTP bridge: receives uploads from Azure API, runs RabbitMQ pipeline, callbacks."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from local_bridge.callback_client import post_result_callback
from local_bridge.prefs import parse_demo_preferences
from local_bridge.rabbit_flow import run_local_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROUTINE_WAIT_TIMEOUT_SECONDS: float = float(
    os.getenv("ROUTINE_WAIT_TIMEOUT_SECONDS", "600")
)

app = FastAPI(title="SafeGlow AI — Local Bridge (demo)")


def _job(
    file_content: bytes,
    filename: str,
    request_id: str,
    skin_type: str,
    has_breakouts: str,
    sensitivities: str,
    callback_url: str,
    callback_token: str,
) -> None:
    path: str | None = None
    suffix = Path(filename or "upload.jpg").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = tmp.name
    try:
        tmp.write(file_content)
        tmp.close()
        logger.info(
            "Bridge temp file saved request_id=%s path=%s bytes=%d",
            request_id,
            path,
            len(file_content),
        )

        prefs = parse_demo_preferences(skin_type, has_breakouts, sensitivities)
        result = run_local_pipeline(
            request_id=request_id,
            image_path=path,
            user_preferences=prefs,
            wait_timeout_seconds=ROUTINE_WAIT_TIMEOUT_SECONDS,
        )
        post_result_callback(
            callback_url,
            callback_token,
            request_id=request_id,
            status="completed",
            result=result,
        )
    except Exception as exc:
        logger.exception("Bridge job failed request_id=%s", request_id)
        try:
            post_result_callback(
                callback_url,
                callback_token,
                request_id=request_id,
                status="failed",
                error=str(exc),
            )
        except Exception:
            logger.exception("Callback POST failed after job error request_id=%s", request_id)
    finally:
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                logger.warning("Could not remove temp file %s", path)


@app.post("/demo-submit")
async def demo_submit(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    request_id: str = Form(...),
    skin_type: str = Form(...),
    has_breakouts: str = Form("false"),
    sensitivities: str = Form(...),
    callback_url: str = Form(...),
    callback_token: str = Form(...),
) -> dict[str, str]:
    """Accept multipart from Azure, queue work, respond immediately."""
    content = await image.read()
    fname = image.filename or "upload.jpg"
    logger.info(
        "demo-submit received request_id=%s filename=%r size=%d",
        request_id,
        fname,
        len(content),
    )
    background_tasks.add_task(
        _job,
        content,
        fname,
        request_id,
        skin_type,
        has_breakouts,
        sensitivities,
        callback_url,
        callback_token,
    )
    return JSONResponse(
        {"status": "accepted", "request_id": request_id},
        status_code=202,
    )
