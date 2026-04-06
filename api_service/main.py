"""API Service — async request ingress + result polling via Redis.

Event chain (local pipeline):
  api_service -> image.uploaded -> vision_service
  -> signals.detected -> matching_service
  -> routine.matched -> explanation_service
  -> routine.completed -> api_service (background consumer -> Redis)

Bridge demo (Azure UI + local workers):
  Browser -> POST /submit -> forward to LOCAL_BRIDGE -> local RabbitMQ pipeline
  -> bridge POST /internal/result-callback -> in-memory store
  -> browser polls GET /result/{request_id}
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pika
import redis
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api_service.bridge_forward import post_to_local_bridge
from api_service.request_store import bridge_request_store
from matching_service.core.models import UserPreferences

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="SafeGlow AI — API Service")

_STATIC_DIR = Path(__file__).resolve().parent / "static"

RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# Set to false on Azure when using only the local-bridge flow (no RabbitMQ in the cloud).
ENABLE_RABBIT_RESULT_CONSUMER: bool = os.getenv(
    "ENABLE_RABBIT_RESULT_CONSUMER", "true"
).strip().lower() in ("1", "true", "yes", "on")

QUEUE_IMAGE_UPLOADED: str = "image.uploaded"

EXCHANGE_ROUTINE_EVENTS: str = "routine.events"
ROUTING_KEY_COMPLETED: str = "routine.completed"
QUEUE_COMPLETED: str = "api.routine.completed.q"

RESULT_TTL_SECONDS: int = 20 * 60

_stop_event = threading.Event()
_consumer_thread: threading.Thread | None = None
_redis_client: redis.Redis | None = None


class RecommendRequest(BaseModel):
    image_path: str
    user_preferences: UserPreferences


class ResultCallbackBody(BaseModel):
    request_id: str
    status: str = Field(..., pattern="^(completed|failed)$")
    result: dict[str, Any] | None = None
    error: str | None = None


def redis_result_key(request_id: str) -> str:
    return f"result:{request_id}"


def store_completed_result(
    redis_client: redis.Redis,
    request_id: str,
    result: dict[str, Any],
    ttl_seconds: int = RESULT_TTL_SECONDS,
) -> None:
    redis_client.setex(redis_result_key(request_id), ttl_seconds, json.dumps(result))


def read_completed_result(redis_client: redis.Redis, request_id: str) -> dict[str, Any] | None:
    payload = redis_client.get(redis_result_key(request_id))
    if payload is None:
        return None
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)


def publish_image_uploaded(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    request_id: str,
    image_path: str,
    user_preferences: UserPreferences,
) -> None:
    """Publish an image.uploaded event for the vision service."""
    payload: dict[str, Any] = {
        "request_id": request_id,
        "image_path": image_path,
        "user_preferences": user_preferences.model_dump(mode="json"),
    }
    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_IMAGE_UPLOADED,
        properties=pika.BasicProperties(
            content_type="application/json",
            content_encoding="utf-8",
            delivery_mode=2,
        ),
        body=json.dumps(payload).encode("utf-8"),
    )


def _build_rabbitmq_channel(
    max_retries: int = 10,
    retry_delay: float = 5.0,
) -> tuple[pika.BlockingConnection, pika.adapters.blocking_connection.BlockingChannel]:
    for attempt in range(1, max_retries + 1):
        try:
            connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_IMAGE_UPLOADED, durable=True)
            channel.exchange_declare(
                exchange=EXCHANGE_ROUTINE_EVENTS,
                exchange_type="direct",
                durable=True,
            )
            channel.queue_declare(queue=QUEUE_COMPLETED, durable=True)
            channel.queue_bind(
                queue=QUEUE_COMPLETED,
                exchange=EXCHANGE_ROUTINE_EVENTS,
                routing_key=ROUTING_KEY_COMPLETED,
            )
            logger.info("Connected to RabbitMQ (attempt %d/%d)", attempt, max_retries)
            return connection, channel
        except pika.exceptions.AMQPConnectionError:
            if attempt == max_retries:
                logger.critical(
                    "Failed to connect to RabbitMQ after %d attempts — giving up",
                    max_retries,
                )
                raise
            logger.warning(
                "RabbitMQ not ready, retrying in %ds... (attempt %d/%d)",
                retry_delay,
                attempt,
                max_retries,
            )
            time.sleep(retry_delay)
    raise RuntimeError("Unreachable")


def _consume_completed_events() -> None:
    """Background thread: consume routine.completed and store results in Redis."""
    global _redis_client

    if _redis_client is None:
        _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)

    connection, channel = _build_rabbitmq_channel()

    def on_message(
        ch: pika.adapters.blocking_connection.BlockingChannel,
        method: pika.spec.Basic.Deliver,
        _properties: pika.BasicProperties,
        body: bytes,
    ) -> None:
        try:
            logger.info("Received message from %s", QUEUE_COMPLETED)
            payload: dict[str, Any] = json.loads(body.decode("utf-8"))
            request_id = extract_request_id(payload)
            store_completed_result(_redis_client, request_id, payload)
            logger.info("Stored completed result request_id=%s in Redis", request_id)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            logger.exception("Failed to handle routine.completed message")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_COMPLETED, on_message_callback=on_message)

    logger.info("API result consumer started queue=%s", QUEUE_COMPLETED)
    while not _stop_event.is_set():
        connection.process_data_events(time_limit=1.0)

    try:
        channel.close()
    finally:
        connection.close()


@app.on_event("startup")
def on_startup() -> None:
    global _consumer_thread, _redis_client

    _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    if not ENABLE_RABBIT_RESULT_CONSUMER:
        logger.info(
            "Skipping RabbitMQ routine.completed consumer (ENABLE_RABBIT_RESULT_CONSUMER=false)"
        )
        return
    _stop_event.clear()
    _consumer_thread = threading.Thread(target=_consume_completed_events, daemon=True)
    _consumer_thread.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    _stop_event.set()
    if _consumer_thread is not None and _consumer_thread.is_alive():
        _consumer_thread.join(timeout=3)


def _forward_to_bridge_task(
    request_id: str,
    file_content: bytes,
    filename: str,
    skin_type: str,
    has_breakouts: bool,
    sensitivities_json: str,
    bridge_url: str,
    callback_url: str,
    callback_token: str,
) -> None:
    """Background: POST multipart to local tunnel bridge."""
    try:
        bridge_request_store.set_status(request_id, "processing")
        post_to_local_bridge(
            bridge_url,
            file_content=file_content,
            filename=filename,
            request_id=request_id,
            skin_type=skin_type,
            has_breakouts=has_breakouts,
            sensitivities_json=sensitivities_json,
            callback_url=callback_url,
            callback_token=callback_token,
        )
        logger.info("Bridge forward finished for request_id=%s (awaiting callback)", request_id)
    except Exception:
        logger.exception("Bridge forward failed request_id=%s", request_id)
        bridge_request_store.fail(request_id, "Failed to reach local bridge or bridge rejected request")


@app.post("/submit")
async def submit_demo(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    skin_type: str = Form(...),
    has_breakouts: str = Form("false"),
    sensitivities: str = Form(...),
) -> dict[str, Any]:
    """Multipart demo submit for Azure UI: forwards to ``LOCAL_BRIDGE_URL``."""
    bridge_url = os.getenv("LOCAL_BRIDGE_URL", "").strip()
    result_callback_url = os.getenv("RESULT_CALLBACK_URL", "").strip()
    callback_token = os.getenv("CALLBACK_TOKEN", "").strip()
    if not bridge_url:
        raise HTTPException(
            status_code=503,
            detail="LOCAL_BRIDGE_URL is not configured on this server.",
        )
    if not result_callback_url or not callback_token:
        raise HTTPException(
            status_code=503,
            detail="RESULT_CALLBACK_URL and CALLBACK_TOKEN must be configured.",
        )

    request_id = uuid.uuid4().hex
    file_content = await image.read()
    filename = image.filename or "upload.jpg"
    hb = has_breakouts.lower() in ("true", "1", "yes", "on")

    bridge_request_store.create(request_id, "queued")
    logger.info(
        "submit received request_id=%s skin_type=%s has_breakouts=%s filename=%r size=%d",
        request_id,
        skin_type,
        hb,
        filename,
        len(file_content),
    )

    background_tasks.add_task(
        _forward_to_bridge_task,
        request_id,
        file_content,
        filename,
        skin_type,
        hb,
        sensitivities,
        bridge_url,
        result_callback_url,
        callback_token,
    )

    return {"request_id": request_id, "status": "queued"}


@app.post("/internal/result-callback")
async def result_callback(
    body: ResultCallbackBody,
    x_callback_token: str | None = Header(default=None, alias="X-Callback-Token"),
) -> dict[str, str]:
    """Local bridge posts final routine payload here (protected)."""
    expected_token = os.getenv("CALLBACK_TOKEN", "").strip()
    if not expected_token:
        raise HTTPException(status_code=503, detail="CALLBACK_TOKEN not configured")
    if x_callback_token != expected_token:
        logger.warning("Callback rejected: invalid or missing X-Callback-Token")
        raise HTTPException(status_code=401, detail="Invalid callback token")

    if body.status == "completed":
        if not body.result:
            raise HTTPException(status_code=400, detail="result required when status is completed")
        ok = bridge_request_store.complete(body.request_id, body.result)
        if not ok:
            logger.warning("Callback for unknown request_id=%s", body.request_id)
            raise HTTPException(status_code=404, detail="Unknown request_id")
        logger.info("Callback received request_id=%s status=completed", body.request_id)
        return {"ok": "true"}

    err = body.error or "Unknown error"
    ok = bridge_request_store.fail(body.request_id, err)
    if not ok:
        logger.warning("Callback fail for unknown request_id=%s", body.request_id)
        raise HTTPException(status_code=404, detail="Unknown request_id")
    logger.info("Callback received request_id=%s status=failed", body.request_id)
    return {"ok": "true"}


@app.post("/recommend")
async def recommend(request: RecommendRequest) -> dict[str, str]:
    request_id = uuid.uuid4().hex

    connection, channel = _build_rabbitmq_channel()
    try:
        publish_image_uploaded(
            channel,
            request_id=request_id,
            image_path=request.image_path,
            user_preferences=request.user_preferences,
        )
        logger.info("Published image.uploaded request_id=%s", request_id)
    finally:
        if channel.is_open:
            channel.close()
        if connection.is_open:
            connection.close()

    return {"status": "processing", "request_id": request_id}


@app.get("/result/{request_id}")
async def get_result(request_id: str) -> dict[str, Any]:
    bridge_row = bridge_request_store.get(request_id)
    if bridge_row is not None:
        out: dict[str, Any] = {
            "request_id": request_id,
            "status": bridge_row["status"],
        }
        if bridge_row["status"] == "completed" and bridge_row["result"] is not None:
            out["result"] = bridge_row["result"]
        if bridge_row["status"] == "failed" and bridge_row["error"]:
            out["error"] = bridge_row["error"]
        return out

    if _redis_client is None:
        raise RuntimeError("Redis client was not initialized")

    result = read_completed_result(_redis_client, request_id)
    if result is None:
        return {"status": "pending", "request_id": request_id}
    return {"status": "completed", "request_id": request_id, "result": result}


def extract_request_id(payload: dict[str, Any]) -> str:
    """Extract and validate request_id from a routine.completed payload."""
    request_id: str | None = payload.get("request_id")
    if not request_id:
        raise ValueError("Missing request_id in routine.completed payload")
    return request_id


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
