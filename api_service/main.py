"""API Service — async request ingress + result polling via Redis.

Temporary flow:
  POST /recommend -> publish to signals.detected queue (bypasses `vision_service`)
  GET /result/{request_id} -> read completed result from Redis

Event chain:
  api_service -> signals.detected -> matching_service
  -> routine.matched -> explanation_service
  -> routine.completed -> api_service (background consumer -> Redis)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Any

import pika
import redis
from fastapi import FastAPI
from pydantic import BaseModel

from matching_service.core.models import SkinType, UserPreferences

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="SafeGlow AI — API Service")

RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# TEMPORARY simplified flow (for wiring checks only).
# We bypass `vision_service` and derive `visual_signals` from questionnaire input.
QUEUE_SIGNALS_DETECTED: str = "signals.detected"

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


def _build_visual_signals_from_preferences(user_preferences: UserPreferences) -> list[str]:
    """Derive visual signals from questionnaire input (temp flow)."""
    signals: list[str] = []
    if user_preferences.skin_type == SkinType.DRY:
        signals.append("dry")
    elif user_preferences.skin_type == SkinType.OILY:
        signals.append("oily")
    elif user_preferences.skin_type == SkinType.COMBINATION:
        signals.append("combination")

    if user_preferences.has_breakouts:
        signals.append("acne")

    return signals


def publish_signals_detected(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    request_id: str,
    user_preferences: UserPreferences,
) -> None:
    """Publish a signals.detected event (temp flow)."""
    payload: dict[str, Any] = {
        "request_id": request_id,
        "visual_signals": _build_visual_signals_from_preferences(user_preferences),
        "user_preferences": user_preferences.model_dump(mode="json"),
    }
    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_SIGNALS_DETECTED,
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
            channel.queue_declare(queue=QUEUE_SIGNALS_DETECTED, durable=True)
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
    _stop_event.clear()
    _consumer_thread = threading.Thread(target=_consume_completed_events, daemon=True)
    _consumer_thread.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    _stop_event.set()
    if _consumer_thread and _consumer_thread.is_alive():
        _consumer_thread.join(timeout=3)


@app.post("/recommend")
async def recommend(request: RecommendRequest) -> dict[str, str]:
    request_id = uuid.uuid4().hex

    connection, channel = _build_rabbitmq_channel()
    try:
        publish_signals_detected(
            channel,
            request_id=request_id,
            user_preferences=request.user_preferences,
        )
        logger.info("Published signals.detected request_id=%s", request_id)
    finally:
        if channel.is_open:
            channel.close()
        if connection.is_open:
            connection.close()

    return {"status": "processing", "request_id": request_id}


@app.get("/result/{request_id}")
async def get_result(request_id: str) -> dict[str, Any]:
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
