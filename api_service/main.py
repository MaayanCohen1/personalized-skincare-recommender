"""API Service — async request ingress + result polling via Redis.

Flow in this vertical slice:
  POST /recommend -> publish routine.requested
  GET /result/{request_id} -> read completed result from Redis

A background RabbitMQ consumer stores routine.completed events into Redis.
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
from pydantic import BaseModel, Field

from shared.models import RoutineCompletedEvent, RoutineRequestedEvent, UserConstraints

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="SafeGlow AI — API Service")

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

EXCHANGE_NAME = "routine.events"
ROUTING_KEY_REQUESTED = "routine.requested"
ROUTING_KEY_COMPLETED = "routine.completed"
QUEUE_COMPLETED = "api.routine.completed.q"
RESULT_TTL_SECONDS = 20 * 60

_stop_event = threading.Event()
_consumer_thread: threading.Thread | None = None
_redis_client: redis.Redis | None = None


class RecommendRequest(BaseModel):
    sensitivities: list[str] = Field(default_factory=list)
    max_products: int = Field(default=5, ge=1, le=10)
    image_path: str | None = None
    catalog_ref: str = "default"


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


def publish_routine_requested(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    event: RoutineRequestedEvent,
    request_id: str,
) -> None:
    body = {"request_id": request_id, "event": event.model_dump(mode="json")}
    props = pika.BasicProperties(
        content_type="application/json",
        correlation_id=request_id,
        delivery_mode=2,
    )
    channel.basic_publish(
        exchange=EXCHANGE_NAME,
        routing_key=ROUTING_KEY_REQUESTED,
        properties=props,
        body=json.dumps(body),
    )


def _build_rabbitmq_channel(
    max_retries: int = 10,
    retry_delay: float = 5.0,
) -> tuple[pika.BlockingConnection, pika.adapters.blocking_connection.BlockingChannel]:
    for attempt in range(1, max_retries + 1):
        try:
            connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
            channel = connection.channel()
            channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type="direct", durable=True)
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
    global _redis_client

    if _redis_client is None:
        _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)

    connection, channel = _build_rabbitmq_channel()
    channel.queue_declare(queue=QUEUE_COMPLETED, durable=True)
    channel.queue_bind(
        queue=QUEUE_COMPLETED,
        exchange=EXCHANGE_NAME,
        routing_key=ROUTING_KEY_COMPLETED,
    )

    def on_message(
        ch: pika.adapters.blocking_connection.BlockingChannel,
        method: pika.spec.Basic.Deliver,
        properties: pika.BasicProperties,
        body: bytes,
    ) -> None:
        try:
            payload = json.loads(body.decode("utf-8"))
            request_id, event = extract_completed_envelope(
                payload=payload,
                correlation_id=properties.correlation_id,
            )
            store_completed_result(
                _redis_client,
                request_id,
                event.model_dump(mode="json"),
            )
            logger.info("Stored completed result request_id=%s in Redis", request_id)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            logger.exception("Failed to handle routine.completed message")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_COMPLETED, on_message_callback=on_message)

    logger.info("API completion consumer started queue=%s", QUEUE_COMPLETED)
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
    constraints = UserConstraints(
        request_id=request_id,
        sensitivities=request.sensitivities,
        max_products=request.max_products,
        image_path=request.image_path,
    )
    event = RoutineRequestedEvent(
        constraints=constraints,
        catalog_ref="default",
    )

    connection, channel = _build_rabbitmq_channel()
    try:
        publish_routine_requested(channel, event, request_id=request_id)
        logger.info("Published routine.requested request_id=%s", request_id)
    finally:
        channel.close()
        connection.close()

    return {"request_id": request_id}


@app.get("/result/{request_id}")
async def get_result(request_id: str) -> dict[str, Any]:
    if _redis_client is None:
        raise RuntimeError("Redis client was not initialized")

    result = read_completed_result(_redis_client, request_id)
    if result is None:
        return {"status": "pending", "request_id": request_id}
    return {"status": "completed", "request_id": request_id, "result": result}


def extract_completed_envelope(
    payload: dict[str, Any],
    correlation_id: str | None,
) -> tuple[str, RoutineCompletedEvent]:
    request_id = payload.get("request_id") or correlation_id
    if not request_id:
        raise ValueError("Missing request_id in envelope and correlation_id")
    if "event" not in payload:
        raise ValueError("Missing 'event' in routine.completed envelope")
    event = RoutineCompletedEvent.model_validate(payload["event"])
    return request_id, event
