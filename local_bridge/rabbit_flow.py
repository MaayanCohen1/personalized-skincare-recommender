"""Publish image.uploaded and wait for matching routine.completed (local demo)."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import pika

from matching_service.core.models import UserPreferences

logger = logging.getLogger(__name__)

RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_IMAGE_UPLOADED: str = "image.uploaded"
EXCHANGE_ROUTINE_EVENTS: str = "routine.events"
ROUTING_KEY_COMPLETED: str = "routine.completed"


def publish_image_uploaded(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    request_id: str,
    image_path: str,
    user_preferences: UserPreferences,
) -> None:
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


def run_local_pipeline(
    *,
    request_id: str,
    image_path: str,
    user_preferences: UserPreferences,
    wait_timeout_seconds: float,
) -> dict[str, Any]:
    """Drive the existing RabbitMQ pipeline and return the routine.completed body."""
    logger.info("Local flow started request_id=%s image_path=%s", request_id, image_path)
    connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
    channel = connection.channel()
    try:
        channel.queue_declare(queue=QUEUE_IMAGE_UPLOADED, durable=True)
        channel.exchange_declare(
            exchange=EXCHANGE_ROUTINE_EVENTS,
            exchange_type="direct",
            durable=True,
        )
        result_queue = channel.queue_declare(queue="", exclusive=True, auto_delete=True)
        qname = result_queue.method.queue
        channel.queue_bind(
            exchange=EXCHANGE_ROUTINE_EVENTS,
            queue=qname,
            routing_key=ROUTING_KEY_COMPLETED,
        )

        publish_image_uploaded(
            channel,
            request_id=request_id,
            image_path=image_path,
            user_preferences=user_preferences,
        )
        logger.info("Published image.uploaded request_id=%s", request_id)

        deadline = time.monotonic() + wait_timeout_seconds
        found: dict[str, Any] | None = None

        def on_message(
            ch: pika.adapters.blocking_connection.BlockingChannel,
            method: pika.spec.Basic.Deliver,
            _props: pika.BasicProperties,
            body: bytes,
        ) -> None:
            nonlocal found
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            rid = payload.get("request_id")
            if rid == request_id:
                found = payload
                ch.basic_ack(delivery_tag=method.delivery_tag)
                ch.stop_consuming()
            else:
                ch.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_qos(prefetch_count=10)
        channel.basic_consume(queue=qname, on_message_callback=on_message, auto_ack=False)
        while found is None and time.monotonic() < deadline:
            connection.process_data_events(time_limit=1.0)
        try:
            channel.stop_consuming()
        except Exception:
            pass

        if found is None:
            raise TimeoutError(
                f"No routine.completed for request_id={request_id} within {wait_timeout_seconds}s"
            )
        logger.info("Local flow finished request_id=%s", request_id)
        return found
    finally:
        if channel.is_open:
            channel.close()
        if connection.is_open:
            connection.close()
