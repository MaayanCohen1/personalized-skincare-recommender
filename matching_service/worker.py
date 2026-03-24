"""RabbitMQ worker for the matching service.

Consumes signals.detected, delegates to the pure message handler,
and publishes to products.matched.
"""

from __future__ import annotations

import json
import logging
import os
import time

import pika
import pika.adapters.blocking_connection
import pika.exceptions
import pika.spec

from matching_service.core.message_handler import handle_signals_detected_message
from matching_service.matching import default_catalog
from shared.models import Product

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

QUEUE_CONSUME: str = "signals.detected"
QUEUE_PUBLISH: str = "products.matched"


def _connect(
    max_retries: int = 10,
    retry_delay: float = 5.0,
) -> tuple[pika.BlockingConnection, pika.adapters.blocking_connection.BlockingChannel]:
    """Connect to RabbitMQ with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_CONSUME, durable=True)
            channel.queue_declare(queue=QUEUE_PUBLISH, durable=True)
            logger.info("Connected to RabbitMQ (attempt %d/%d)", attempt, max_retries)
            return connection, channel
        except pika.exceptions.AMQPConnectionError:
            if attempt == max_retries:
                logger.critical(
                    "Failed to connect to RabbitMQ after %d attempts",
                    max_retries,
                )
                raise
            logger.warning(
                "RabbitMQ not ready, retrying in %.0fs… (attempt %d/%d)",
                retry_delay,
                attempt,
                max_retries,
            )
            time.sleep(retry_delay)
    raise RuntimeError("Unreachable")


def _on_message(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    method: pika.spec.Basic.Deliver,
    properties: pika.BasicProperties,
    body: bytes,
    catalog: list[Product],
) -> None:
    """Process a single signals.detected message."""
    try:
        logger.info(
            "Received message from %s (delivery_tag=%s)",
            QUEUE_CONSUME,
            method.delivery_tag,
        )

        result = handle_signals_detected_message(body, catalog)

        channel.basic_publish(
            exchange="",
            routing_key=QUEUE_PUBLISH,
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
            ),
            body=json.dumps(result).encode("utf-8"),
        )

        channel.basic_ack(delivery_tag=method.delivery_tag)
        logger.info(
            "Published to %s (request_id=%s)",
            QUEUE_PUBLISH,
            result.get("request_id"),
        )
    except Exception:
        logger.exception("Failed to process signals.detected message")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main() -> None:
    """Entry point: load catalog, connect, and start consuming."""
    catalog = default_catalog()
    logger.info("Loaded catalog with %d products", len(catalog))

    connection, channel = _connect()

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue=QUEUE_CONSUME,
        on_message_callback=lambda ch, method, props, body: _on_message(
            ch, method, props, body, catalog
        ),
    )

    logger.info("Worker started — consuming from %s", QUEUE_CONSUME)
    try:
        channel.start_consuming()
    finally:
        if channel.is_open:
            channel.close()
        if connection.is_open:
            connection.close()

if __name__ == "__main__":
    main()
