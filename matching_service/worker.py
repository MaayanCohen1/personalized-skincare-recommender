"""Matching Service RabbitMQ worker.

Consumes ``signals.detected`` messages, runs the matching pipeline
(hard-filter + optional semantic ranking), and publishes the result
to ``routine.matched`` via the ``routine.events`` exchange.

Event chain:
  api_service -> signals.detected -> **this worker**
  -> routine.matched -> explanation_service
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pika

from matching_service.core.message_handler import handle_signals_detected_message
from shared.models import Product

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE_SIGNALS_DETECTED: str = "signals.detected"
EXCHANGE_NAME: str = "routine.events"
ROUTING_KEY_MATCHED: str = "routine.matched"

_DATA_DIR: Path = Path(__file__).resolve().parent / "data"
_CATALOG_PATH: Path = _DATA_DIR / "products.json"


def _load_catalog() -> list[Product]:
    """Load the product catalog from the bundled JSON file."""
    raw = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    products_raw: list[dict[str, Any]] = raw.get("products", raw) if isinstance(raw, dict) else raw
    return [Product.model_validate(p) for p in products_raw]


def _build_semantic_ranker() -> Callable[[list[str], list[Product]], list[Product]] | None:
    """Return a SemanticMatcher.rank callable, or None on failure."""
    try:
        from matching_service.core.semantic_search import SemanticMatcher

        matcher = SemanticMatcher()
        logger.info("SemanticMatcher initialised successfully")
        return matcher.rank
    except Exception:
        logger.exception("SemanticMatcher init failed; ranking falls back to category priority")
        return None


def _connect(
    max_retries: int = 10,
    retry_delay: float = 5.0,
) -> tuple[pika.BlockingConnection, pika.adapters.blocking_connection.BlockingChannel]:
    """Connect to RabbitMQ with retries, declare queues and exchange."""
    for attempt in range(1, max_retries + 1):
        try:
            connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_SIGNALS_DETECTED, durable=True)
            channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type="direct", durable=True)
            logger.info("Connected to RabbitMQ (attempt %d/%d)", attempt, max_retries)
            return connection, channel
        except pika.exceptions.AMQPConnectionError:
            if attempt == max_retries:
                logger.critical("Failed to connect after %d attempts — giving up", max_retries)
                raise
            logger.warning(
                "RabbitMQ not ready, retrying in %ds… (attempt %d/%d)",
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
    ranker: Callable[[list[str], list[Product]], list[Product]] | None = None,
) -> None:
    """Process a single signals.detected message and publish routine.matched."""
    try:
        result = handle_signals_detected_message(body, catalog, ranker=ranker)

        channel.basic_publish(
            exchange=EXCHANGE_NAME,
            routing_key=ROUTING_KEY_MATCHED,
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
            ),
            body=json.dumps(result).encode("utf-8"),
        )
        logger.info("Published routine.matched request_id=%s", result.get("request_id"))
        channel.basic_ack(delivery_tag=method.delivery_tag)

    except Exception:
        logger.exception("Failed to process signals.detected message")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main() -> None:
    """Entry point — load data, connect, and start consuming."""
    catalog = _load_catalog()
    logger.info("Loaded %d products from catalog", len(catalog))

    ranker = _build_semantic_ranker()

    connection, channel = _connect()
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue=QUEUE_SIGNALS_DETECTED,
        on_message_callback=lambda ch, method, props, body: _on_message(
            ch, method, props, body, catalog, ranker,
        ),
    )

    logger.info("Matching worker started — consuming queue=%s", QUEUE_SIGNALS_DETECTED)
    try:
        channel.start_consuming()
    finally:
        channel.close()
        connection.close()


if __name__ == "__main__":
    main()
