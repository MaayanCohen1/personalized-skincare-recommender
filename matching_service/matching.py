"""Matching service I/O layer for the MVP event pipeline.

Consumes routine.requested and publishes routine.matched.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import pika

from matching_service.core.semantic_search import SemanticMatcher
from matching_service.rules_engine import build_routine, filter_safe_products
from shared.models import (
    Product,
    ProductCategory,
    RoutineMatchedEvent,
    RoutineRequestedEvent,
    SkinProfile,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

EXCHANGE_NAME = "routine.events"
ROUTING_KEY_REQUESTED = "routine.requested"
ROUTING_KEY_MATCHED = "routine.matched"
QUEUE_REQUESTED = "matching.routine.requested.q"


def default_catalog() -> list[Product]:
    return [
        Product(
            id="p-clean-1",
            name="Gentle Cleanser",
            category=ProductCategory.CLEANSER,
            ingredients=["glycerin", "water"],
            description="A gentle cleanser designed for daily use.",
        ),
        Product(
            id="p-moist-1",
            name="Barrier Moisturizer",
            category=ProductCategory.MOISTURIZER,
            ingredients=["niacinamide", "glycerin"],
            description="A moisturizer that supports skin barrier comfort.",
        ),
        Product(
            id="p-spf-1",
            name="Daily Mineral SPF",
            category=ProductCategory.SPF,
            ingredients=["zinc oxide", "glycerin"],
            description="A mineral sunscreen for daily UV protection.",
        ),
        Product(
            id="p-serum-1",
            name="Hydration Serum",
            category=ProductCategory.SERUM,
            ingredients=["hyaluronic acid", "aloe vera"],
            description="A hydration-focused serum for dry skin comfort.",
        ),
    ]


def rank_products_if_possible(
    safe_products: list[Product],
    skin_conditions: list[str],
) -> list[Product]:
    if not safe_products:
        return []
    if not skin_conditions:
        return list(safe_products)

    try:
        matcher = SemanticMatcher()
        return matcher.rank(skin_conditions=skin_conditions, products=safe_products)
    except Exception:
        logger.exception("Semantic rank failed; falling back to unsorted safe products")
        return list(safe_products)


def publish_routine_matched(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    event: RoutineMatchedEvent,
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
        routing_key=ROUTING_KEY_MATCHED,
        properties=props,
        body=json.dumps(body),
    )


def _build_channel(
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


def _handle_requested_payload(payload: dict[str, Any], correlation_id: str | None) -> tuple[RoutineMatchedEvent, str]:
    request_id = payload.get("request_id") or correlation_id
    if "event" not in payload:
        raise ValueError("Missing 'event' in routine.requested envelope")
    event = RoutineRequestedEvent.model_validate(payload["event"])
    if not request_id:
        raise ValueError("Missing request_id in envelope and correlation_id")

    profile = SkinProfile(request_id=request_id, skin_conditions=[])
    products = default_catalog()
    safe = filter_safe_products(products=products, constraints=event.constraints)
    ranked = rank_products_if_possible(safe_products=safe, skin_conditions=profile.skin_conditions)
    routine = build_routine(safe_products=ranked, constraints=event.constraints)

    matched_event = RoutineMatchedEvent(
        matched_products=routine,
        profile=profile,
        constraints=event.constraints,
    )
    return matched_event, request_id


def main() -> None:
    connection, channel = _build_channel()
    channel.queue_declare(queue=QUEUE_REQUESTED, durable=True)
    channel.queue_bind(
        queue=QUEUE_REQUESTED,
        exchange=EXCHANGE_NAME,
        routing_key=ROUTING_KEY_REQUESTED,
    )

    def on_message(
        ch: pika.adapters.blocking_connection.BlockingChannel,
        method: pika.spec.Basic.Deliver,
        properties: pika.BasicProperties,
        body: bytes,
    ) -> None:
        try:
            payload = json.loads(body.decode("utf-8"))
            matched_event, request_id = _handle_requested_payload(
                payload=payload,
                correlation_id=properties.correlation_id,
            )
            publish_routine_matched(ch, event=matched_event, request_id=request_id)
            logger.info("Published routine.matched request_id=%s", request_id)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            logger.exception("Failed to process routine.requested message")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_REQUESTED, on_message_callback=on_message)

    logger.info("Matching consumer started queue=%s", QUEUE_REQUESTED)
    try:
        channel.start_consuming()
    finally:
        channel.close()
        connection.close()


if __name__ == "__main__":
    main()
