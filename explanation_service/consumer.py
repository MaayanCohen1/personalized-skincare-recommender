"""Explanation Service — consumes routine.matched and publishes routine.completed."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import pika

from explanation_service.crew import generate_explanation_for_product
from shared.models import Product, RoutineCompletedEvent, RoutineMatchedEvent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

EXCHANGE_NAME = "routine.events"
ROUTING_KEY_MATCHED = "routine.matched"
ROUTING_KEY_COMPLETED = "routine.completed"
QUEUE_MATCHED = "explanation.routine.matched.q"


def publish_routine_completed(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    event: RoutineCompletedEvent,
    request_id: str,
    image_analysis: dict[str, Any] | None = None,
    routine_rationale: dict[str, Any] | None = None,
) -> None:
    body: dict[str, Any] = {
        "request_id": request_id,
        "event": event.model_dump(mode="json"),
    }
    if image_analysis is not None:
        body["image_analysis"] = image_analysis
    if routine_rationale is not None:
        body["routine_rationale"] = routine_rationale
    props = pika.BasicProperties(
        content_type="application/json",
        correlation_id=request_id,
        delivery_mode=2,
    )
    channel.basic_publish(
        exchange=EXCHANGE_NAME,
        routing_key=ROUTING_KEY_COMPLETED,
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


def _generate_explanations(
    matched_products: list[Product],
    skin_conditions: list[str],
    request_id: str,
) -> dict[str, str]:
    explanations: dict[str, str] = {}
    for product in matched_products:
        response = generate_explanation_for_product(
            skin_conditions=skin_conditions,
            product_name=product.name,
            ingredients=product.ingredients,
            request_id=request_id,
        )
        explanations[product.id] = str(response.get("explanation_text", ""))
    return explanations


class _MatchedResult:
    """Internal container for a processed routine.matched message."""

    __slots__ = ("completed", "request_id", "image_analysis", "routine_rationale")

    def __init__(
        self,
        completed: RoutineCompletedEvent,
        request_id: str,
        image_analysis: dict[str, Any] | None,
        routine_rationale: dict[str, Any] | None,
    ) -> None:
        self.completed = completed
        self.request_id = request_id
        self.image_analysis = image_analysis
        self.routine_rationale = routine_rationale


def _handle_matched_payload(
    payload: dict[str, Any],
    correlation_id: str | None,
) -> _MatchedResult:
    request_id = payload.get("request_id") or correlation_id
    if "event" not in payload:
        raise ValueError("Missing 'event' in routine.matched envelope")
    matched_event = RoutineMatchedEvent.model_validate(payload["event"])
    if not request_id:
        raise ValueError("Missing request_id in envelope and correlation_id")

    explanations = _generate_explanations(
        matched_products=matched_event.matched_products,
        skin_conditions=matched_event.profile.skin_conditions,
        request_id=request_id,
    )

    completed = RoutineCompletedEvent(
        matched_products=matched_event.matched_products,
        explanations=explanations,
    )

    return _MatchedResult(
        completed=completed,
        request_id=request_id,
        image_analysis=payload.get("image_analysis"),
        routine_rationale=payload.get("routine_rationale"),
    )


def main() -> None:
    connection, channel = _build_channel()
    channel.queue_declare(queue=QUEUE_MATCHED, durable=True)
    channel.queue_bind(
        queue=QUEUE_MATCHED,
        exchange=EXCHANGE_NAME,
        routing_key=ROUTING_KEY_MATCHED,
    )

    def on_message(
        ch: pika.adapters.blocking_connection.BlockingChannel,
        method: pika.spec.Basic.Deliver,
        properties: pika.BasicProperties,
        body: bytes,
    ) -> None:
        try:
            payload = json.loads(body.decode("utf-8"))
            result = _handle_matched_payload(
                payload=payload,
                correlation_id=properties.correlation_id,
            )
            publish_routine_completed(
                ch,
                event=result.completed,
                request_id=result.request_id,
                image_analysis=result.image_analysis,
                routine_rationale=result.routine_rationale,
            )
            logger.info("Published routine.completed request_id=%s", result.request_id)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            logger.exception("Failed to process routine.matched message")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_MATCHED, on_message_callback=on_message)

    logger.info("Explanation consumer started queue=%s", QUEUE_MATCHED)
    try:
        channel.start_consuming()
    finally:
        channel.close()
        connection.close()


if __name__ == "__main__":
    main()
