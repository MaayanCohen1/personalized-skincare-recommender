"""Explanation Service — I/O layer.

Consumes routine.matched events from RabbitMQ, triggers the CrewAI Crew,
and publishes routine.completed events.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")


def main() -> None:
    # TODO: connect to RabbitMQ, consume routine.matched,
    #       invoke crew.run(), publish RoutineCompletedEvent
    raise NotImplementedError


if __name__ == "__main__":
    main()
