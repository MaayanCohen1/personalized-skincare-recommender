"""Vision Service — consumes routine.requested, returns mocked skin analysis."""

import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")


def main() -> None:
    # TODO: connect to RabbitMQ, consume routine.requested,
    #       run mocked skin analysis, publish ProfileAnalyzedEvent
    raise NotImplementedError


if __name__ == "__main__":
    main()
