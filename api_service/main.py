"""API Service — receives HTTP requests and publishes routine.requested events."""

import logging
import os

from fastapi import FastAPI

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] request_id=%(name)s %(message)s",
)

app = FastAPI(title="SafeGlow AI — API Service")

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")


@app.post("/recommend")
async def recommend() -> dict:
    # TODO: parse UserConstraints from request body, publish RoutineRequestedEvent
    raise NotImplementedError
