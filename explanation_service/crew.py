"""Crew assembly — wires agents and tasks into a sequential CrewAI pipeline."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_crew():
    # TODO: instantiate agents and tasks, assemble into a sequential Crew
    raise NotImplementedError


def run_crew(matched_products, profile, constraints) -> dict[str, str]:
    """Entry point called by consumer.py — returns product_id → explanation mapping."""
    # TODO: call build_crew().kickoff() with the relevant inputs
    raise NotImplementedError
