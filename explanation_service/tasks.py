"""Pure task definitions for the explanation Crew — one task per agent."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_research_task():
    # TODO: define evidence-gathering task for Researcher agent
    raise NotImplementedError


def build_advising_task():
    # TODO: define explanation-drafting task for Advisor agent
    raise NotImplementedError


def build_qa_task():
    # TODO: define audit task for QA Auditor agent
    raise NotImplementedError
