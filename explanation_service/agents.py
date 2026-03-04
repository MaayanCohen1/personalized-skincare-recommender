"""Pure agent definitions for the explanation Crew.

Agents: Researcher, Advisor, QA Auditor.
No I/O here — only agent configuration.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_researcher_agent():
    # TODO: define Researcher agent (evidence retrieval)
    raise NotImplementedError


def build_advisor_agent():
    # TODO: define Advisor agent (routine explanation drafting)
    raise NotImplementedError


def build_qa_auditor_agent():
    # TODO: define QA Auditor agent (accuracy and safety check)
    raise NotImplementedError
