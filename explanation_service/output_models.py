"""Structured output models for CrewAI task outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class IngredientEvidence(BaseModel):
    ingredient: str = Field(min_length=1)
    fact: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def coerce_legacy_source_field(cls, data: Any) -> Any:
        """Accept legacy LLM JSON with a single ``source`` string (optionally ``;``-joined)."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "source" in out and "sources" in out:
            out.pop("source", None)
        elif "sources" not in out and "source" in out:
            raw = out.pop("source", None)
            s = str(raw or "").strip()
            parts = [p.strip() for p in s.split(";") if p.strip()]
            out["sources"] = parts if parts else ["generic"]
        return out


class ResearchOutput(BaseModel):
    items: list[IngredientEvidence]


class FinalExplanation(BaseModel):
    explanation_text: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)


class DraftExplanation(BaseModel):
    explanation_text: str = Field(min_length=1)
