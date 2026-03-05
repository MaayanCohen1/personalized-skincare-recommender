"""Structured output models for CrewAI task outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngredientEvidence(BaseModel):
    ingredient: str = Field(min_length=1)
    fact: str = Field(min_length=1)
    source: str = Field(min_length=1)


class ResearchOutput(BaseModel):
    items: list[IngredientEvidence]


class FinalExplanation(BaseModel):
    explanation_text: str = Field(min_length=1)
    sources: list[str] = Field(min_length=1)


class DraftExplanation(BaseModel):
    explanation_text: str = Field(min_length=1)
