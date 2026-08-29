"""
DataLens AI — Evidence Layer

The agent never asserts a numerical claim that isn't backed by an Evidence
object sourced from a tool call (MASTER PROMPT #14, #20). This module
defines that structure and a running EvidenceChain the orchestrator
appends to during an investigation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Evidence:
    metric: str                     # e.g. "monthly_spending", "category_change"
    value: Any                      # the primary number/object of interest
    source_tool: str                # which tool produced this
    comparison: Optional[Any] = None
    change_percent: Optional[float] = None
    raw_result: Optional[dict] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class EvidenceChain:
    """Accumulates evidence across the steps of a single investigation."""

    def __init__(self):
        self._items: list[Evidence] = []
        self._trace: list[str] = []  # user-facing action trace (MASTER PROMPT #35)

    def add(self, evidence: Evidence) -> None:
        self._items.append(evidence)

    def add_trace(self, action: str) -> None:
        """Record a concise, user-facing step description — never hidden chain-of-thought."""
        self._trace.append(action)

    @property
    def items(self) -> list[Evidence]:
        return self._items

    @property
    def trace(self) -> list[str]:
        return self._trace

    def as_llm_context(self) -> list[dict]:
        """Compact evidence representation to feed back to the LLM for synthesis."""
        return [item.to_dict() for item in self._items]

    def is_empty(self) -> bool:
        return len(self._items) == 0
