"""Shared types for the Track 1 multi-agent harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMCallMetrics:
    """Metrics accumulated across internal planner/subagent calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    cost: float = 0.0
    elapsed_ms: float = 0.0
    num_calls: int = 0

    def add(self, other: "LLMCallMetrics") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.thinking_tokens += other.thinking_tokens
        self.cost += other.cost
        self.elapsed_ms += other.elapsed_ms
        self.num_calls += other.num_calls

    @classmethod
    def from_litellm_response(cls, response: Any, elapsed_ms: float) -> "LLMCallMetrics":
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        thinking_tokens = 0
        details = getattr(usage, "completion_tokens_details", None)
        if details:
            thinking_tokens = getattr(details, "reasoning_tokens", 0) or 0
        cost = getattr(response, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=thinking_tokens,
            cost=cost,
            elapsed_ms=elapsed_ms,
            num_calls=1,
        )


@dataclass
class SubagentProposal:
    """A private subagent proposal before planner validation."""

    agent: str
    understood_intent: str = ""
    proposed_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    ask_user: str | None = None
    final_response: str | None = None
    required_facts: list[str] = field(default_factory=list)
    policy_risks: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerResult:
    """One benchmark-visible next action plus private accounting."""

    next_action: dict[str, Any]
    metrics: LLMCallMetrics = field(default_factory=LLMCallMetrics)
    internal_calls: int = 0
    debug: dict[str, Any] = field(default_factory=dict)
