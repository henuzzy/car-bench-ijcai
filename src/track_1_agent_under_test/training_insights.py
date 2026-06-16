"""Training-set-derived generic planning hints.

This module intentionally extracts only abstract tool names and coarse intent
patterns from the public training set.  It does not expose task ids, concrete
location ids, route ids, contact details, or expected argument values at
runtime, so it remains a general prior rather than a task-answer lookup table.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sunroof": ("sunroof", "sunshade", "fresh air"),
    "fog_lights": ("fog light", "fog lights", "reduced visibility"),
    "defrost": ("defrost", "fog up", "foggy", "windshield", "mist"),
    "climate": (
        "air conditioning",
        "climate",
        "fan",
        "temperature",
        "cool",
        "warm",
        "stuffy",
        "air circulation",
    ),
    "navigation_poi": (
        "restaurant",
        "charging station",
        "charger",
        "parking",
        "supermarket",
        "bakery",
        "airport",
        "toilet",
        "destination",
    ),
    "navigation_edit": (
        "waypoint",
        "final destination",
        "replace",
        "remove",
        "delete",
        "cancel navigation",
        "current destination",
    ),
    "email": ("email", "mail", "attendees", "send message"),
    "calendar": ("calendar", "meeting", "appointment", "schedule"),
    "charging": ("charge", "charging", "battery", "soc", "range"),
    "lights": ("headlight", "headlights", "high beam", "low beam"),
    "phone": ("call", "phone", "telephone"),
}

READ_ONLY_TOOL_PREFIXES = (
    "get_",
    "search_",
    "calculate_",
)

DEFAULT_MAX_RECIPES = 5
DEFAULT_MAX_NEXT_TOOLS = 8


@dataclass(frozen=True)
class TrainingRecipe:
    domains: tuple[str, ...]
    sequence: tuple[str, ...]
    support: int

    def to_dict(self, completed_tools: set[str], available_tools: set[str]) -> dict[str, Any]:
        next_tool = _first_missing_available(self.sequence, completed_tools, available_tools)
        return {
            "domains": list(self.domains),
            "support": self.support,
            "tool_sequence": [tool for tool in self.sequence if tool in available_tools],
            "next_tool": next_tool,
        }


class TrainingInsightStore:
    """Compact generic priors derived from public base training tasks."""

    def __init__(
        self,
        *,
        recipes_by_domain: dict[str, list[TrainingRecipe]],
        transitions: dict[str, Counter[str]],
        prerequisites: dict[str, Counter[str]],
        loaded_rows: int,
        source: str,
    ) -> None:
        self.recipes_by_domain = recipes_by_domain
        self.transitions = transitions
        self.prerequisites = prerequisites
        self.loaded_rows = loaded_rows
        self.source = source

    @classmethod
    def from_base_train(cls, path: Path | None = None) -> "TrainingInsightStore":
        path = path or _default_base_train_path()
        if not path.exists():
            return cls(
                recipes_by_domain={},
                transitions={},
                prerequisites={},
                loaded_rows=0,
                source=str(path),
            )

        sequence_counts: Counter[tuple[str, ...]] = Counter()
        sequence_domains: dict[tuple[str, ...], set[str]] = defaultdict(set)
        transitions: dict[str, Counter[str]] = defaultdict(Counter)
        prerequisites: dict[str, Counter[str]] = defaultdict(Counter)
        loaded_rows = 0

        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                actions = json.loads(row.get("actions") or "[]")
            except (json.JSONDecodeError, TypeError):
                continue
            sequence = tuple(
                str(action.get("name") or "")
                for action in actions
                if isinstance(action, dict) and action.get("name")
            )
            if not sequence:
                continue
            loaded_rows += 1
            domains = _infer_domains_from_text(str(row.get("instruction") or ""))
            sequence_counts[sequence] += 1
            for domain in domains:
                sequence_domains[sequence].add(domain)
            for before, after in zip(sequence, sequence[1:]):
                transitions[before][after] += 1
            for index, tool_name in enumerate(sequence):
                if _is_read_only_tool(tool_name):
                    continue
                for prereq in sequence[:index]:
                    if _is_read_only_tool(prereq):
                        prerequisites[tool_name][prereq] += 1

        recipes_by_domain: dict[str, list[TrainingRecipe]] = defaultdict(list)
        for sequence, support in sequence_counts.items():
            domains = tuple(sorted(sequence_domains.get(sequence) or {"general"}))
            recipe = TrainingRecipe(domains=domains, sequence=sequence, support=support)
            for domain in domains:
                recipes_by_domain[domain].append(recipe)

        for domain, recipes in recipes_by_domain.items():
            recipes_by_domain[domain] = sorted(
                recipes,
                key=lambda recipe: (-recipe.support, len(recipe.sequence), recipe.sequence),
            )
        return cls(
            recipes_by_domain=dict(recipes_by_domain),
            transitions=dict(transitions),
            prerequisites=dict(prerequisites),
            loaded_rows=loaded_rows,
            source=str(path),
        )

    def hints_for(
        self,
        *,
        user_text: str,
        completed_tools: list[str],
        available_tools: set[str],
        max_recipes: int = DEFAULT_MAX_RECIPES,
        max_next_tools: int = DEFAULT_MAX_NEXT_TOOLS,
    ) -> dict[str, Any]:
        if not self.loaded_rows:
            return {"source": self.source, "loaded_rows": 0, "hints": []}

        completed = set(completed_tools)
        domains = _infer_domains_from_text(user_text)
        recipes = self._matching_recipes(domains, available_tools, max_recipes)
        recipe_dicts = [
            recipe.to_dict(completed_tools=completed, available_tools=available_tools)
            for recipe in recipes
        ]
        next_tools = self._suggest_next_tools(
            recipes=recipes,
            completed_tools=completed,
            available_tools=available_tools,
            max_next_tools=max_next_tools,
        )
        policy_hints = self._policy_hints(
            domains=domains,
            completed_tools=completed,
            available_tools=available_tools,
        )
        transition_hints = self._transition_hints(
            completed_tools=completed_tools,
            available_tools=available_tools,
        )
        return {
            "source": "public_base_train_abstract_tool_sequences",
            "loaded_rows": self.loaded_rows,
            "matched_domains": domains,
            "suggested_next_tools": next_tools,
            "observed_recipes": recipe_dicts,
            "transition_hints": transition_hints,
            "policy_hints": policy_hints,
            "usage_limits": [
                "Use these as generic priors only.",
                "Do not infer or copy concrete ids, arguments, route choices, contacts, or task ids from training data.",
            ],
        }

    def _matching_recipes(
        self,
        domains: list[str],
        available_tools: set[str],
        max_recipes: int,
    ) -> list[TrainingRecipe]:
        scored: list[tuple[int, TrainingRecipe]] = []
        for domain in domains or ["general"]:
            for recipe in self.recipes_by_domain.get(domain, []):
                available_count = sum(1 for tool in recipe.sequence if tool in available_tools)
                if available_count == 0:
                    continue
                scored.append((available_count, recipe))
        unique: dict[tuple[str, ...], tuple[int, TrainingRecipe]] = {}
        for available_count, recipe in scored:
            key = recipe.sequence
            current = unique.get(key)
            if current is None or (recipe.support, available_count) > (current[1].support, current[0]):
                unique[key] = (available_count, recipe)
        ranked = sorted(
            unique.values(),
            key=lambda item: (-item[1].support, -item[0], len(item[1].sequence), item[1].sequence),
        )
        return [recipe for _, recipe in ranked[:max_recipes]]

    def _suggest_next_tools(
        self,
        *,
        recipes: list[TrainingRecipe],
        completed_tools: set[str],
        available_tools: set[str],
        max_next_tools: int,
    ) -> list[str]:
        suggestions: list[str] = []
        for recipe in recipes:
            next_tool = _first_missing_available(recipe.sequence, completed_tools, available_tools)
            if next_tool and next_tool not in suggestions:
                suggestions.append(next_tool)
        for tool_name in reversed([tool for tool in completed_tools if tool in self.transitions]):
            for next_tool, _ in self.transitions[tool_name].most_common():
                if next_tool in available_tools and next_tool not in completed_tools and next_tool not in suggestions:
                    suggestions.append(next_tool)
                if len(suggestions) >= max_next_tools:
                    return suggestions
        return suggestions[:max_next_tools]

    def _transition_hints(
        self,
        *,
        completed_tools: list[str],
        available_tools: set[str],
    ) -> list[str]:
        hints: list[str] = []
        for tool_name in reversed(completed_tools[-3:]):
            next_tools = [
                name
                for name, _ in self.transitions.get(tool_name, Counter()).most_common(3)
                if name in available_tools
            ]
            if next_tools:
                hints.append(f"After {tool_name}, training sequences commonly continue with: {', '.join(next_tools)}.")
        return hints

    def _policy_hints(
        self,
        *,
        domains: list[str],
        completed_tools: set[str],
        available_tools: set[str],
    ) -> list[str]:
        hints: list[str] = []
        if "sunroof" in domains and "open_close_sunroof" in available_tools:
            if "get_weather" in available_tools and "get_weather" not in completed_tools:
                hints.append("Sunroof recipes gather weather before opening the sunroof.")
            if "open_close_sunshade" in available_tools:
                hints.append("Sunroof recipes open the sunshade fully before opening the sunroof when needed.")
        if "fog_lights" in domains:
            if "get_weather" in available_tools and "get_weather" not in completed_tools:
                hints.append("Fog-light recipes gather weather before changing exterior lights.")
            if "get_exterior_lights_status" in available_tools and "get_exterior_lights_status" not in completed_tools:
                hints.append("Exterior-light recipes inspect current light status before changing high/low/fog beams.")
        if "email" in domains and "send_email" in available_tools:
            if "get_contact_information" in available_tools and "get_contact_information" not in completed_tools:
                hints.append("Email recipes gather contact information before send_email.")
            if "calendar" in domains and "get_entries_from_calendar" in available_tools and "get_entries_from_calendar" not in completed_tools:
                hints.append("Meeting email recipes gather calendar entries before composing the email.")
        if "charging" in domains and "get_charging_specs_and_status" in available_tools and "get_charging_specs_and_status" not in completed_tools:
            hints.append("Charging recipes gather charging specs/status before route, charger, or charging-time calculations.")
        if "navigation_poi" in domains:
            hints.append("POI navigation recipes resolve city/location id, search POI, then route to the selected POI id before navigation changes.")
        if "navigation_edit" in domains and "get_current_navigation_state" in available_tools and "get_current_navigation_state" not in completed_tools:
            hints.append("Navigation edit recipes inspect current navigation state before replacing, adding, or deleting destinations.")
        return hints


@lru_cache(maxsize=1)
def default_training_insights() -> TrainingInsightStore:
    return TrainingInsightStore.from_base_train()


def _default_base_train_path() -> Path:
    return Path(__file__).resolve().parents[2] / "car-bench-dataset" / "tasks" / "base_train.jsonl"


def _infer_domains_from_text(text: str) -> list[str]:
    lowered = f" {_normalize_text(text)} "
    domains = [
        domain
        for domain, keywords in DOMAIN_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    if "navigation_edit" in domains and "navigation_poi" not in domains:
        if any(piece in lowered for piece in ("restaurant", "charging station", "charger", "parking", "destination")):
            domains.append("navigation_poi")
    return domains or ["general"]


def _first_missing_available(
    sequence: tuple[str, ...],
    completed_tools: set[str],
    available_tools: set[str],
) -> str | None:
    for tool_name in sequence:
        if tool_name in available_tools and tool_name not in completed_tools:
            return tool_name
    return None


def _is_read_only_tool(tool_name: str) -> bool:
    return tool_name.startswith(READ_ONLY_TOOL_PREFIXES)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
