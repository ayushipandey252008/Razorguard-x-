from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RuleResult:
    rule_id: str
    rule_name: str
    severity: str
    score_contribution: float
    explanation: str
    evidence: dict[str, Any] = field(default_factory=dict)
    triggered: bool = False


RuleFn = Callable[[dict, dict], RuleResult | None]


class RuleEngine:
    def __init__(self) -> None:
        self._rules: list[RuleFn] = []

    def register(self, fn: RuleFn) -> RuleFn:
        self._rules.append(fn)
        return fn

    def evaluate(self, txn: dict, context: dict) -> list[RuleResult]:
        fired: list[RuleResult] = []
        for fn in self._rules:
            result = fn(txn, context)
            if result and result.triggered:
                fired.append(result)
        return fired

    def aggregate_score(self, fired: list[RuleResult]) -> float:
        return float(min(100.0, sum(r.score_contribution for r in fired)))
