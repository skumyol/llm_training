import random
import os
from pathlib import Path
from typing import Optional

import yaml


SCENARIO_TYPES = [
    "secret_extraction",
    "apology_repair",
    "alliance_negotiation",
    "rumor_confrontation",
    "threat_escalation",
    "trust_building",
    "deception_detection",
]


class ScenarioBank:
    def __init__(self, bank_dir: str, seed: int = 42):
        self.bank_dir = Path(bank_dir)
        self.rng = random.Random(seed)
        self._scenarios: dict[str, list[dict]] = {}
        self._load_all()

    def _load_all(self) -> None:
        for scenario_type in SCENARIO_TYPES:
            path = self.bank_dir / f"{scenario_type}.yaml"
            if path.exists():
                with open(path, "r") as f:
                    self._scenarios[scenario_type] = yaml.safe_load(f) or []
            else:
                self._scenarios[scenario_type] = []

    def sample(
        self,
        n: int,
        scenario_type: Optional[str] = None,
        weights: Optional[dict[str, float]] = None,
    ) -> list[dict]:
        if scenario_type is not None:
            pool = self._scenarios.get(scenario_type, [])
            if not pool:
                raise ValueError(f"No scenarios found for type: {scenario_type}")
            return [self.rng.choice(pool) for _ in range(n)]

        if weights is None:
            weights = {t: 1.0 for t in SCENARIO_TYPES}

        types = [t for t in SCENARIO_TYPES if self._scenarios.get(t)]
        w = [weights.get(t, 1.0) for t in types]
        total = sum(w)
        w = [x / total for x in w]

        chosen_types = self.rng.choices(types, weights=w, k=n)
        results = []
        for t in chosen_types:
            pool = self._scenarios[t]
            results.append(dict(self.rng.choice(pool)))
        return results

    def all_scenarios(self, scenario_type: Optional[str] = None) -> list[dict]:
        if scenario_type:
            return list(self._scenarios.get(scenario_type, []))
        return [s for pool in self._scenarios.values() for s in pool]

    def stats(self) -> dict:
        return {t: len(v) for t, v in self._scenarios.items()}
