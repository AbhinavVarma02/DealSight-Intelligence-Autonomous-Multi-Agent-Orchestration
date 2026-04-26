"""Generic evaluation harness for any pricing predictor.

Pass in a callable that maps a prompt string to a price and call
`evaluate(items)` to get back mean absolute error, RMSLE, and a hit rate
(prediction within $40 or 20% of the true price).
"""

from __future__ import annotations

import math
from collections.abc import Callable


class Tester:
    def __init__(self, predictor: Callable[[str], float]) -> None:
        self.predictor = predictor

    def evaluate(self, items) -> dict[str, float]:
        errors = []
        squared_log_errors = []
        hits = 0
        for item in items:
            prediction = max(0.0, float(self.predictor(item.test_prompt)))
            actual = max(0.0, float(item.price))
            error = abs(prediction - actual)
            errors.append(error)
            squared_log_errors.append((math.log1p(prediction) - math.log1p(actual)) ** 2)
            if error < 40 or (actual and error / actual < 0.2):
                hits += 1
        count = len(errors) or 1
        return {
            "mean_absolute_error": sum(errors) / count,
            "rmsle": math.sqrt(sum(squared_log_errors) / count),
            "hit_rate": hits / count,
        }
