from collections import deque
from dataclasses import dataclass, field


@dataclass
class TemporalSmoother:
    window_size: int = 5
    alert_threshold: float = 0.25
    values: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.values = deque(maxlen=max(1, int(self.window_size)))

    def step(self, score: float) -> dict:
        self.values.append(float(score))
        smoothed = sum(self.values) / len(self.values)
        return {"smoothed": smoothed, "alert": smoothed > self.alert_threshold}

    def reset(self) -> None:
        self.values.clear()

    def update_threshold(self, alert_threshold: float) -> None:
        self.alert_threshold = float(alert_threshold)
