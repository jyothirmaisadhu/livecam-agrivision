from __future__ import annotations

from collections import Counter, deque


class MajorityVote:
    """Stabilise per-frame labels: report the most common label of the last `window` frames,
    and only once it holds a strict majority."""

    def __init__(self, window: int = 5):
        self.history: deque[str | None] = deque(maxlen=window)

    def update(self, label: str | None) -> str | None:
        self.history.append(label)
        winner, count = Counter(self.history).most_common(1)[0]
        return winner if count > len(self.history) // 2 else None
