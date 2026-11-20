"""An evaluation's results.json: the atomic write, and the `subtask_*` block.

Shared by `scripts/eval/eval_policy_client.py` and `scripts/eval/aggregate.py`.
Imports no torch, cohub or Isaac, so it runs under the system `python3`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def write_json_atomically(path: str | os.PathLike, payload: object) -> None:
    """Write `payload` to `path` through a sibling temp file and `os.replace`.

    A reader sees the whole old file or the whole new one, never a partial write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def flag_counts(episodes: list[dict], key: str = "subtasks") -> tuple[list[str], dict[str, int]]:
    """``(names in first-seen order, {name: episodes that achieved it})``.

    An episode without the key counts as achieving nothing.
    """
    names: list[str] = []
    for episode in episodes:
        for name in episode.get(key) or {}:
            if name not in names:
                names.append(name)
    counts = {
        name: sum(1 for e in episodes if (e.get(key) or {}).get(name)) for name in names
    }
    return names, counts


def subtask_summary(episodes: list[dict]) -> dict:
    """The `subtask_successes`/`subtask_success_rate` pair of a results.json,
    plus the `subtask_detail_*` pair when the task's tracker records details.
    """
    n = len(episodes)

    def rates(counts: dict[str, int]) -> dict[str, float | None]:
        return {name: round(count / n, 4) if n else None for name, count in counts.items()}

    _, counts = flag_counts(episodes)
    summary: dict = {"subtask_successes": counts, "subtask_success_rate": rates(counts)}
    _, details = flag_counts(episodes, key="subtask_details")
    if details:
        summary["subtask_detail_successes"] = details
        summary["subtask_detail_success_rate"] = rates(details)
    return summary
