#!/usr/bin/env python3
"""Fold per-episode result files into one run summary (results.json).

    eval/aggregate.py <results.json> <episodes dir>
    eval/aggregate.py --rescore-subtasks <results.json> [...]

Reads <episodes dir>/shard*/client_status.tsv (or a flat client_status.tsv) and
the epNNN.json files beside it. A status line is tab-separated: first episode
index, rc, wall seconds, chunk. An episode with no record is scored a failure.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


# Runs under a plain python3, so the package path is added by hand.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "source") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "source"))
from cohub.results import subtask_summary, write_json_atomically  # noqa: E402


def _status_files(ep_dir: Path) -> list[Path]:
    sharded = sorted(ep_dir.glob("shard*/client_status.tsv"))
    flat = ep_dir / "client_status.tsv"
    return sharded or ([flat] if flat.is_file() else [])


def _read_status(path: Path) -> dict[int, tuple[int, int, int]]:
    status: dict[int, tuple[int, int, int]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        chunk = int(parts[3]) if len(parts) > 3 else 1
        status[int(parts[0])] = (int(parts[1]), int(parts[2]), chunk)
    return status


def collect(ep_dir: Path) -> tuple[list[dict], dict | None]:
    """Return every episode the run was supposed to produce, in episode order.

    `meta` is the first payload found, with `seed` set to the run's base seed.
    """
    episodes: list[dict] = []
    meta: dict | None = None
    base_seed: int | None = None

    for status_path in _status_files(ep_dir):
        status = _read_status(status_path)
        for start in sorted(status):
            rc, wall, chunk = status[start]
            payload_path = status_path.parent / f"ep{start:03d}.json"
            written: list[dict] = []
            if payload_path.is_file():
                payload = json.loads(payload_path.read_text())
                if base_seed is None and isinstance(payload.get("seed"), int):
                    base_seed = payload["seed"] - start
                    meta = dict(payload, seed=base_seed)
                meta = meta or payload
                written = payload.get("episodes") or []
            # A process that died part way keeps the episodes it finished;
            # the rest of its chunk is a crash.
            for offset in range(chunk):
                if offset < len(written):
                    entry = dict(written[offset])
                    entry["episode"] = start + offset
                    entry["client_rc"] = rc
                    # Process wall time not spent in episodes is Isaac's boot.
                    if offset == 0:
                        rollout = sum(e.get("wall_s") or 0 for e in written)
                        entry["client_wall_s"] = wall
                        entry["boot_s"] = round(max(0.0, wall - rollout), 1)
                else:
                    entry = {
                        "success": False,
                        "reason": f"client_crash_rc{rc}",
                        "steps": None,
                        "episode": start + offset,
                        "seed": base_seed + start + offset if base_seed is not None else None,
                        "wall_s": None,
                        "client_rc": rc,
                    }
                episodes.append(entry)

    episodes.sort(key=lambda e: e["episode"])
    return episodes, meta


def _timing_totals(episodes: list[dict]) -> dict:
    """Return per-step timing totals over every episode that reported timing.

    `unaccounted` is rollout time that neither the env nor the client measured.
    """
    totals: dict[str, float] = {}
    rpc: dict[str, float] = {}
    rpc_n: dict[str, int] = {}
    steps = boot = rollout = 0
    for episode in episodes:
        timing = episode.get("timing") or {}
        for name, value in timing.items():
            if name == "rpc_s":
                for call, seconds in value.items():
                    rpc[call] = rpc.get(call, 0.0) + seconds
            elif name == "rpc_n":
                for call, count in value.items():
                    rpc_n[call] = rpc_n.get(call, 0) + count
            elif isinstance(value, (int, float)):
                totals[name] = totals.get(name, 0.0) + value
        steps += episode.get("steps") or 0
        boot += episode.get("boot_s") or 0
        rollout += episode.get("wall_s") or 0

    if not steps:
        return {}
    measured = sum(v for k, v in totals.items() if k != "boot_s") + sum(rpc.values())
    summary = {
        "steps_total": steps,
        "ms_per_step": {
            **{name: round(seconds / steps * 1000, 1)
               for name, seconds in sorted(totals.items()) if name != "boot_s"},
            **{f"rpc.{call}": round(seconds / steps * 1000, 1)
               for call, seconds in sorted(rpc.items())},
            "unaccounted": round(max(0.0, rollout - measured) / steps * 1000, 1),
            "TOTAL": round(rollout / steps * 1000, 1),
        },
        "rpc_calls": dict(sorted(rpc_n.items())),
        "boot_s_total": round(boot, 1),
        "rollout_s_total": round(rollout, 1),
    }
    return summary


def summarize(episodes: list[dict], meta: dict | None) -> dict:
    n = len(episodes)
    successes = sum(1 for e in episodes if e.get("success"))
    reasons = Counter(e["reason"] for e in episodes)
    steps = sorted(e["steps"] for e in episodes if e.get("steps") is not None)

    summary = {
        key: (meta or {}).get(key)
        for key in ("bench_name", "task_name", "env_cfg_type", "policy_name", "episode_step_limit", "seed")
    }
    summary.update(
        episodes_run=n,
        successes=successes,
        success_rate=round(successes / n, 4) if n else None,
        **subtask_summary(episodes),
        outcomes={r: {"n": c, "rate": round(c / n, 4)} for r, c in reasons.most_common()},
        steps_median=steps[len(steps) // 2] if steps else None,
        steps_min=steps[0] if steps else None,
        steps_max=steps[-1] if steps else None,
        # Hosts, seeded-episode count and the runner's provenance block.
        hosts=sorted({e["host"] for e in episodes if e.get("host")}),
        server_seeded_episodes=sum(1 for e in episodes if e.get("server_seeded")),
        provenance=(meta or {}).get("provenance"),
        timing=_timing_totals(episodes),
        final=True,
        episodes=episodes,
    )
    return summary



def rescore_subtasks(path: Path) -> str:
    """Recompute a finished run's subtask block in place and return what changed.

    Everything outside the block is left as it was. A file whose stored
    per-subtask counts disagree with its own episodes is skipped, and so is a
    run with no subtask chain.
    """
    summary = json.loads(path.read_text())
    episodes = summary.get("episodes") or []
    if not any(e.get("subtasks") for e in episodes):
        return "no subtask chain -- skipped"
    block = subtask_summary(episodes)
    stored = summary.get("subtask_successes")
    if stored is not None and stored != block["subtask_successes"]:
        return f"stored counts {stored} != recomputed {block['subtask_successes']} -- SKIPPED"

    refreshed: dict = {}
    for key, value in summary.items():
        if key in block or key.startswith("subtask_joint_"):
            continue  # re-emitted below in the block's order; subtask_joint_* keys are dropped
        if key == "outcomes":  # the block sits between success_rate and outcomes
            refreshed.update(block)
        refreshed[key] = value
    if not set(block) <= set(refreshed):
        refreshed.update(block)  # no outcomes key to anchor on; append instead
    if refreshed == summary:
        return "already current"
    write_json_atomically(path, refreshed)
    added = {k: v for k, v in block.items() if k not in summary}
    return f"updated {sorted(added) or 'subtask block'}"


def main(argv: list[str]) -> int:
    if len(argv) > 2 and argv[1] == "--rescore-subtasks":
        for name in argv[2:]:
            print(f"[rescore] {name}: {rescore_subtasks(Path(name))}")
        return 0
    if len(argv) != 3:
        print("usage: eval/aggregate.py <results.json> <episodes dir>", file=sys.stderr)
        print("       eval/aggregate.py --rescore-subtasks <results.json> [...]",
              file=sys.stderr)
        return 2
    results_path, ep_dir = Path(argv[1]), Path(argv[2])

    episodes, meta = collect(ep_dir)
    if not episodes:
        print(f"[aggregate] no client status under {ep_dir} -- nothing ran", file=sys.stderr)
        return 1

    summary = summarize(episodes, meta)
    write_json_atomically(results_path, summary)

    n = summary["episodes_run"]
    print(f"[aggregate] {summary['successes']}/{n} success -> {results_path}")
    for reason, count in Counter(e["reason"] for e in episodes).most_common():
        print(f"[aggregate]   {reason:24s} {count:3d}  {count / n:5.1%}")
    for name, count in summary["subtask_successes"].items():
        print(f"[aggregate]   subtask {name:16s} {count:3d}  {count / n:5.1%}")
    for name, count in summary.get("subtask_detail_successes", {}).items():
        print(f"[aggregate]   detail  {name:16s} {count:3d}  {count / n:5.1%}")
    budget = summary["timing"].get("ms_per_step")
    if budget:
        print(f"[aggregate] per control step over {summary['timing']['steps_total']} steps:")
        for name, ms in sorted(budget.items(), key=lambda kv: -kv[1]):
            print(f"[aggregate]   {name:24s} {ms:8.1f} ms")
        print(f"[aggregate]   Isaac boot, all clients {summary['timing']['boot_s_total']:8.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
