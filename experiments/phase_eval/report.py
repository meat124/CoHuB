from __future__ import annotations

import csv
import json
import math
import sys

from pathlib import Path

KINDS = ("collaborative", "non_collaborative")


def rate(successes: int, n: int) -> float | None:
    return round(successes / n, 4) if n else None


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


START_KINDS = ("initial", "lived", "restored")


def _by_start(valid: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for kind in START_KINDS:
        mine = [t for t in valid if (t.get("start_mode") or "restored") == kind]
        if mine:
            successes = sum(1 for t in mine if t.get("target_success"))
            out[kind] = {"starts": len(mine), "successes": successes, "rate": rate(successes, len(mine))}
    return out


def merge_by_start(into: dict[str, dict], other: dict[str, dict] | None) -> dict[str, dict]:
    for kind, cell in (other or {}).items():
        mine = into.setdefault(kind, {"starts": 0, "successes": 0, "rate": None})
        mine["starts"] += cell["starts"]
        mine["successes"] += cell["successes"]
        mine["rate"] = rate(mine["successes"], mine["starts"])
    return into


GROUPS = {True: "collaborative", False: "non_collaborative", None: "unlabelled"}


def summarize(trials: list[dict], chain: list[str], collaborative: dict[str, bool] | None = None,
              unstartable: list[dict] | None = None) -> dict:
    collaborative = collaborative or {}
    unstartable = unstartable or []
    rows: dict[str, dict] = {}
    for name in chain:
        mine = [t for t in trials if t.get("phase") == name]
        valid = [t for t in mine if t.get("start_valid")]
        successes = sum(1 for t in mine if t.get("target_success"))
        steps = sorted(t["steps_to_target"] for t in mine if t.get("target_success") and t.get("steps_to_target") is not None)
        reasons: dict[str, int] = {}
        for t in mine:
            if not t.get("target_success"):
                reason = str(t.get("reason"))
                reasons[reason] = reasons.get(reason, 0) + 1
        rows[name] = {
            "collaborative": collaborative.get(name),
            "trials": len(mine),
            "unstartable": sum(1 for m in unstartable if m.get("phase") == name),
            "valid_starts": len(valid),
            "lived_starts": sum(1 for t in valid if t.get("start_mode") == "lived"),
            "by_start": _by_start(valid),
            "prev_achieved_live": sum(1 for t in valid if t.get("prev_achieved_live", True)),
            "successes": successes,
            "rate": rate(successes, len(mine)),
            "rate_valid": rate(successes, len(valid)),
            "wilson95": wilson(successes, len(mine)),
            "step_budget": next((t.get("step_budget") for t in mine), None),
            "steps_to_target_median": steps[len(steps) // 2] if steps else None,
            "failure_reasons": dict(sorted(reasons.items(), key=lambda item: -item[1])),
        }

    by_label: dict[str, dict] = {}
    for name, row in rows.items():
        label = GROUPS[row["collaborative"]]
        pooled = by_label.setdefault(label, {"phases": [], "trials": 0, "valid_starts": 0, "successes": 0,
                                             "by_start": {}})
        pooled["phases"].append(name)
        for key in ("trials", "valid_starts", "successes"):
            pooled[key] += row[key]
        merge_by_start(pooled["by_start"], row["by_start"])
    for pooled in by_label.values():
        pooled["rate"] = rate(pooled["successes"], pooled["trials"])
        pooled["rate_valid"] = rate(pooled["successes"], pooled["valid_starts"])
        pooled["wilson95"] = wilson(pooled["successes"], pooled["trials"])
    return {"phase_starts": rows, "by_phase_kind": by_label}


def table(summary: dict) -> str:
    lines = [f"{'phase':<16}{'kind':<19}{'success':>9}{'valid':>7}{'rate':>8}{'rate_valid':>12}  wilson95"
             f"{'':4}by start"]
    for name, row in summary["phase_starts"].items():
        split = "  ".join(f"{kind} {cell['successes']}/{cell['starts']}" for kind, cell in (row.get("by_start") or {}).items())
        lines.append(
            f"{name:<16}{GROUPS[row['collaborative']]:<19}{row['successes']:>5}/{row['trials']:<3}{row['valid_starts']:>7}"
            f"{row['rate'] if row['rate'] is not None else float('nan'):>8.3f}"
            f"{row['rate_valid'] if row['rate_valid'] is not None else float('nan'):>12.3f}  {str(row['wilson95']):<20}{split}"
        )
    return "\n".join(lines)


SENSITIVE_P = 0.01

MIN_STARTS = 5


def fisher(lived: dict | None, restored: dict | None) -> float | None:
    from math import comb

    if not lived or not restored or min(lived["starts"], restored["starts"]) < MIN_STARTS:
        return None
    a, b = lived["successes"], lived["starts"] - lived["successes"]
    c, d = restored["successes"], restored["starts"] - restored["successes"]
    total, row, col = a + b + c + d, a + b, a + c

    def prob(x: int) -> float:
        return comb(col, x) * comb(total - col, row - x) / comb(total, row)

    observed = prob(a)
    return float(sum(prob(x) for x in range(max(0, row + col - total), min(row, col) + 1) if prob(x) <= observed * (1 + 1e-9)))


def load(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.glob("*/*/results.json")):
        try:
            payload = json.loads(path.read_text())
        except ValueError:
            print(f"[report] {path}: not valid JSON, skipped", file=sys.stderr)
            continue
        if payload.get("protocol") != "phase_starts":
            continue
        payload["_task"], payload["_run"], payload["_path"] = path.parts[-3], path.parts[-2], str(path.relative_to(root))
        runs.append(payload)
    return runs


def fold(runs: list[dict]) -> dict:
    rows, per_run, warnings = [], {}, []
    labels_seen: dict[str, tuple[str, dict]] = {}
    for run in runs:
        task, name = run["_task"], run["_run"]
        key = (run.get("labels") or {}).get("trials_digest")
        if key is None:
            warnings.append(f"{task}/{name}: no trials digest recorded; cannot tell which labels it was scored on")
        elif task in labels_seen and labels_seen[task][1] != key:
            warnings.append(f"{task}: {name} ({key}) and {labels_seen[task][0]} ({labels_seen[task][1]}) were scored "
                            "on different trials; their numbers are not comparable")
        if key is not None:
            labels_seen.setdefault(task, (name, key))
        if not run.get("final"):
            warnings.append(f"{task}/{name}: the run did not finish ({len(run.get('trials') or [])} trials so far)")
        bucket = per_run.setdefault(name, {"policy": run.get("policy_name"), "tasks": {}, "sensitive": [],
                                           **{kind: {"successes": 0, "trials": 0, "valid_starts": 0, "by_start": {}}
                                              for kind in KINDS}})
        bucket["tasks"][task] = {}
        for phase, row in (run.get("phase_starts") or {}).items():
            kind = GROUPS[row.get("collaborative")]
            split = row.get("by_start") or {}
            lived, restored = split.get("lived"), split.get("restored")
            rows.append({"run": name, "policy": run.get("policy_name"), "task": task, "phase": phase, "kind": kind,
                         "successes": row["successes"], "trials": row["trials"], "valid_starts": row["valid_starts"],
                         "unstartable": row.get("unstartable"), "lived_starts": row.get("lived_starts"),
                         "start": run.get("start"),
                         "lived_successes": lived and lived["successes"], "lived_held": lived and lived["starts"],
                         "restored_successes": restored and restored["successes"],
                         "restored_held": restored and restored["starts"],
                         "rate": row["rate"], "rate_valid": row["rate_valid"], "step_budget": row.get("step_budget"),
                         "final": bool(run.get("final"))})
            bucket["tasks"][task][phase] = {"kind": kind, "successes": row["successes"], "trials": row["trials"],
                                            "valid_starts": row["valid_starts"], "by_start": split}
            p_value = fisher(lived, restored)
            if p_value is not None and p_value < SENSITIVE_P:
                bucket["sensitive"].append({"task": task, "phase": phase, "kind": kind, "lived": lived,
                                            "restored": restored, "p": round(p_value, 4)})
            if kind in KINDS:
                for field in ("successes", "trials", "valid_starts"):
                    bucket[kind][field] += row[field]
                merge_by_start(bucket[kind]["by_start"], split)
    for bucket in per_run.values():
        for kind in KINDS:
            pooled = bucket[kind]
            pooled["rate"] = rate(pooled["successes"], pooled["trials"])
            pooled["rate_valid"] = rate(pooled["successes"], pooled["valid_starts"])
            pooled["wilson95"] = wilson(pooled["successes"], pooled["trials"])
        a, b = bucket["non_collaborative"]["rate_valid"], bucket["collaborative"]["rate_valid"]
        bucket["gap_non_minus_collab"] = None if a is None or b is None else round(a - b, 4)
    return {"runs": per_run, "rows": rows, "warnings": warnings}


def percent(value: float | None) -> str:
    return "  --" if value is None else f"{100 * value:5.1f}"


def phase_cell(cell: dict) -> str:
    text = f"{cell['successes']}/{cell['valid_starts']}"
    split = cell.get("by_start") or {}
    parts = [f"{tag} {split[kind]['successes']}/{split[kind]['starts']}"
             for tag, kind in (("L", "lived"), ("R", "restored")) if kind in split]
    return text + (f" ({', '.join(parts)})" if parts else "")


def markdown(summary: dict) -> str:
    out = ["# Phase-level evaluation", "",
           "Each phase is scored over the same trials, one per demonstration, every one started from that "
           "demonstration's completion of the phase before it. Rates are percentages of the starts "
           "that held.", "",
           "## Collaborative vs non-collaborative phases", "",
           "| run | tasks | non-collaborative | collaborative | gap (pp) |", "|---|---|---|---|---|"]
    for name, run in summary["runs"].items():
        non, col = run["non_collaborative"], run["collaborative"]
        gap = run["gap_non_minus_collab"]
        out.append(f"| {name} | {len(run['tasks'])} | {percent(non['rate_valid'])} ({non['successes']}/{non['valid_starts']}) "
                   f"| {percent(col['rate_valid'])} ({col['successes']}/{col['valid_starts']}) "
                   f"| {'--' if gap is None else f'{100 * gap:+.1f}'} |")
    names = sorted({task for run in summary["runs"].values() for task in run["tasks"]})
    out += ["", "## Per task", "", "Success rate over the phases of each kind, and the mean over the tasks that have that kind.", "",
            "| run | kind | " + " | ".join(names) + " | mean |", "|---|---|" + "---|" * (len(names) + 1)]
    for name, run in summary["runs"].items():
        for kind in KINDS:
            rates = []
            for task in names:
                cells = [c for c in run["tasks"].get(task, {}).values() if c["kind"] == kind]
                rates.append(rate(sum(c["successes"] for c in cells), sum(c["valid_starts"] for c in cells)))
            held = [r for r in rates if r is not None]
            mean = sum(held) / len(held) if held else None
            out.append(f"| {name} | {kind.replace('_', '-')} | " + " | ".join(percent(r) for r in rates) + f" | {percent(mean)} |")
    out += ["", "### By how the trial was started", "",
            "| run | kind | " + " | ".join(START_KINDS) + " |", "|---|---|" + "---|" * len(START_KINDS)]
    for name, run in summary["runs"].items():
        for kind in KINDS:
            split = run[kind]["by_start"]
            out.append(f"| {name} | {kind.replace('_', '-')} | " + " | ".join(
                f"{percent(split[k]['rate'])} ({split[k]['successes']}/{split[k]['starts']})" if k in split else "--"
                for k in START_KINDS) + " |")
    out += ["", "## Per phase", "",
            "`C` collaborative, `n` non-collaborative; successes / starts that held, then the same split by start "
            "(`L` lived, `R` restored; the first phase starts at the demo's own initial state).", ""]
    tasks: dict[str, list[str]] = {}
    for row in summary["rows"]:
        phases = tasks.setdefault(row["task"], [])
        if row["phase"] not in phases:
            phases.append(row["phase"])
    kind_of = {(row["task"], row["phase"]): row["kind"] for row in summary["rows"]}
    for task, phases in tasks.items():
        head = " | ".join(f"{s} ({'C' if kind_of[(task, s)] == 'collaborative' else 'n'})" for s in phases)
        out += [f"### {task}", "", f"| run | {head} |", "|---|" + "---|" * len(phases)]
        for name, run in summary["runs"].items():
            cells = run["tasks"].get(task)
            if not cells:
                continue
            out.append(f"| {name} | " + " | ".join(phase_cell(cells[s]) if s in cells else "--" for s in phases) + " |")
        out.append("")
    sensitive = [(name, item) for name, run in summary["runs"].items() for item in run.get("sensitive", [])]
    if sensitive:
        out += ["## Start sensitivity", "",
                f"Phases whose lived and restored starts differ (Fisher exact, p < {SENSITIVE_P}). Read these per "
                "kind of start, not pooled.", "", "| run | task | phase | lived | restored | p |", "|---|---|---|---|---|---|"]
        for name, item in sensitive:
            out.append(f"| {name} | {item['task']} | {item['phase']} ({'C' if item['kind'] == 'collaborative' else 'n'}) "
                       f"| {item['lived']['successes']}/{item['lived']['starts']} "
                       f"| {item['restored']['successes']}/{item['restored']['starts']} | {item['p']} |")
        out.append("")
    if summary["warnings"]:
        out += ["## Warnings", ""] + [f"- {w}" for w in summary["warnings"]] + [""]
    return "\n".join(out)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "eval_results_phase"
    if not root.is_dir():
        print(f"[report] no {root}: nothing has been evaluated yet", file=sys.stderr)
        return 1
    runs = load(root)
    if not runs:
        print(f"[report] no phase-start results.json under {root}", file=sys.stderr)
        return 1
    summary = fold(runs)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(root / "summary.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])
    text = markdown(summary)
    (root / "summary.md").write_text(text)
    print(text)
    print(f"[report] {len(runs)} run(s) -> {root}/summary.{{json,csv,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
