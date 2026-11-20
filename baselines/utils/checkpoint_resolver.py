"""Checkpoint-directory resolution shared by all policy adapters.

Precedence, highest first:
  1. an explicit path key in the deploy config (``model_path``, ``checkpoint_path``, ...)
  2. ``ckpt_name`` given as a path; relative paths resolve against the policy directory
  3. ``checkpoints/{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}-{seed}``
  4. ``checkpoints/<ckpt_name>``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# Deploy-config keys that carry an explicit checkpoint path.
DEFAULT_EXPLICIT_KEYS = (
    "model_path",
    "checkpoint_path",
    "ckpt_path",
    "model_dir",
    "pretrained_path",
)


def _clean(value: Any) -> Optional[str]:
    """Return a stripped non-empty string, or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def ckpt_name_is_path(ckpt_name: Any) -> bool:
    """True when ``ckpt_name`` is absolute or contains a path separator."""
    text = _clean(ckpt_name)
    if text is None:
        return False
    expanded = os.path.expanduser(text)
    if Path(expanded).is_absolute():
        return True
    if "/" in text or os.sep in text:
        return True
    return bool(os.altsep and os.altsep in text)


def build_run_dir_name(model_cfg: dict, *, include_seed: bool = True) -> Optional[str]:
    """Build ``{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}[-{seed}]``.

    Returns None when a component is missing. ``bench_name`` falls back to
    ``dataset_name``.
    """
    bench = _clean(model_cfg.get("bench_name") or model_cfg.get("dataset_name"))
    ckpt = _clean(model_cfg.get("ckpt_name"))
    env_cfg_type = _clean(model_cfg.get("env_cfg_type"))
    action_type = _clean(model_cfg.get("action_type"))
    parts = [bench, ckpt, env_cfg_type, action_type]
    if include_seed:
        parts.append(_clean(model_cfg.get("seed")))
    if any(part is None for part in parts):
        return None
    return "-".join(parts)


def _as_path(value: Any, base_dir: Optional[Path]) -> Path:
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path


def candidate_checkpoint_roots(
    model_cfg: dict,
    checkpoints_dir: os.PathLike | str,
    *,
    policy_dir: os.PathLike | str | None = None,
    explicit_keys: tuple[str, ...] = DEFAULT_EXPLICIT_KEYS,
    include_seed: bool = True,
    resolve: bool = True,
) -> list[Path]:
    """Return ordered, de-duplicated candidate checkpoint roots.

    ``policy_dir`` is the base for relative paths. It defaults to the parent
    of ``checkpoints_dir``.
    """
    checkpoints_dir = Path(checkpoints_dir)
    base_dir = Path(policy_dir) if policy_dir is not None else checkpoints_dir.parent

    roots: list[Path] = []
    seen: set[str] = set()

    def add(path: os.PathLike | str) -> None:
        candidate = Path(path)
        if resolve:
            try:
                candidate = candidate.resolve()
            except OSError:
                pass
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            roots.append(candidate)

    # 1. explicit path keys
    for key in explicit_keys:
        value = _clean(model_cfg.get(key))
        if value:
            add(_as_path(value, base_dir))

    ckpt_name = _clean(model_cfg.get("ckpt_name"))
    is_path = ckpt_name is not None and ckpt_name_is_path(ckpt_name)

    # 2. ckpt_name given as a path
    if is_path:
        add(_as_path(ckpt_name, base_dir))
    else:
        # 3. run-dir name built from the eval args
        run_dir_name = build_run_dir_name(model_cfg, include_seed=include_seed)
        if run_dir_name:
            add(checkpoints_dir / run_dir_name)

        # 4. checkpoints/<ckpt_name>
        if ckpt_name:
            add(checkpoints_dir / ckpt_name)

    return roots


def resolve_checkpoint_root(
    model_cfg: dict,
    checkpoints_dir: os.PathLike | str,
    *,
    policy_dir: os.PathLike | str | None = None,
    explicit_keys: tuple[str, ...] = DEFAULT_EXPLICIT_KEYS,
    include_seed: bool = True,
    must_exist: bool = True,
) -> Path:
    """Return the first existing candidate checkpoint root.

    When nothing exists: raises ``FileNotFoundError`` if ``must_exist``, else
    returns the highest-priority candidate.
    """
    candidates = candidate_checkpoint_roots(
        model_cfg,
        checkpoints_dir,
        policy_dir=policy_dir,
        explicit_keys=explicit_keys,
        include_seed=include_seed,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not must_exist and candidates:
        return candidates[0]
    checked = "\n  ".join(str(path) for path in candidates) or "  <no checkpoint candidates>"
    raise FileNotFoundError(
        "Could not resolve a checkpoint directory. Pass a valid ckpt_name "
        "(full run-dir name or a path), or set an explicit checkpoint path in "
        f"deploy.yml ({'/'.join(explicit_keys)}).\nChecked:\n  {checked}"
    )
