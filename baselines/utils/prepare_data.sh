#!/usr/bin/env bash
# Build what a shared policy trains on from the CoHuB dataset on the Hub.
#
#   baselines/policy/<POLICY>/train_hook.sh data               (= bash baselines/utils/prepare_data.sh <view>)
#   bash baselines/utils/prepare_data.sh <view>    view: gr00t | pi05 | psi0 | fastwam | all
#   bash baselines/utils/prepare_data.sh fetch     steps 1-2 only, for the per-task policies
#                                                  (ACT, DP, LatentToM, GauDP)
#
#   1. datasets/_hf/<repo>/      the Hub snapshot (COHUB_DATA_REPO), downloaded if absent and
#                                resumed if cut off.
#                                COHUB_DATA_DOWNLOAD=0 never touches the network.
#   2. datasets/<task>/lerobot   links into it. An export whose tasks.jsonl differs from
#                                cohub.task_registry is refused.
#   3. baselines/data/multitask  the flattened all-task dataset (every episode once per robot)
#                                and the requested view(s) of it
#   4. pi0.5 normalisation stats when the pi05 view is built
set -euo pipefail
VIEW=${1:?usage: prepare_data.sh <gr00t|pi05|psi0|fastwam|all|fetch>}
case "$VIEW" in gr00t|pi05|fastwam|psi0|all|fetch) ;; *) echo "unknown view $VIEW" >&2; exit 2 ;; esac
WT=${COHUB_WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}; export COHUB_WT=$WT; cd "$WT"
# shellcheck source=../common.sh
source "$WT/baselines/common.sh"
export PATH="$HOME/.local/bin:$WT/baselines/bin:$PATH"
POLICIES=$WT/baselines
SIM_PY=${COHUB_SIM_PY:-$WT/.venv/bin/python}
[ -x "$SIM_PY" ] || { echo "no simulation python at $SIM_PY (README, Installation) -- or set COHUB_SIM_PY" >&2; exit 1; }
# COHUB_TASK_SET: `all` is the two-robot task set, `all3` the three-robot set.
TASK_SET=${COHUB_TASK_SET:-all}
case "$TASK_SET" in
  all)  TASKS=$MH_ALL_TASKS;  MULTI=multitask;  DEFAULT_REPO=meat000124/CoHuB-2-agents ;;
  all3) TASKS=$MH_ALL3_TASKS; MULTI=multitask3; DEFAULT_REPO=meat000124/CoHuB-3-agents ;;
  *) echo "unknown COHUB_TASK_SET=$TASK_SET (all|all3)" >&2; exit 2 ;;
esac
REPO=${COHUB_DATA_REPO:-$DEFAULT_REPO}; DL=$WT/datasets/_hf/${REPO##*/}
REVISION=${COHUB_DATA_REVISION:-}
echo "[data] host=$(hostname) view=$VIEW repo=$REPO revision=${REVISION:-latest} set=$TASK_SET -> $MULTI"

# ---- 1. the Hub snapshot ----
REV_FILE=$DL/.cohub_revision   # written by a finished download: "<revision asked for> <commit>"
download() {                   # fetch the snapshot, resuming one that was cut off
  echo "[data] 1. download"
  # Plain HTTP, two workers.
  HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 "$SIM_PY" - "$REPO" "$DL" "$REVISION" "$REV_FILE" "$DEFAULT_REPO" <<'PY'
import sys, time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
repo, dest, revision, rev_file, default_repo = sys.argv[1:]
for attempt in range(1, 9):
    try:
        commit = HfApi().dataset_info(repo, revision=revision or None).sha
        snapshot_download(repo, repo_type="dataset", revision=commit, local_dir=dest, max_workers=2)
        Path(rev_file).write_text(f"{revision or 'latest'} {commit}\n")
        print(f"downloaded {repo}@{commit} to {dest}"); break
    except Exception as error:  # noqa: BLE001
        status = getattr(getattr(error, "response", None), "status_code", None)
        if isinstance(error, HfHubHTTPError) and status in (401, 403, 404):
            # Private, gated or not there: waiting does not change it.
            if repo == default_repo:
                raise SystemExit(f"the CoHuB dataset ({repo}) is not released yet. It is coming soon; "
                                 f"see https://meat124.github.io/CoHuB/. To use another dataset, "
                                 f"name it with COHUB_DATA_REPO")
            raise SystemExit(f"cannot read the dataset {repo}{'@' + revision if revision else ''} "
                             f"({type(error).__name__}, HTTP {status}): log in (huggingface-cli login), "
                             f"or name another with COHUB_DATA_REPO / COHUB_DATA_REVISION")
        wait = min(120 * attempt, 900); print(f"attempt {attempt} failed: {type(error).__name__}: {str(error)[:200]}; retrying in {wait}s", flush=True); time.sleep(wait)
else:
    raise SystemExit("snapshot_download kept failing")
PY
}
recorded=$(cat "$REV_FILE" 2>/dev/null || true)
downloaded=0
need=0; [ -f "$DL/README.md" ] || need=1
if [ -n "$REVISION" ]; then
  if [ -z "$recorded" ]; then need=1   # not known to be that revision
  elif [ "${recorded%% *}" != "$REVISION" ] && [[ "${recorded##* }" != "$REVISION"* ]]; then
    echo "[data] the snapshot at $DL is revision ${recorded##* }, not $REVISION; remove it to download that one" >&2
    exit 1
  fi
fi
if [ "$need" = 1 ] && [ "${COHUB_DATA_DOWNLOAD:-1}" = 1 ]; then
  download; downloaded=1
elif [ "$need" = 1 ] && [ -n "$REVISION" ] && [ -f "$DL/README.md" ]; then
  echo "[data] the snapshot at $DL does not record its revision and COHUB_DATA_DOWNLOAD=0; cannot tell it is $REVISION" >&2
  exit 1
else
  echo "[data] 1. download skipped ($DL${recorded:+ @ ${recorded##* }} present, or COHUB_DATA_DOWNLOAD=0)"
fi

# ---- 2. datasets/<task>/lerobot -> the snapshot ----
link_exports() {              # exit 3: the snapshot is not all there
"$SIM_PY" - "$DL" "$WT" "$TASKS" <<'PY'
#!/usr/bin/env python3
"""Link every LeRobot export of the download as datasets/<task>/lerobot.

    bash baselines/utils/prepare_data.sh <view>

Each export must hold 60 episodes split train 0:50 / val 50:60, and its
meta/tasks.jsonl must carry the sentences cohub.task_registry sends at eval time.
An existing datasets/<task>/lerobot directory is moved aside; a symlink is replaced.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT / "source"))

from cohub import task_registry  # noqa: E402
EXPECTED = tuple(sys.argv[3].split())
"""The tasks this build needs, from the caller (MH_ALL_TASKS or MH_ALL3_TASKS)."""


def slug_of(name: str) -> str | None:
    """A Hub directory name -> the benchmark's task word, if it is one."""
    word = re.sub(r"[^a-z0-9]", "", name.lower())
    aliases = {"framehang": "framehang", "frame_hang": "framehang",
               "handovereasy": "handover", "trash": "trashcollection"}
    for slug in task_registry.slugs():
        if word == slug or word == slug.replace("_", ""):
            return slug
    return aliases.get(word)



def gym_id_of(slug: str) -> str:
    for gym_id, row in task_registry.TASKS.items():
        if row["slug"] == slug:
            return gym_id
    raise SystemExit(f"{slug}: not in cohub.task_registry")


def check_sentences(export: Path, slug: str) -> None:
    rows = [json.loads(line) for line in (export / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip()]
    by_index = {int(r["task_index"]): r["task"] for r in rows}
    want = task_registry.sentences(gym_id_of(slug))
    if want is None:
        raise SystemExit(f"{slug}: no sentences in cohub.task_registry")
    got = tuple(by_index.get(i) for i in range(len(want)))
    if got != tuple(want):
        for i, (g, w) in enumerate(zip(got, want)):
            if g != w:
                print(f"  row {i}\n    export: {g!r}\n    wire  : {w!r}")
        raise SystemExit(f"{slug}: meta/tasks.jsonl does not carry the sentences eval sends "
                         f"(cohub.task_registry) -- refusing to link it")


def incomplete(message: str):
    """The snapshot is not all there: exit 3, which the caller answers by resuming the download."""
    print(message, file=sys.stderr)
    raise SystemExit(3)


def ensure_complete(export: Path, info: dict) -> None:
    """Refuse an export whose parquet or mp4 files are not all on disk."""
    parquets = len(list((export / "data").rglob("*.parquet")))
    mp4s = len(list((export / "videos").rglob("*.mp4"))) if (export / "videos").is_dir() else 0
    want_ep, want_mp4 = int(info.get("total_episodes", 0)), int(info.get("total_videos", 0))
    if parquets != want_ep or mp4s < want_mp4:
        incomplete(f"{export.name}: {parquets}/{want_ep} parquet files and {mp4s}/{want_mp4} videos on disk -- "
                   f"the snapshot is incomplete")


def ensure_splits(export: Path) -> dict:
    info_path = export / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    n = int(info["total_episodes"])
    want = {"train": "0:50", "val": "50:60"}
    if n != 60:
        raise SystemExit(f"{export}: {n} episodes, expected 60 (50 train + 10 val)")
    splits = info.get("splits")
    if not splits:
        info["splits"] = want
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        print(f"  wrote splits {want} into {info_path.relative_to(ROOT)}")
    elif {k: str(v) for k, v in splits.items()} != want:
        raise SystemExit(f"{export}: declares splits {splits}, the benchmark's are {want}")
    return info


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    download = Path(sys.argv[1]).resolve()
    exports = sorted(p.parent.parent for p in download.rglob("meta/info.json"))
    if not exports:
        incomplete(f"no LeRobot export (meta/info.json) under {download}")
    linked = {}
    for export in exports:
        slug = slug_of(export.name)
        if slug is None:
            print(f"  skipping {export.relative_to(download)}: not a benchmark task name")
            continue
        info = ensure_splits(export)
        ensure_complete(export, info)
        check_sentences(export, slug)
        dest = ROOT / "datasets" / slug / "lerobot"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            # missing_ok: two builds may relink the same task at once.
            dest.unlink(missing_ok=True)
        elif dest.exists():
            aside = dest.with_name(f"lerobot._pre_cohub_{time.strftime('%Y%m%d')}")
            print(f"  moving the previous {dest.relative_to(ROOT)} aside as {aside.name}")
            os.rename(dest, aside)
        dest.symlink_to(export)
        linked[slug] = export
        print(f"  {slug:16s} <- {export.relative_to(download)}  "
              f"({info['total_episodes']} ep, {info['total_frames']} frames, {info.get('codebase_version')})")
    missing = [s for s in EXPECTED if s not in linked]
    if missing:
        incomplete(f"download holds no export for: {missing}")
    print(f"[cohub] linked {len(linked)} task(s) into datasets/<task>/lerobot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
}
echo "[data] 2. link"
rc=0; link_exports || rc=$?
if [ "$rc" = 3 ] && [ "$downloaded" = 0 ] && [ "${COHUB_DATA_DOWNLOAD:-1}" = 1 ]; then
  echo "[data] the snapshot at $DL is incomplete -- resuming the download"
  download; rc=0; link_exports || rc=$?
fi
[ "$rc" = 0 ] || { [ "$rc" != 3 ] || echo "[data] the snapshot at $DL is incomplete (COHUB_DATA_DOWNLOAD=1 resumes it)" >&2; exit 1; }
for task in $TASKS; do [ -f "$WT/datasets/$task/lerobot/meta/info.json" ] || { echo "missing datasets/$task/lerobot" >&2; exit 1; }; done

[ "$VIEW" != fetch ] || { echo "[data] fetch: done (datasets/<task>/lerobot linked for: $TASKS)"; exit 0; }

# ---- 3. the flattened dataset and its views ----
echo "[data] 3. multitask views: $VIEW"
# The flattened dataset is shared by every view. It, the views and their stats
# are reused if present; COHUB_DATA_REBUILD=1 rebuilds them. Do not run two
# view builds at once.
export MULTITASK_REBUILD=${COHUB_DATA_REBUILD:-0}
BUILT=0                       # 1 once a dataset was built in this run: what derives from it follows
PI05_FRESH=0
WHAT=$VIEW
PY=${COHUB_CONVERT_PY:-$WT/.venv/bin/python}
POLICIES=$WT/baselines
build() {  # <split> <out dir>
  if [ -f "$2/meta/info.json" ] && [ "${MULTITASK_REBUILD:-0}" != 1 ]; then
    echo "  reusing ${2#$WT/} (MULTITASK_REBUILD=1 to rebuild)"
    return
  fi
  "$PY" "$WT/scripts/export/convert_lerobot.py" --format multitask --split "$1" --out "$2" --tasks "${TASKS// /,}" --overwrite
  BUILT=1
}
ENV_CFG=$(mh_env_cfg_type decentralized)
DATA_ID=$(mh_data_id "$MULTI" "$ENV_CFG")

[ -x "$PY" ] || { echo "no converter python at $PY -- set COHUB_CONVERT_PY" >&2; exit 1; }
for task in $TASKS; do
  [ -f "$WT/datasets/$task/lerobot/meta/info.json" ] || {
    echo "missing LeRobot export for $task -- materialise datasets/$task first (scripts/export/export_lerobot.py)" >&2
    exit 1; }
done

link() {  # <src dir> <policy> <dataset_id>
  mkdir -p "$POLICIES/policy/$2/data"
  ln -sfn "$1" "$POLICIES/policy/$2/data/$3"
  echo "  linked $2/data/$3 -> ${1#$WT/}"
}

if [ "$WHAT" = gr00t ] || [ "$WHAT" = pi05 ] || [ "$WHAT" = psi0 ] || [ "$WHAT" = all ]; then
  build all "$WT/baselines/data/$MULTI/lerobot"
fi

if [ "$WHAT" = gr00t ] || [ "$WHAT" = all ]; then
  # Compute the stats here so two starting runs do not write the same meta/ directory.
  GR00T_PY=$(cohub_policy_python GR00T_N17)
  if [ -x "$GR00T_PY" ]; then
    ( cd "$POLICIES/policy/GR00T_N17/gr00t_n17" && \
      COHUB_ROBOT=shared COHUB_ACTION_HORIZON=${COHUB_ACTION_HORIZON:-40} \
      "$GR00T_PY" gr00t/data/stats.py --dataset-path "$WT/baselines/data/$MULTI/lerobot" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path ../configs/unitree_g1x2_decentralized_config.py )
  else
    echo "  [note] no GR00T python at $GR00T_PY -- training will generate meta/stats.json itself" >&2
  fi
  mkdir -p "$WT/baselines/data/gr00t"
  # Link the export itself: gr00t reads <dataset_path>/meta/info.json.
  ln -sfn "$WT/baselines/data/$MULTI/lerobot" "$WT/baselines/data/gr00t/$DATA_ID"
  echo "  linked baselines/data/gr00t/$DATA_ID -> baselines/data/$MULTI/lerobot"
fi

if [ "$WHAT" = pi05 ] || [ "$WHAT" = all ]; then
  # shellcheck source=../policy/Pi_05/common.sh
  source "$WT/baselines/policy/Pi_05/common.sh"
  if [ "$BUILT" = 1 ] || [ "$MULTITASK_REBUILD" = 1 ]; then rm -rf "${HF_LEROBOT_HOME:?}/cohub-$MULTI"; fi
  [ -f "$HF_LEROBOT_HOME/cohub-$MULTI/meta/info.json" ] || PI05_FRESH=1
  require_dataset "$MULTI"
  echo "  v3.0 view: $HF_LEROBOT_HOME/cohub-$MULTI"
fi

if [ "$WHAT" = fastwam ] || [ "$WHAT" = all ]; then
  root=$WT/baselines/data/fastwam/${MULTI}_fastwam
  BUILT=0
  build train "$root/lerobot"
  build val   "$root/lerobot_val"
  FASTWAM_PY=$(cohub_policy_python FastWAM)
  if [ -f "$root/dataset_stats.json" ] && [ "$BUILT" = 0 ]; then
    echo "  reusing ${root#$WT/}/dataset_stats.json"
  elif [ -x "$FASTWAM_PY" ]; then
    "$FASTWAM_PY" "$WT/baselines/policy/FastWAM/dataset_stats.py" \
        --task cohub_uncond_1cam_192_1e-4 \
        --train-dir "$root/lerobot" --out "$root/dataset_stats.json" \
        --action-dim 35 --state-dim 43
  else
    echo "  [note] no fastwam python at $FASTWAM_PY -- dataset_stats.json not built" >&2
  fi
  link "$root" FastWAM "$DATA_ID"
  echo "  precompute the text embeddings once:"
  echo "    cd $POLICIES/policy/FastWAM/FastWAM && \$FASTWAM_PY scripts/precompute_text_embeds.py \\"
  echo "      task=cohub_uncond_1cam_192_1e-4 \\"
  echo "      data.train.dataset_dirs=[$root/lerobot] \\"
  echo "      data.val.dataset_dirs=[$root/lerobot_val] \\"
  echo "      data.train.text_embedding_cache_dir=data/text_embeds_cache/XPolicyLab/$DATA_ID \\"
  echo "      data.val.text_embedding_cache_dir=data/text_embeds_cache/XPolicyLab/$DATA_ID"
fi

if [ "$WHAT" = psi0 ] || [ "$WHAT" = all ]; then
  root=$WT/baselines/data/psi0/${MULTI}_psi0
  # Psi0 separates train from val by repo, so two directories.
  # --format psi0 writes Psi0's 36D layout and meta/stats_psi0.json.
  for split in train val; do
    dst=$root/lerobot
    [ "$split" = val ] && dst=$root/lerobot_val
    if [ -f "$dst/meta/info.json" ] && [ "${MULTITASK_REBUILD:-0}" != 1 ]; then
      echo "  reusing ${dst#$WT/} (MULTITASK_REBUILD=1 to rebuild)"
    else
      # The script's default --source is the two-robot set.
      "$PY" "$WT/scripts/export/convert_lerobot.py" --format psi0 --source "$WT/baselines/data/$MULTI/lerobot" \
        --split "$split" --out "$dst" --overwrite
    fi
  done
  link "$root/lerobot"     Psi0 "$DATA_ID"
  link "$root/lerobot_val" Psi0 "${DATA_ID}_val"
fi

echo "[prepare-multitask] $WHAT done"

# ---- 4. pi0.5 normalisation statistics ----
case "$VIEW" in pi05|all)
  echo "[data] 4. pi0.5 norm stats"
  stats=$POLICIES/policy/Pi_05/openpi/assets/pi05_cohub_${MULTI}_decentralized/cohub-$MULTI
  if [ -f "$stats/norm_stats.json" ] && [ "$PI05_FRESH" = 0 ]; then
    echo "  reusing ${stats#$WT/}/norm_stats.json"
  else
    rm -rf "$stats"
    ( cd "$POLICIES/policy/Pi_05/openpi" && uv run scripts/compute_norm_stats.py --config-name pi05_cohub_${MULTI}_decentralized )
  fi ;;
esac
date > "$WT/baselines/data/cohub_data_ready"
echo "[data] done"
