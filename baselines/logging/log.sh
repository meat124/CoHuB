#!/usr/bin/env bash
# Upload one evaluation's results.json and videos to wandb by hand.
#
#   baselines/logging/log.sh Pi_05 eval_results/cocarry/Pi_05-decentralized-cohub2-seed1-ckpt40000
#   baselines/logging/log.sh DP eval_results/framehang/DP-decentralized-b256-e600-seed1-ckpt600 50
#
# scripts/eval/runner.sh does this as its last step. The third argument is the minimum
# episode count upload.py accepts (default COHUB_EVAL_EPISODES, 100).
# README.md says what is logged where.

WT=${COHUB_WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}; export COHUB_WT=$WT
source "$WT/baselines/common.sh"
POLICY=${1:?usage: log.sh <POLICY> <eval_results[_<label>]/<scene>/<run-id>> [min episodes]}
DIR=${2:?usage: log.sh <POLICY> <eval_results[_<label>]/<scene>/<run-id>> [min episodes]}
case "$DIR" in /*) ;; *) DIR=$WT/$DIR ;; esac
DIR=${DIR%/}
# The run's tag and task set are the directory's, as the runner named it
# (<POLICY>-<mode>[-<tag>]-seed<N>... under <scene>/), not this shell's or the config's default.
if [[ "$(basename "$DIR")" =~ ^.+-(de)?centralized-(.+)-seed[0-9]+ ]]; then export CKPT_TAG=${BASH_REMATCH[2]}; else export CKPT_TAG=; fi
export COHUB_TASK_SET; COHUB_TASK_SET=$(cohub_task_set "$(basename "$(dirname "$DIR")")") || exit 2
cohub_init "$POLICY"
[ -f "$DIR/results.json" ] || { echo "[log] no results.json under ${DIR#$WT/}" >&2; exit 1; }
cohub_upload "$POLICY" "$DIR" "${3:-${COHUB_EVAL_EPISODES:-100}}"
