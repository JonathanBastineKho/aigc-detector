#!/usr/bin/env bash
# Staged dataset download for aigc-detector.
#
# The full corpus is ~400 GB (SID_Set 140 GB + WildFake ~250 GB) and most of it
# is never used. Stages are ordered so that each one unblocks the next piece of
# work; stop as soon as you have what you need.
#
#   stage0  WildFake metadata      ~0.4 GB   generator CSVs + official split scripts
#   stage1  Held-out TEST set      ~28  GB   COCO val2017 + DALL-E Advanced
#   stage2  SID_Set validation     ~17  GB   30k images, day-1 train/probe set
#   stage3  WildFake generators    ~35  GB   leave-one-generator-out pool
#   stage4  SID_Set train          ~123 GB   only if stage2 proves insufficient
#
# Usage:  ./scripts/download_data.sh stage0 [stage1 ...]
#         ./scripts/download_data.sh all
#
# Everything resumes; re-running a completed stage is a cheap no-op.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -d "$REPO_ROOT/.venv/bin" ] && PATH="$REPO_ROOT/.venv/bin:$PATH"

DATA_ROOT="${AIGCD_DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/data}"
mkdir -p "$DATA_ROOT"

WILDFAKE="hy2628982280/WildFake"
SIDSET="saberzl/SID_Set"

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Refuse to start a stage that cannot fit. Cheaper than a failure 20 GB in.
require_space() {
  local need_gb=$1 avail_gb
  avail_gb=$(df -g "$DATA_ROOT" | awk 'NR==2 {print $4}')
  log "disk: ${avail_gb} GB available, stage needs ~${need_gb} GB"
  [ "$avail_gb" -ge "$need_gb" ] || die \
    "Not enough space at $DATA_ROOT (${avail_gb} GB free, need ${need_gb} GB).
     Set AIGCD_DATA_ROOT to cluster scratch or an external volume and re-run."
}

check_tools() {
  command -v modelscope >/dev/null || die "modelscope not found.  pip install modelscope"
  command -v hf          >/dev/null || die "hf not found.  pip install -U 'huggingface_hub[cli]'"
}

# ---------------------------------------------------------------------------
# stage0 — metadata only. Run this first, always.
#
# Tells you which generators exist and which files belong to which split, so
# stages 1 and 3 download exactly what is needed instead of whole archives.
# Also contains the authors' own cross-generator split scripts, which are the
# basis for our leave-one-generator-out evaluation.
# ---------------------------------------------------------------------------
stage0() {
  require_space 2
  log "stage0: WildFake metadata (generator CSVs + official split scripts)"
  modelscope download --dataset "$WILDFAKE" \
    --include 'label_csv_files/*' 'split_train_test/*' \
    --local_dir "$DATA_ROOT/wildfake"
  log "stage0 done -> $DATA_ROOT/wildfake/label_csv_files/"
  log "  Inspect real_coco.csv and the dalle CSVs before running stage1."
}

# ---------------------------------------------------------------------------
# stage1 — the held-out TEST set.
#
# TikTok specifies COCO val2017 (4,998 real) + DALL-E Advanced (8,843 fake) as a
# reference benchmark. We treat it as a TEST set, not a validation set: it has a
# single generator on the fake side, so tuning against it would produce a
# DALL-E detector. It lands in heldout/ and no training script may read that path.
# ---------------------------------------------------------------------------
stage1() {
  require_space 60   # 28 GB archives + room to extract
  log "stage1: held-out test set (COCO val2017 + DALL-E Advanced)"
  mkdir -p "$DATA_ROOT/heldout"
  modelscope download --dataset "$WILDFAKE" \
    --include 'Images/Real/coco.zip' 'Images/Diffusion_based/DALLE.zip' \
    --local_dir "$DATA_ROOT/heldout/_archives"
  log "stage1 archives done. Extract with scripts/build_heldout.py (filters to"
  log "  val2017 + Advanced only, then hashes every file for the leakage guard)."
}

# ---------------------------------------------------------------------------
# stage2 — SID_Set validation split ONLY (30k images, parquet).
#
# This is the day-1 training and probe set. The 123 GB train split is stage4 and
# is deliberately deferred: if a linear probe on 30k images does not work, more
# data is not the missing ingredient.
# ---------------------------------------------------------------------------
stage2() {
  require_space 25
  log "stage2: SID_Set validation split (30k images, ~17 GB)"
  hf download "$SIDSET" --repo-type dataset \
    --include 'data/validation-*' --include 'README.md' --include 'config.json' \
    --local-dir "$DATA_ROOT/sid_set"
  log "stage2 done -> $DATA_ROOT/sid_set/data/"
}

# ---------------------------------------------------------------------------
# stage3 — WildFake generator pool for leave-one-generator-out.
#
# Chosen for architecture diversity per GB, not size. Deliberately excludes
# laion5b.zip (24.8 GB) and GAN_based.zip (47.3 GB) — add them only if the LOGO
# splits turn out to need more generator variety.
# ---------------------------------------------------------------------------
stage3() {
  require_space 80
  log "stage3: WildFake generator pool (DDIM, DDPM, ADM + real ImageNet/FFHQ)"
  modelscope download --dataset "$WILDFAKE" \
    --include \
      'Images/Diffusion_based/DDIM.zip' \
      'Images/Diffusion_based/DDPM.zip' \
      'Images/Diffusion_based/ADM.zip' \
      'Images/Real/imagenet.zip' \
      'Images/Real/ffhq.zip' \
    --local_dir "$DATA_ROOT/wildfake"
  log "stage3 done"
}

# ---------------------------------------------------------------------------
# stage4 — SID_Set train split. 123 GB. Only if stage2 is genuinely exhausted.
# ---------------------------------------------------------------------------
stage4() {
  require_space 140
  log "stage4: SID_Set train split (210k images, ~123 GB) — this takes hours"
  hf download "$SIDSET" --repo-type dataset \
    --include 'data/train-*' --local-dir "$DATA_ROOT/sid_set"
  log "stage4 done"
}

check_tools
[ $# -gt 0 ] || die "usage: $0 stage0 [stage1 stage2 stage3 stage4] | all"
if [ "${1:-}" = "all" ]; then set -- stage0 stage1 stage2 stage3; fi
for s in "$@"; do
  case "$s" in
    stage0|stage1|stage2|stage3|stage4) "$s" ;;
    *) die "unknown stage: $s" ;;
  esac
done
log "requested stages complete. DATA_ROOT=$DATA_ROOT"
