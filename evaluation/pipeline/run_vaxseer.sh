#!/usr/bin/env bash
set -euo pipefail

# ---- author args ----
subtype=${1:-a_h3n2}
year=${2:-2018}
device=${3:-0}    # kept for compatibility but not used
month="02"

# ---- resolve paths like authors do ----
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
EVAL_DIR="$(realpath "$SCRIPT_DIR/..")"
ROOT="$(realpath "$EVAL_DIR/..")"
cd "$EVAL_DIR"

# make vaxseer importable
: "${PYTHONPATH:=}"
export PYTHONPATH="$ROOT:$PYTHONPATH"

ckpt_root_dir="$ROOT/runs"

# ===== MINIMAL PATCH #1: pin your *single* LM checkpoint (works for any year) =====
lm_ckpt="$ROOT/runs/flu_lm/2003-10_to_2018-02_2M/$subtype/human_minBinSize100_lenQuantile0.2/weight_loss_by_count/lightning_logs/version_0/checkpoints/epoch=99-step=101100.ckpt"
[[ -f "$lm_ckpt" ]] || { echo "[ERROR] LM ckpt not found: $lm_ckpt"; exit 1; }
# ================================================================================

# ===== MINIMAL PATCH #2: pin your HI checkpoint you verified on disk (2018-02) ===
hi_ckpt="$ROOT/runs/flu_hi_msa_regressor/before_2018-02/${subtype}_seed=1005/random_split/max_steps_150k/lightning_logs/version_0/checkpoints/epoch=132-step=143241.ckpt"
[[ -f "$hi_ckpt" ]] || { echo "[ERROR] HI ckpt not found: $hi_ckpt"; exit 1; }
# ================================================================================

echo "$lm_ckpt"
echo "$hi_ckpt"

# ---- authors' time logic ----
year_minus_three=$((year - 3))
year_plus_one=$((year + 1))
index=$(( (year - 2018) * 2 + 30 ))
testing_time=$index

if [[ "$lm_ckpt" == *"_2M"* ]]; then
  min_testing_time=$(( (year - 2004) * 6 + 1 + 5 ))
  max_testing_time=$(( (year - 2004) * 6 + 1 + 7 ))
else
  min_testing_time=$testing_time
  max_testing_time=$testing_time
fi

# ---- author-compatible inputs (your normalizer creates these) ----
cand_3y="$ROOT/data/gisaid/ha_processed/${year_minus_three}-${month}_to_${year}-${month}_9999M/$subtype/human_minBinSize1000_lenQuantile0.2_minCnt5.fasta"
test_3y="$cand_3y"

echo "candidate_vaccine_path: $cand_3y"
echo "testing_viruses_path: $test_3y"

# output root (authors used $ROOT/runs; you point to your writable folder via OUTPUT_ROOT)
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runs}"
work_base="$OUTPUT_ROOT/pipeline/${year}-${month}/${subtype}"
working_directory_3y="$work_base/vaccine_set=${year_minus_three}-${month}-${year}-${month}___virus_set=${year_minus_three}-${month}-${year}-${month}"
echo "working_directory: $working_directory_3y"

echo "hi_predictor_ckpt: $hi_ckpt"
echo "min_testing_time: $min_testing_time"
echo "max_testing_time: $max_testing_time"
echo "domiance_predictor_ckpt: $lm_ckpt"
mkdir -p "$working_directory_3y"

# ===== MINIMAL PATCH #3: force CPU (replace '--devices $device') =================
bash pipeline/run.sh \
  --candidate_vaccine_path "$cand_3y" \
  --testing_viruses_path   "$test_3y" \
  --working_directory      "$working_directory_3y" \
  --accelerator            cpu \
  --devices                1 \
  --hi_predictor_ckpt      "$hi_ckpt" \
  --domiance_predictor_ckpt "$lm_ckpt" \
  --min_testing_time       "$min_testing_time" \
  --max_testing_time       "$max_testing_time"
# ================================================================================

# WHO candidate (author behavior: try and skip if missing)
who_cand="$ROOT/data/recommended_vaccines_from_gisaid_ha/${year}-${year_plus_one}_NH_${subtype}.fasta"
echo "candidate_vaccine_path: $who_cand"
echo "testing_viruses_path: $test_3y"
working_directory_who="$work_base/vaccine_set=who___virus_set=${year_minus_three}-${month}-${year}-${month}"
echo "working_directory: $working_directory_who"

if [[ -f "$who_cand" ]]; then
  mkdir -p "$working_directory_who"
  # same CPU flags here
  bash pipeline/run.sh \
    --candidate_vaccine_path "$who_cand" \
    --testing_viruses_path   "$test_3y" \
    --working_directory      "$working_directory_who" \
    --accelerator            cpu \
    --devices                1 \
    --hi_predictor_ckpt      "$hi_ckpt" \
    --domiance_predictor_ckpt "$lm_ckpt" \
    --min_testing_time       "$min_testing_time" \
    --max_testing_time       "$max_testing_time"
else
  echo "[WARN] WHO FASTA not found → skipping WHO run."
fi

echo "[DONE] All runs complete."