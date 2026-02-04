#!/usr/bin/env bash
set -euo pipefail

# -------- Parse arguments --------
VALID_ARGS=$(getopt -o c:t:d:h:p:a:b:e:f:g: \
  --long candidate_vaccine_path:,testing_viruses_path:,working_directory:,hi_predictor_ckpt:,domiance_predictor_ckpt:,accelerator:,devices:,min_testing_time:,max_testing_time: -- "$@") || exit 1
eval set -- "$VALID_ARGS"

candidate_vaccine_path=""
testing_viruses_path=""
working_directory=""
hi_predictor_ckpt=""
domiance_predictor_ckpt=""
accelerator=""
devices=""
min_testing_time=""
max_testing_time=""

while true; do
  case "$1" in
    -c|--candidate_vaccine_path) candidate_vaccine_path="$2"; echo "candidate_vaccine_path: $candidate_vaccine_path"; shift 2 ;;
    -t|--testing_viruses_path)   testing_viruses_path="$2";   echo "testing_viruses_path: $testing_viruses_path";   shift 2 ;;
    -d|--working_directory)      working_directory="$2";      echo "working_directory: $working_directory";        shift 2 ;;
    -h|--hi_predictor_ckpt)      hi_predictor_ckpt="$2";      echo "hi_predictor_ckpt: $hi_predictor_ckpt";        shift 2 ;;
    -p|--domiance_predictor_ckpt)domiance_predictor_ckpt="$2";echo "domiance_predictor_ckpt: $domiance_predictor_ckpt"; shift 2 ;;
    --accelerator)               accelerator="$2";            echo "accelerator: $accelerator";                    shift 2 ;;
    -e|--devices)                devices="$2";                echo "devices: $devices";                            shift 2 ;;
    -f|--min_testing_time)       min_testing_time="$2";       echo "min_testing_time: $min_testing_time";          shift 2 ;;
    -g|--max_testing_time)       max_testing_time="$2";       echo "max_testing_time: $max_testing_time";          shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# -------- Resolve repo root (run.sh is in evaluation/pipeline/) --------
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
EVAL_DIR="$(realpath "$SCRIPT_DIR/..")"
ROOT="$(realpath "$EVAL_DIR/..")"

# Ensure PYTHONPATH contains the repo root (so `python -m vaxseer...` resolves)
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# -------- CPU-safe defaults --------
: "${accelerator:=cpu}"   # if unset, force CPU
: "${devices:=1}"         # Lightning expects an integer for CPU

# On CPU: prefer full precision and no DDP
if [[ "$accelerator" == "cpu" ]]; then
  precision_arg="32"
  strategy_arg=""         # no ddp on single CPU process
else
  precision_arg="16"      # AMP on GPU
  strategy_arg="--strategy ddp"
fi

# -------- Optional: clean the working directory (toggle with CLEAN_WORKDIR=1) --------
if [[ -n "${CLEAN_WORKDIR:-}" && "${CLEAN_WORKDIR:-0}" -eq 1 ]]; then
  echo "[CLEAN] removing old working directory: $working_directory"
  rm -rf "$working_directory"
fi

# -------- 1) MMseqs to build vaccine-virus pairs --------
mmseqs_save_dir="$working_directory/vaccine_virus_pairs"
mkdir -p "$mmseqs_save_dir"
mmseqs_save_path="$mmseqs_save_dir/align.m8"
tmp_dir="$mmseqs_save_dir/tmp"

if [[ ! -f "$mmseqs_save_path" ]]; then
  echo "[MMSEQS] building alignment to $mmseqs_save_path"
  mkdir -p "$tmp_dir"
  mmseqs easy-search \
    "$testing_viruses_path" \
    "$candidate_vaccine_path" \
    "$mmseqs_save_path" \
    
    "$tmp_dir" \
    --format-output "query,target,qaln,taln,qstart,qend,tstart,tend,mismatch" \
    --max-seqs 2000
  # Best effort cleanup
  mmseqs rmdb "$tmp_dir" -v 3 || true
fi

# -------- 1b) Build antigenicity pairs.csv --------
pairs_save_path="$mmseqs_save_dir/pairs.csv"
if [[ ! -f "$pairs_save_path" ]]; then
  echo "[ANTIGENICITY] building pairs to $pairs_save_path"
  pushd "$ROOT/data" >/dev/null
  python -m antigenicity.build_test_pairs \
    --alignment_path "$mmseqs_save_path" \
    --virus_fasta_path "$testing_viruses_path" \
    --vaccine_fasta_path "$candidate_vaccine_path" \
    --save_path "$pairs_save_path"
  popd >/dev/null
fi

# -------- 2) Predict HI values for pairs (ESM regressor) --------
classifier_root_dir="$working_directory/vaccine_virus_pairs/prediction/max_steps_150k"
hi_pred_path="$classifier_root_dir/predictions.csv"

if [[ ! -f "$hi_pred_path" ]]; then
  echo "[HI] predicting HI values → $hi_pred_path"
  mkdir -p "$classifier_root_dir"

  set -x
  python -m vaxseer.bin.train \
    --default_root_dir "$classifier_root_dir" \
    --data_module hi_regression_aln \
    --model esm_regressor \
    --accelerator "$accelerator" \
    --devices "$devices" \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --num_workers 8 \
    --precision "$precision_arg" \
    --max_epochs 100 \
    --predict \
    --resume_from_checkpoint "$hi_predictor_ckpt" \
    --predict_index_path "$pairs_save_path" \
    --category false
  set +x
fi

# -------- 3) Predict dominance for viruses (GPT2Time) --------
lm_root_dir="$working_directory/dominance_prediction"
prob_pred_path="$lm_root_dir/lightning_logs/version_0/test_results.csv"

if [[ ! -f "$prob_pred_path" ]]; then
  echo "[LM] predicting dominance → $prob_pred_path"
  mkdir -p "$lm_root_dir"

  set -x
  python -m vaxseer.bin.train \
    --default_root_dir "$lm_root_dir" \
    --test_data_paths "$testing_viruses_path" \
    --max_position_embeddings 1024 \
    --accelerator "$accelerator" \
    --devices "$devices" \
    --batch_size 64 \
    --precision "$precision_arg" \
    --num_workers 8 \
    --test \
    --resume_from_checkpoint "$domiance_predictor_ckpt" \
    --model gpt2_time_new \
    --max_testing_time "$max_testing_time" \
    --min_testing_time "$min_testing_time"
  set +x

  # copy HI predictions into classifier_root_dir if PL changed the log path
  if [[ -f "$classifier_root_dir/lightning_logs/version_0/predictions.csv" ]]; then
    cp "$classifier_root_dir/lightning_logs/version_0/predictions.csv" "$hi_pred_path" || true
    rm -rf "$classifier_root_dir/lightning_logs" || true
  fi
fi

echo "[DONE] run.sh completed: $working_directory"