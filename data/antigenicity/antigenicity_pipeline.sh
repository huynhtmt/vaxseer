
year=$1 # "2018"
month=$2 # "02"

raw_pairs_dir="hi_processed"
sequences_path="../gisaid/ha.fasta"
seed="1005"
all_subtypes=("a_h3n2" "a_h1n1")

max_date_exclude="$year-$month"

for subtype in "${all_subtypes[@]}"
do
    echo "Processing $subtype"
    raw_pairs_path="$raw_pairs_dir/${subtype}_pairs.csv"
    subtype_dir="$raw_pairs_dir/before_${max_date_exclude}"
    vacc_fasta="$subtype_dir/${subtype}_vaccine.fasta"
    virus_fasta="$subtype_dir/${subtype}_virus.fasta"
    m8_path="$subtype_dir/${subtype}.m8"

    # 1) Build vaccine/virus sets (prepare.py)
    if [ ! -f "$subtype_dir/${subtype}_hi_folds.csv" ]; then
        python prepare.py \
          --pairs_path "$raw_pairs_path" \
          --max_date_exclude "$max_date_exclude" \
          --sequences_path "$sequences_path" \
          --output_root_dir "$raw_pairs_dir"
    fi

    # >>> INSERTED GUARD (right after prepare.py)
    # Guard: if no vaccine/virus sequences were produced, skip MMseqs for this subtype
    if [ ! -s "$vacc_fasta" ] || [ ! -s "$virus_fasta" ]; then
      echo "[SKIP] No non-empty vaccine/virus FASTAs for ${subtype} before ${max_date_exclude}. Skipping HI alignment."
      # Optionally continue to next subtype; or you could still create empty placeholders for downstream.
      continue
    fi
    # <<< END GUARD

    # 2) MMseqs alignment (only if non-empty)
    if [ ! -f "$m8_path" ]; then
        mmseqs easy-search \
          "$vacc_fasta" \
          "$virus_fasta" \
          "$m8_path" \
          mmseqs_tmp \
          --format-output "query,target,qaln,taln,qstart,qend,tstart,tend,mismatch" \
          --max-seqs 5000

        rm -rf mmseqs_tmp
    fi

    # 3) Split into folds / prepare HI labels
    if [ ! -d "$subtype_dir/${subtype}_seed=$seed" ]; then
        echo "splitting real values..."
        python split.py \
          --pairs_dir "$subtype_dir" \
          --subtype "$subtype" \
          --seed "$seed"
    fi

done
