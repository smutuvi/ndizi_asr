#!/usr/bin/env bash
# Scaling study rescoring, on the GPU SERVER, after copying fair_train.json and flagged ids there:
#   scp laptop:.../results/variant_audit/fair_train.json  results/variant_audit/
#   scp laptop:.../results/variant_audit/final/outliers/common.txt  results/variant_audit/common_outliers.txt
#   bash scripts/run_scaling_on_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LISTS="${1:-results/variant_audit/fair_train.json}"
CLEAN="${2:-results/variant_audit/common_outliers.txt}"
C1="smutuvi_ndizi-1-sw:smutuvi_ndizi-1-sw_403:403"
C2="smutuvi_ndizi-1-sw:smutuvi_ndizi-1-sw_829:829"
export PYTHONPATH=src:scripts
python scripts/scaling_runs.py --results-dir results > /tmp/scaling_systems.txt
cat /tmp/scaling_systems.txt
mapfile -t SYS < /tmp/scaling_systems.txt
python scripts/rescore_systems.py --systems "${SYS[@]}" --lists "$LISTS" --exclude-sample-ids "$C1" "$C2" --clean-ids-file "$CLEAN" \
  --out results/variant_audit/rescore_scaling.csv | grep FROZEN
echo "Copy results/variant_audit/rescore_scaling.csv to the laptop (aggregates only), then rerun run_all_final.sh."
