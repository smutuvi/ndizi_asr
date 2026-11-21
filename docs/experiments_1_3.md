# Experiments 1-3 for the AfricaNLP paper

Copy the three .py files into ndizi_asr/scripts/ and run from the ndizi_asr folder with PYTHONPATH=src:scripts
(the scripts import the repository's own Swahili Fair normalizer and reproduce it exactly; rescore_systems.py
checks this on every text before scoring).

## 1a. Audit the variant list and mine a training-only replacement (minutes)
PYTHONPATH=src:scripts python scripts/variant_audit.py \
  --train response_coding/sampling/train_inventory.csv \
  --valid results/coding_gold/validation_inventory.csv \
  --test response_coding/sampling/test_inventory.csv \
  --out-dir results/variant_audit
Drop --valid if that file does not exist. Read the printed status counts: how many map entries occur only in
test references (test_only) or nowhere (unattested). Then have a Swahili reader review mined_variants.csv:
set accepted to yes or no per pair (the rules cannot tell a variant from a different word).

## 1b, 2, 6. Rescore every system (step by step, both lists, with and without the consent-note clips)
PYTHONPATH=src:scripts python scripts/rescore_systems.py \
  --systems "LoRA-QC-v2=results/sunflower51_ndizi1sw_lora_qc_v2_test/hf_outputs_sunflower51_ndizi1sw_lora_qc_v2_merged/predictions.jsonl" \
            "LoRA-30s=results/summaries/ndizi1sw_sunflower/sunflower51-lora/utterances.csv" \
            "Whisper-Turbo=results/summaries/ndizi1sw_whisper_gemini/ndizi-whisper-turbo/utterances.csv" \
            "Gemini 2.5 Flash=results/summaries/ndizi1sw_whisper_gemini/gemini-2.5-flash/utterances.csv" \
            "SF51 zero-shot=<path to the zero-shot predictions or utterances file>" \
  --mined-map results/variant_audit/mined_variants.csv \
  --compound-audit results/variant_audit/compound_audit.csv \
  --exclude-sample-ids "smutuvi_ndizi-1-sw:smutuvi_ndizi-1-sw_403:403" "smutuvi_ndizi-1-sw:smutuvi_ndizi-1-sw_829:829" \
  --out results/variant_audit/rescore.csv
The S4 row for LoRA-QC-v2 on all clips should print 14.60 (the paper's number). Paths for the other systems
come from scripts/age_rank_correlation.py; check them with ls first.

## 3. Two more training seeds for LoRA-QC-v2 (GPU)
Make the two configs (works on macOS and Linux):
python - <<'PY'
from pathlib import Path
src = Path("configs/train_sunbird51_ndizi_lora.yaml").read_text()
for s in (7, 13):
    out = src.replace("experiment_name: sunflower51_ndizi1sw_lora_qc_v2\n",
                      f"experiment_name: sunflower51_ndizi1sw_lora_qc_v2_seed{s}\n", 1)
    out = out.replace("\nextra:\n", f"\nextra:\n  seed: {s}\n", 1)
    Path(f"configs/train_sunbird51_ndizi_lora_seed{s}.yaml").write_text(out)
    print(s, "seed line present:", f"  seed: {s}" in out, "| name changed:", f"_seed{s}" in out)
PY
Then train, evaluate each merged model on the test split as for LoRA-QC-v2 (see the comment at the top of the
config), and score the predictions with rescore_systems.py. The original run used the default seed, 42.
ndizi-asr-train --config configs/train_sunbird51_ndizi_lora_seed7.yaml
ndizi-asr-train --config configs/train_sunbird51_ndizi_lora_seed13.yaml
