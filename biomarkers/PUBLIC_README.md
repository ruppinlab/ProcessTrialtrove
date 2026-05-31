# TrialTrove Pipeline

This directory contains a **subset** of the analysis pipeline for investigating whether biomarker mentions in fields indicating eligiblity restriction (inclusion/exclusion criteria) vs. fields indicating non-restricted eligiblity (results, endpoints, objectives) have different associations with positive trial outcomes. The pipeline uses a fine-tuned BioClinicalBERT model to classify biomarker-in-context mentions, followed by statistical analysis.

## Dependencies

Install via `pip install -r requirements.txt`. Requires Python 3.9+, and PyTorch with CUDA for GPU-based model training.

Key packages:
- `pandas>=2.1.0`
- `openpyxl`
- `scikit-learn`
- `accelerate`
- `matplotlib`
- `torch`
- `transformers`
- `tqdm`

## Pipeline Overview

The pipeline has three sequential phases:

1. **Context extraction** - filter eligible trials and extract biomarker mentions with surrounding context
2. **Model training and prediction** - fine-tune BioClinicalBERT and generate predictions for all contexts
3. **Statistical analysis** - compute success rates per biomarker and run chi-square tests

## Scripts

### 1. `eligible_trials.py` — Generate Eligible Trials

```bash
python eligible_trials.py -i all_trials.xlsx -o eligible_trials.xlsx
```

Reads an aggregated trials Excel file, filters trials based on eligibility criteria, and saves the result:
- Removes trials with empty Oncology Biomarker fields
- Removes trials with indeterminate or unknown outcomes

#### Arguments

| Flag | Name        | Description                                         | Default                  |
|------|-------------|-----------------------------------------------------|--------------------------|
| `-i` | Input path  | Path to the aggregated trials Excel file            | `all_trials.xlsx`        |
| `-o` | Output path | Path to save output file with eligible trials       | `eligible_trials.xlsx`   |

---

### 2. `top_N_biomarkers.py` — Analyze Top Biomarkers

```bash
python top_N_biomarkers.py -i eligible_trials.xlsx -o top_genes.txt -n 200
```

Reads eligible trials and generates a ranked list of the top N biomarkers by trial mention count. For each biomarker, the following is recorded:
- `gene`: name of gene/biomarker
- `trial_mentions`: number of trials the gene appears in the Oncology Biomarker field
- `family`: True if gene is part of a gene family; False otherwise
- `missing_trials`: number of trials that mention the gene in the Oncology Biomarker field but do not contain the alias/gene in any of the 8 biomarker fields
- `missing_proportion`: proportion of `missing_trials` / `trial_mentions`

#### Arguments

| Flag | Name        | Description                                         | Default                  |
|------|-------------|-----------------------------------------------------|--------------------------|
| `-n` | Number      | Number of top biomarkers to analyze                 | `200`                    |
| `-i` | Input path  | Path to the eligible trials Excel file              | `eligible_trials.xlsx`   |
| `-o` | Output path | Path to save output file with top N biomarker stats | `top_genes.txt`          |

---

### 3. `table_generation_v2.py` — Extract Biomarker Contexts

```bash
python table_generation_v2.py \
  -i eligible_trials.xlsx \
  -g top_genes.txt \
  -f helper_files/Gene_Family_Entries_20240516.txt \
  -a helper_files/gene_alias_names_20240507.txt \
  -b helper_files/biomarker_fields2.txt \
  -d output_with_duplicates.txt \
  -o output_no_duplicates.txt
```

Reads eligible trials and the top-N biomarker list, searches all 8 trial fields for biomarker/alias occurrences, and outputs context quintuples of the form `(trial_id, context_text, column_field, biomarker, alias)`. Produces both a version with duplicates and a deduplicated version.

#### Arguments

| Flag | Name        | Description                                          |
|------|-------------|------------------------------------------------------|
| `-i` | Input path  | Path to the eligible trials Excel file               |
| `-g` | Genes file  | Top-N genes file output from `top_N_biomarkers.py`   |
| `-f` | Gene fam.   | Gene families file (`Gene_Family_Entries_20240516.txt`) |
| `-a` | Alias file  | Biomarker alias file (`gene_alias_names_20240507.txt`)  |
| `-b` | Fields      | File listing the 8 biomarker fields (`biomarker_fields2.txt`) |
| `-o` | Output path | Path to save context quintuples without duplicates   |
| `-d` | Dup. path   | Path to save context quintuples with duplicates      |

---

### 4. `train.py` — Fine-Tune BioClinicalBERT

```bash
python train.py \
  --epochs 3 \
  --input gene_table_curated.csv \
  --output ./results \
  --model emilyalsentzer/Bio_ClinicalBERT
```

Fine-tunes a BioClinicalBERT-based model on the curated biomarker mention classification task. The model (`ModifiedBertForSequenceClassification`) augments the BERT `[CLS]` representation with learned embeddings for the biomarker name, alias, and column field before the classification head. Rows labeled `T`, `I`, or `F` in the input file are used for training; rows labeled `N` are ignored.

Training uses a stratified 70/15/15 train/val/test split, focal loss, and saves the best checkpoint by validation loss.

Outputs (saved to a timestamped subfolder under `--output`):
- `best_model_weights.pt`: Best model checkpoint
- `encoders.pkl`: LabelEncoders for alias, biomarker, and column fields (required by `predict_v2.py`)
- `val_results_*.json` / `test_results_*.json`: Classification reports
- `training_history.json`: Per-epoch train/val loss

#### Arguments

| Flag      | Description                             | Default                           |
|-----------|-----------------------------------------|-----------------------------------|
| `--epochs` | Number of training epochs              | `3`                               |
| `--input`  | Path to input CSV with curated contexts | Required                          |
| `--output` | Path to output folder                   | Required                          |
| `--model`  | HuggingFace model name                  | Required (`emilyalsentzer/Bio_ClinicalBERT`) |

---

### 5. `predict_v2.py` — Run Predictions

```bash
python predict_v2.py \
  --input gene_table_curated.csv \
  --output ./results \
  --model ./results/model
```

Loads a fine-tuned model trained by `train.py` and generates predictions for all rows labeled `N` in the input file. Merges predictions back into the full input table and saves as a CSV named `prediction_{N_rows}_{YYYYMMDD}.csv`.

#### Arguments

| Flag      | Description                                      | Default  |
|-----------|--------------------------------------------------|----------|
| `--input`  | Path to input CSV (rows labeled `N` are predicted) | Required |
| `--output` | Path to output folder                             | Required |
| `--model`  | Path to trained model directory (from `train.py`) | Required |

---

### 6. `success_table_gen.py` — Compute Success Counts

```bash
python success_table_gen.py \
  -c prediction_179009_20250718.csv \
  -e eligible_trials.xlsx \
  -o success_table.csv
```

Loads model predictions and eligible trial outcomes, then computes success rates for each biomarker across 12 criteria sets. Success is defined as a trial outcome of `"Completed, Positive outcome"` or `"Completed, Early positive outcome"`.

Each cell in the output table contains a tuple:
```
((criteria_num, criteria_denom, criteria_prop), (noncriteria_num, noncriteria_denom, noncriteria_prop))
```

The 12 criteria sets analyzed are shown in `criteria_sets.png`.

#### Arguments

| Flag | Description                                              |
|------|----------------------------------------------------------|
| `-c` | CSV file with model predictions                          |
| `-e` | `eligible_trials.xlsx` from the context extraction phase |
| `-o` | Output file path for the success count table             |

---

### 7. `analyze_success_table.py` — Chi-Square Analysis

```bash
python analyze_success_table.py -i success_table.csv -o analyzed_success_table_summed.csv
```

Loads the success count table from `success_table_gen.py`, sums counts across all biomarkers to obtain overall numerators/denominators, and performs a chi-square contingency test per criteria set (criteria vs. non-criteria success rates).

Significance threshold: `alpha = 0.05/8` (Bonferroni correction for 8 fields).

Each cell in the output table contains one of:
- `"invalid test"` — if 2 or more counts in the contingency table are 0
- `"not significant"` — if `p > alpha`
- A tuple `(greater_proportion, 0_or_1)` — when `p < alpha`, where `0` indicates the criteria proportion is higher and `1` indicates the non-criteria proportion is higher

#### Arguments

| Flag | Description                             |
|------|-----------------------------------------|
| `-i` | Success counts file from `success_table_gen.py` |
| `-o` | Output file path for chi-square results |

---

### 8. `compare_results.py` — Pivot Subset Results

```bash
python compare_results.py -i analysis_results/phase_analysis
python compare_results.py -i analysis_results/phase_analysis -d 20260408  # pin to specific date
```

Reads all `[PREFIX_]analyzed_success_table_summed_YYYYMMDD.csv` files from a given directory and pivots them into a single comparison table where rows are the 12 criteria sets and columns are the baseline + each subset. Saves the result as `comparison_table_YYYYMMDD.csv` in the same directory.

#### Arguments

| Flag | Description                                                                 |
|------|-----------------------------------------------------------------------------|
| `-i` | Directory containing `analyzed_success_table_summed_*.csv` files           |
| `-d` | (Optional) Date string (YYYYMMDD) to restrict which files are read; defaults to the most recent date found |

---

### 9. `helper.py` — Shared Parsing Utilities

Not run directly; imported by other pipeline scripts. Provides shared functions for loading the helper files:

- `parse_alias_dict()` — reads gene alias file; returns a DataFrame
- `get_aliases(gene, df)` — returns the full set of aliases (symbol, alias_symbol, prev_symbol) for a gene
- `parse_gene_families()` — reads gene family file; returns `gene_to_family` and `family_to_gene` dicts
- `parse_biomarker_fields()` — reads `biomarker_fields2.txt`; returns the list of 8 field names

---

### 10. `bootstrap_cv.py` — Bootstrap Cross-Validation

```bash
python bootstrap_cv.py \
  --datasets model/results/4.20_BioClinicalBERT_5/datasets \
  --model emilyalsentzer/Bio_ClinicalBERT \
  --output model/results/4.20_BioClinicalBERT_5/bootstrap \
  --epochs 3 \
  --B 100 \
  --S 100 \
  --seed 42
```

Performs hierarchical stratified bootstrap validation of the BERT-MLP classifier. An outer loop (`B` iterations) resamples each original train/val/test split independently with replacement; an inner loop (`S` iterations) resamples each outer sample, trains, and evaluates. The `B×S` accuracy values are used to compute a 95% confidence interval via the percentile method.

#### Arguments

| Flag        | Description                                        | Default |
|-------------|----------------------------------------------------|---------|
| `--datasets` | Path to directory with saved train/val/test splits | Required |
| `--model`    | HuggingFace model name                             | Required |
| `--output`   | Path to output directory for bootstrap results     | Required |
| `--epochs`   | Training epochs per inner-loop iteration           | `3`     |
| `--B`        | Number of outer bootstrap iterations               | `100`   |
| `--S`        | Number of inner bootstrap iterations               | `100`   |
| `--seed`     | Random seed                                        | `42`    |

---

## Helper Files

| File | Description |
|------|-------------|
| `helper_files/biomarker_fields2.txt` | Lists the 8 trial fields searched for biomarker mentions; required by `table_generation_v2.py` and `helper.py` |
| `helper_files/Gene_Family_Entries_20240516.txt` | Gene family groupings; used by `table_generation_v2.py` and `top_N_biomarkers.py` to expand gene families into individual member searches |
| `helper_files/gene_alias_names_20240507.txt` | Gene symbol aliases (2.4 MB); used by `table_generation_v2.py` and `top_N_biomarkers.py` to match alternate gene names |
| `helper_files/drug_classifications_20260316.xlsx` | Manual treatment type labels per trial; used by subset selection scripts |
| `helper_files/Trialtrove_oncology_disease_classification_20260312.xlsx` | Manual disease type classifications (solid/blood/mixed) per trial; used by `select_by_disease.py` |
