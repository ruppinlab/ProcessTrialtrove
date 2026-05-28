"""
hierarchical stratified bootstrap validation for the BERT-MLP classifier.

procedure (from cross_val.MD):
  outer loop (B iterations): stratified resample each original split independently with replacement
  inner loop (S iterations): stratified resample each outer sample with replacement, train + evaluate
  result: B*S accuracy values -> 95% CI via percentile method

usage:
  python bootstrap_cv.py \
      --datasets  "model/results/4.20_BioClinicalBERT_5/datasets" \
      --model     "emilyalsentzer/Bio_ClinicalBERT" \
      --output    "model/results/4.20_BioClinicalBERT_5/bootstrap" \
      --epochs    3 \
      --B         100 \
      --S         100 \
      --seed      42
"""

import argparse
import copy
import json
import os
import random
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# parsing
parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str,
                    default=r"model/results/4.20_BioClinicalBERT_5/datasets")
parser.add_argument("--model",    type=str,
                    default="emilyalsentzer/Bio_ClinicalBERT")
parser.add_argument("--output",   type=str,
                    default=r"model/results/4.20_BioClinicalBERT_5/bootstrap")
parser.add_argument("--epochs",   type=int, default=5)
parser.add_argument("--B",        type=int, default=100)
parser.add_argument("--S",        type=int, default=100)
parser.add_argument("--seed",     type=int, default=42)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# dataset class - copied from train.py
class ClinicalTrialDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, alias_encoder, biomarker_encoder, column_encoder):
        self.data              = data.reset_index(drop=True)
        self.tokenizer         = tokenizer
        self.max_len           = max_len
        self.alias_encoder     = alias_encoder
        self.biomarker_encoder = biomarker_encoder
        self.column_encoder    = column_encoder
        self.label_mapping     = {'T': 0, 'I': 1, 'F': 2, 'N': -1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row         = self.data.iloc[index]
        raw_context = row['context']
        text        = row['context'].replace(row['alias'], f"[*] {row['alias']} [*]")
        label       = row['classification']
        alias       = row['alias']
        biomarker   = row['biomarker']
        column      = row['column']

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        label_value  = self.label_mapping.get(label, -1)
        alias_id     = int(self.alias_encoder.transform([alias])[0])
        biomarker_id = int(self.biomarker_encoder.transform([biomarker])[0])
        column_id    = int(self.column_encoder.transform([column])[0])

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels':         torch.tensor(label_value,  dtype=torch.long),
            'alias_id':       torch.tensor(alias_id,     dtype=torch.long),
            'biomarker_id':   torch.tensor(biomarker_id, dtype=torch.long),
            'column_id':      torch.tensor(column_id,    dtype=torch.long),
            'alias':          alias,
            'biomarker':      biomarker,
            'column':         column,
            'raw_context':    raw_context,
        }

# model architecture — copied from train.py
class ModifiedBertForSequenceClassification(nn.Module):
    def __init__(self, bert_model, num_labels, alias_vocab, biomarker_vocab, column_vocab):
        super(ModifiedBertForSequenceClassification, self).__init__()
        self.bert                = bert_model
        self.alias_embedding     = nn.Embedding(alias_vocab,    16)
        self.biomarker_embedding = nn.Embedding(biomarker_vocab, 16)
        self.column_embedding    = nn.Embedding(column_vocab,   16)
        self.classifier = nn.Sequential(
            nn.Linear(bert_model.config.hidden_size + 48, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, alias_ids, biomarker_ids, column_ids):
        pooled_output    = self.bert(input_ids, attention_mask=attention_mask).pooler_output
        alias_embed      = self.alias_embedding(alias_ids)
        biomarker_embed  = self.biomarker_embedding(biomarker_ids)
        column_embed     = self.column_embedding(column_ids)
        combined_features = torch.cat(
            (pooled_output, alias_embed, biomarker_embed, column_embed), dim=1
        )
        return self.classifier(combined_features)

# focal loss — copied from train.py
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        probs           = F.softmax(inputs, dim=1)
        ce_loss         = -targets_one_hot * torch.log(probs + 1e-9)
        focal_loss      = self.alpha * (1 - probs) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')

# train and evaluate - copied from train.py (criterion and device are globals)
def train(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss   = 0
    progress_bar = tqdm(train_loader, desc="training", leave=False)
    for batch in progress_bar:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)
        alias_ids      = batch['alias_id'].to(device)
        biomarker_ids  = batch['biomarker_id'].to(device)
        column_ids     = batch['column_id'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=attention_mask,
                       alias_ids=alias_ids, biomarker_ids=biomarker_ids, column_ids=column_ids)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)

def evaluate(model, loader, split_name="val"):
    model.eval()
    predictions, true_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluating ({split_name})", leave=False):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            alias_ids      = batch['alias_id'].to(device)
            biomarker_ids  = batch['biomarker_id'].to(device)
            column_ids     = batch['column_id'].to(device)

            logits = model(input_ids, attention_mask=attention_mask,
                           alias_ids=alias_ids, biomarker_ids=biomarker_ids, column_ids=column_ids)
            preds  = torch.argmax(logits, axis=1)
            total_loss += criterion(logits, labels).item()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    report   = classification_report(true_labels, predictions, target_names=["T", "I", "F"], output_dict=True)
    return avg_loss, report

# load base datasets
train_orig = pd.read_csv(os.path.join(args.datasets, "train.csv"))
val_orig   = pd.read_csv(os.path.join(args.datasets, "val.csv"))
test_orig  = pd.read_csv(os.path.join(args.datasets, "test.csv"))

all_labeled = pd.concat([train_orig, val_orig, test_orig], ignore_index=True)

alias_encoder     = LabelEncoder().fit(all_labeled["alias"])
biomarker_encoder = LabelEncoder().fit(all_labeled["biomarker"])
column_encoder    = LabelEncoder().fit(all_labeled["column"])

# load tokenizer + pretrained bert once
print("loading tokenizer and pretrained model...")
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.add_special_tokens({'additional_special_tokens': ['[*]']})

def _build_fresh_model():
    bert = AutoModel.from_pretrained(args.model)
    bert.resize_token_embeddings(len(tokenizer))
    return ModifiedBertForSequenceClassification(
        bert, num_labels=3,
        alias_vocab=len(alias_encoder.classes_),
        biomarker_vocab=len(biomarker_encoder.classes_),
        column_vocab=len(column_encoder.classes_),
    ).to(device)

_pretrained_state = copy.deepcopy(_build_fresh_model().state_dict())

def reset_model():
    """return a model restored to the pretrained checkpoint, avoiding repeated downloads."""
    bert = AutoModel.from_pretrained(args.model)
    bert.resize_token_embeddings(len(tokenizer))
    m = ModifiedBertForSequenceClassification(
        bert, num_labels=3,
        alias_vocab=len(alias_encoder.classes_),
        biomarker_vocab=len(biomarker_encoder.classes_),
        column_vocab=len(column_encoder.classes_),
    ).to(device)
    m.load_state_dict(copy.deepcopy(_pretrained_state))
    return m

# stratified bootstrap resample
def stratified_resample(df, random_state):
    rng   = np.random.default_rng(random_state)
    parts = []
    for _, group in df.groupby("classification", sort=True):
        seed = int(rng.integers(2**31))
        parts.append(group.sample(n=len(group), replace=True, random_state=seed))
    shuffle_seed = int(rng.integers(2**31))
    return pd.concat(parts).sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

# single training run
def run_one(train_df, val_df, test_df):
    model     = reset_model()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    train_loader = DataLoader(
        ClinicalTrialDataset(train_df, tokenizer, 128, alias_encoder, biomarker_encoder, column_encoder),
        batch_size=16, shuffle=True)
    val_loader = DataLoader(
        ClinicalTrialDataset(val_df,   tokenizer, 128, alias_encoder, biomarker_encoder, column_encoder),
        batch_size=16)
    test_loader = DataLoader(
        ClinicalTrialDataset(test_df,  tokenizer, 128, alias_encoder, biomarker_encoder, column_encoder),
        batch_size=16)

    best_val_loss = float('inf')
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        ckpt = tmp.name

    try:
        for _ in range(args.epochs):
            train(model, train_loader, optimizer, criterion)
            val_loss, _ = evaluate(model, val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), ckpt)

        model.load_state_dict(torch.load(ckpt, map_location=device))
        _, val_report  = evaluate(model, val_loader,  split_name="val")
        _, test_report = evaluate(model, test_loader, split_name="test")
    finally:
        os.unlink(ckpt)

    def extract_metrics(report):
        overall = report["accuracy"]
        per_cls = {label: report[label]["recall"] for label in ["T", "I", "F"] if label in report}
        return overall, per_cls

    val_overall,  val_pc  = extract_metrics(val_report)
    test_overall, test_pc = extract_metrics(test_report)
    return val_overall, val_pc, test_overall, test_pc

# bootstrap - pre-generate all seeds so the run is fully reproducible
RESULTS_CSV = os.path.join(args.output, "bootstrap_results.csv")

completed = set()
if os.path.exists(RESULTS_CSV):
    prev      = pd.read_csv(RESULTS_CSV)
    completed = set(zip(prev["b"].tolist(), prev["s"].tolist()))
    print(f"resuming: {len(completed)} iterations already done.")

rng         = np.random.default_rng(args.seed)
outer_seeds = rng.integers(0, 2**31, size=(args.B, 3)).tolist()
inner_seeds = rng.integers(0, 2**31, size=(args.B, args.S, 3)).tolist()

total = args.B * args.S
done  = len(completed)

with tqdm(total=total, initial=done, desc="bootstrap") as pbar:
    for b in range(args.B):
        b_train = stratified_resample(train_orig, outer_seeds[b][0])
        b_val   = stratified_resample(val_orig,   outer_seeds[b][1])
        b_test  = stratified_resample(test_orig,  outer_seeds[b][2])

        for s in range(args.S):
            if (b, s) in completed:
                pbar.update(1)
                continue

            s_train = stratified_resample(b_train, inner_seeds[b][s][0])
            s_val   = stratified_resample(b_val,   inner_seeds[b][s][1])
            s_test  = stratified_resample(b_test,  inner_seeds[b][s][2])

            val_overall, val_pc, test_overall, test_pc = run_one(s_train, s_val, s_test)

            row = {
                "b": b, "s": s,
                "val_overall_acc":  val_overall,
                "val_T_acc":        val_pc.get("T"),
                "val_I_acc":        val_pc.get("I"),
                "val_F_acc":        val_pc.get("F"),
                "test_overall_acc": test_overall,
                "test_T_acc":       test_pc.get("T"),
                "test_I_acc":       test_pc.get("I"),
                "test_F_acc":       test_pc.get("F"),
            }
            pd.DataFrame([row]).to_csv(
                RESULTS_CSV, mode="a",
                header=not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0,
                index=False,
            )
            pbar.update(1)

# CI via percentile method (alpha = 0.95)
results_df = pd.read_csv(RESULTS_CSV)
alpha      = 0.95
lo, hi     = (1 - alpha) / 2 * 100, (1 + alpha) / 2 * 100

metrics = [
    "val_overall_acc",  "val_T_acc",  "val_I_acc",  "val_F_acc",
    "test_overall_acc", "test_T_acc", "test_I_acc", "test_F_acc",
]

ci = {}
for m in metrics:
    vals  = results_df[m].dropna().values
    ci[m] = {
        "mean":   float(np.mean(vals)),
        "median": float(np.median(vals)),
        "ci_lo":  float(np.percentile(vals, lo)),
        "ci_hi":  float(np.percentile(vals, hi)),
        "std":    float(np.std(vals)),
        "n":      int(len(vals)),
    }

ci_path = os.path.join(args.output, "confidence_intervals.json")
with open(ci_path, "w") as f:
    json.dump({"alpha": alpha, "B": args.B, "S": args.S, "metrics": ci}, f, indent=4)

print(f"\n95% confidence intervals ({args.B * args.S} samples)")
print(f"{'metric':<22}  {'mean':>7}  {'ci low':>7}  {'ci high':>7}")
print("-" * 50)
for m, v in ci.items():
    print(f"{m:<22}  {v['mean']:>7.4f}  {v['ci_lo']:>7.4f}  {v['ci_hi']:>7.4f}")

print(f"\nresults saved to {args.output}/")
