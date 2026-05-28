"""## Install Required Libraries"""

import pandas as pd                                                           # data manipulation
import torch                                                                  # PyTorch framework
import os
from transformers import BertTokenizer, BertForSequenceClassification         # tokenizer for BERT model
from torch.utils.data import DataLoader, Dataset, random_split                # helper functions for parsing data
from torch.optim import AdamW                                                 # optimizer for model training
from sklearn.metrics import classification_report                             # evaluation metrics
from transformers import AutoTokenizer, AutoModel                             # used to load the model + tokenizer from HuggingFace
import argparse
from datetime import datetime
import pickle
import random
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from transformers import get_linear_schedule_with_warmup

# parse user arguments
parser = argparse.ArgumentParser(description="train BERT/NN classification model with optional epoch setting")
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--input', type=str, required=True, help='Path to input (training) data')
parser.add_argument('--output', type=str, required=True, help='Path to output folder to store results')
parser.add_argument('--model', type=str, required=True, help='Specify model to use')
args = parser.parse_args()

# set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# define global variables
EPOCHS = args.epochs
MODEL = args.model
INPUT_FILE = args.input
OUTPUT_FOLDER = args.output

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# model names dictionary
model_names = {
    "emilyalsentzer/Bio_ClinicalBERT": "BioClinicalBERT",
    "dmis-lab/biobert-base-cased-v1.2": "BioBERT",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext": "BiomedBERT",
    "medicalai/ClinicalBERT": "ClinicalBERT",
    "Charangan/MedBERT": "MedBERT",
    "michiyasunaga/BioLinkBERT-large": "BioLinkBERT"
}

"""## Specify File Paths""" 

todays_date = datetime.now().day
todays_month = datetime.now().month
save_results_folder = f"{OUTPUT_FOLDER}/{todays_month}.{todays_date}_{model_names[MODEL]}_{EPOCHS}"
save_model_folder = f"{OUTPUT_FOLDER}/{todays_month}.{todays_date}_{model_names[MODEL]}_{EPOCHS}/model"

# ensure folder exists
os.makedirs(save_results_folder, exist_ok=True)
os.makedirs(save_model_folder, exist_ok=True)

"""## Data Preparation

### 1.1 Custom Dataset Class (for training + testing)
#### This is specific to PyTorch, and tells the model how to read and process the data (i.e. tokenization, converting data to the required format of tensors, etc). A DataLoader also speeds up the ooperation of passing data to the model (either for training or testing) by processing the data parallely and in batches - without a DataLoader, each row would be processed/tokenized one-by-one and then passed into the model.
"""

# custom Dataset class for gene table
class ClinicalTrialDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, alias_encoder, biomarker_encoder, column_encoder):
        """
        Initializes the custom dataset with data, tokenizer, and max sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.alias_encoder = alias_encoder
        self.biomarker_encoder = biomarker_encoder
        self.column_encoder = column_encoder
        # 'T'=0, 'I'=1, 'F'=2, 'N'=-1
        self.label_mapping = {'T':0, 'I':1, 'F':2, 'N': -1}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves an item by index, tokenizes the text, and returns encoded tensors.
        """
        row = self.data.iloc[index]  # access row by index
        raw_context = row['context']                                           # extract 'context' from column
        text = row['context'].replace(row['alias'], f"[*] {row['alias']} [*]") # extract context + include entity marking for alias token
        label = row['classification']                                          # extract 'classification' from column
        alias = row['alias']                                                   # extract 'alias' from column
        biomarker = row['biomarker']                                           # extract 'biomarker' from column
        column = row['column']                                                 # extract 'column' from column

        # tokenize text (context) with padding/truncation for uniform length
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # PyTorch tensor format
        )

        label_value = self.label_mapping.get(label, -1)  # map label to integer value

        # use LabelEncoder integer IDs: small, purpose-built tables sized to actual vocabulary
        alias_id     = int(self.alias_encoder.transform([alias])[0])
        biomarker_id = int(self.biomarker_encoder.transform([biomarker])[0])
        column_id    = int(self.column_encoder.transform([column])[0])

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # tensor of token IDs
            'attention_mask': encoding['attention_mask'].squeeze(0),  # mask to differentiate padding
            'labels': torch.tensor(label_value, dtype=torch.long), # classification label as tensor
            'alias_id':     torch.tensor(alias_id,     dtype=torch.long),
            'biomarker_id': torch.tensor(biomarker_id, dtype=torch.long),
            'column_id':    torch.tensor(column_id,    dtype=torch.long),
            'alias': alias,
            'biomarker': biomarker,
            'column': column,
            'raw_context': raw_context
        }

"""### 1.2 Load Data + Model"""

# load BERT model + tokenizer (model link: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)

# add special token
special_tokens = {'additional_special_tokens': ['[*]']}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# read in input data
all_data = pd.read_csv(INPUT_FILE)

# NOTE: training_data = the data manually labeled/curated by Dr. Schaffer (the ones with classification labels)
training_data = all_data[all_data['classification'] != 'N'].reset_index(drop=True)

# stratified 70 / 15 / 15 split on T/I/F label 
labels = training_data['classification'].values

# first cut: 85% (train + val) vs 15% held-out test
sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
trainval_idx, test_idx = next(sss_test.split(training_data, labels))
trainval_data = training_data.iloc[trainval_idx].reset_index(drop=True)
test_data     = training_data.iloc[test_idx].reset_index(drop=True)

# second cut: within train+val, take 15% of the total as validation (0.15/0.85 = 0.1765)
val_frac = round(0.15 / 0.85, 4)
sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=SEED)
train_idx, val_idx = next(sss_val.split(trainval_data, trainval_data['classification'].values))
train_data = trainval_data.iloc[train_idx].reset_index(drop=True)
val_data   = trainval_data.iloc[val_idx].reset_index(drop=True)

print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# fit label encoders on all labeled data
alias_encoder     = LabelEncoder()
biomarker_encoder = LabelEncoder()
column_encoder    = LabelEncoder()
alias_encoder.fit(training_data['alias'])
biomarker_encoder.fit(training_data['biomarker'])
column_encoder.fit(training_data['column'])

# create datasets and DataLoaders for train / validation / test
train_dataset = ClinicalTrialDataset(train_data, tokenizer, max_len=128, alias_encoder=alias_encoder, biomarker_encoder=biomarker_encoder, column_encoder=column_encoder)
val_dataset   = ClinicalTrialDataset(val_data,   tokenizer, max_len=128, alias_encoder=alias_encoder, biomarker_encoder=biomarker_encoder, column_encoder=column_encoder)
test_dataset  = ClinicalTrialDataset(test_data,  tokenizer, max_len=128, alias_encoder=alias_encoder, biomarker_encoder=biomarker_encoder, column_encoder=column_encoder)

# create DataLoaders for train/validation/test datasets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16)
test_loader  = DataLoader(test_dataset,  batch_size=16)

"""## 2. Train Model

### 2.2 Custom model architecture (BERT + NN) + training
"""

import torch.nn as nn

class ModifiedBertForSequenceClassification(nn.Module):
    def __init__(self, bert_model, num_labels, alias_vocab, biomarker_vocab, column_vocab):
        super(ModifiedBertForSequenceClassification, self).__init__()
        self.bert = bert_model
        self.alias_embedding = nn.Embedding(alias_vocab, 16)  # embedding layer for alias
        self.biomarker_embedding = nn.Embedding(biomarker_vocab, 16)  # embedding layer for biomarker
        self.column_embedding = nn.Embedding(column_vocab, 16)  # embedding layer for column
        self.classifier = nn.Sequential(
            nn.Linear(bert_model.config.hidden_size + 48, 128),  # adjusted input size
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, alias_ids, biomarker_ids, column_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # vector of size (1,768) per context

        alias_embed = self.alias_embedding(alias_ids)
        biomarker_embed = self.biomarker_embedding(biomarker_ids)
        column_embed = self.column_embedding(column_ids)

        combined_features = torch.cat((pooled_output, alias_embed, biomarker_embed, column_embed), dim=1)
        # (vector of (1,768), vector of (1,16), vector of (1,16))

        logits = self.classifier(combined_features)
        return logits

"""### 2.3 Adjust model parameters"""

# instantiate model with LabelEncoder vocabulary sizes (much smaller than full BERT vocab)
modified_model = ModifiedBertForSequenceClassification(model, num_labels=3,
    alias_vocab=len(alias_encoder.classes_),
    biomarker_vocab=len(biomarker_encoder.classes_),
    column_vocab=len(column_encoder.classes_))
modified_model.to(device)

# save encoders alongside model weights so predict_v2.py can apply the same mapping at inference time
with open(os.path.join(save_model_folder, "encoders.pkl"), "wb") as f:
    pickle.dump({'alias': alias_encoder, 'biomarker': biomarker_encoder, 'column': column_encoder}, f)

"""Define custom loss function"""
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()       

        # Compute softmax over the inputs
        probs = F.softmax(inputs, dim=1) 

        # Compute the focal loss
        ce_loss = -targets_one_hot * torch.log(probs + 1e-9)
        focal_loss = self.alpha * (1 - probs) ** self.gamma * ce_loss

        # Reduce the loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

"""### 2.4 Train custom model"""

from tqdm import tqdm
import json

optimizer = AdamW(modified_model.parameters(), lr=2e-5)
# loss_fn = torch.nn.CrossEntropyLoss()
criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')

def train(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids    = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels       = batch['labels'].to(device)
        alias_ids    = batch['alias_id'].to(device)
        biomarker_ids = batch['biomarker_id'].to(device)
        column_ids   = batch['column_id'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=attention_mask, alias_ids=alias_ids, biomarker_ids=biomarker_ids, column_ids=column_ids)

        loss = criterion(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, loader, split_name="val", save_outputs=False):
    model.eval()
    predictions, true_labels, results = [], [], []
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Evaluating ({split_name})", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids    = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels       = batch['labels'].to(device)
            alias_ids    = batch['alias_id'].to(device)
            biomarker_ids = batch['biomarker_id'].to(device)
            column_ids   = batch['column_id'].to(device)
            alias        = batch['alias']
            biomarker    = batch['biomarker']
            column       = batch['column']
            raw_context  = batch['raw_context']

            logits = model(input_ids, attention_mask=attention_mask, alias_ids=alias_ids, biomarker_ids=biomarker_ids, column_ids=column_ids)
            preds = torch.argmax(logits, axis=1)
            total_loss += criterion(logits, labels).item()

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            for i in range(len(input_ids)):
                results.append({
                    "text": raw_context[i],
                    "alias": alias[i],
                    "biomarker": biomarker[i],
                    "column": column[i],
                    "true_label": int(labels[i].cpu().numpy()),
                    "predicted_label": int(preds[i].cpu().numpy()),
                    "logits": logits[i].cpu().numpy().tolist()
                })

    avg_loss = total_loss / len(loader)
    report = classification_report(true_labels, predictions, output_dict=True)
    print(report)

    if save_outputs:
        with open(f"{save_results_folder}/{split_name}_results_{len(training_data)}_{EPOCHS}.json", "w") as f:
            json.dump(report, f, indent=4)
        df_results = pd.DataFrame(results)
        df_results.to_csv(f"{save_results_folder}/{split_name}_outputs_{len(training_data)}_{EPOCHS}.csv", index=False)

    return avg_loss, report

best_val_loss   = float('inf')
best_model_path = os.path.join(save_model_folder, "best_model_weights.pt")
history = []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_loss = train(modified_model, train_loader, optimizer, criterion)
    val_loss, _ = evaluate(modified_model, val_loader, split_name="val")

    print(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
    history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})

    # checkpoint best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(modified_model.state_dict(), best_model_path)
        print(f"  New best model saved (val_loss={val_loss:.4f})")

# save per-epoch loss history
with open(f"{save_results_folder}/training_history.json", "w") as f:
    json.dump(history, f, indent=4)

# final evaluation on held-out test set using the best checkpoint
print("\nLoading best checkpoint for final test evaluation...")
modified_model.load_state_dict(torch.load(best_model_path, map_location=device))
evaluate(modified_model, val_loader,  split_name="val",  save_outputs=True)
evaluate(modified_model, test_loader, split_name="test", save_outputs=True)

# save tokenizer
tokenizer.save_pretrained(save_model_folder)