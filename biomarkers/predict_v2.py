# Install required packages
import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import os
from datetime import datetime
# parse user arguments
parser = argparse.ArgumentParser(description="predict classifications with pre-trained BERT/NN classification model")
parser.add_argument('--input', type=str, required=True, help='Path to input data (CSV file of rows TO BE CLASSIFIED)')
parser.add_argument('--output', type=str, required=True, help='Path to output folder to store results')
parser.add_argument('--model', type=str, required=True, help='Specify model to use')
args = parser.parse_args()

# define global variables
MODEL_DIR = args.model
INPUT_FILE = args.input
OUTPUT_FOLDER = args.output

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# define model-specific variables
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
#bert_model = AutoModel.from_pretrained(MODEL_DIR)
#tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
bert_model.resize_token_embeddings(len(tokenizer))

# === Custom model wrapper (same architecture as training) ===
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

    def forward(self, input_ids, attention_mask, alias, biomarker, column):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # vector of size (1,768) per context

        alias_ids = torch.tensor([tokenizer.convert_tokens_to_ids(a) for a in alias], device=input_ids.device)
        biomarker_ids = torch.tensor([tokenizer.convert_tokens_to_ids(b) for b in biomarker], device=input_ids.device)
        column_ids = torch.tensor([tokenizer.convert_tokens_to_ids(c) for c in column], device=input_ids.device)

        alias_embed = self.alias_embedding(alias_ids)
        biomarker_embed = self.biomarker_embedding(biomarker_ids)
        column_embed = self.column_embedding(column_ids)

        combined_features = torch.cat((pooled_output, alias_embed, biomarker_embed, column_embed), dim=1)
        # (vector of (1,768), vector of (1,16), vector of (1,16))

        logits = self.classifier(combined_features)
        return logits

# === Load dataset for prediction ===
class ClinicalTrialDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        """
        Initializes the custom dataset with data, tokenizer, and max sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
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
        raw_context = row['context']
        alias = row['alias']                                                      # extract 'alias' from column
        biomarker = row['biomarker']                                              # extract 'biomarker' from column
        column = row['column']                                                    # extract 'column' from column 
        context = raw_context.replace(alias, f"[*] {alias} [*]")                  # extract context + include entity marking for alias token                                        # extract 'biomarker' from column                                               # extract 'column' from column

        # tokenize text (context) with padding/truncation for uniform length
        encoding = self.tokenizer(
            context,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # PyTorch tensor format
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # tensor of token IDs
            'attention_mask': encoding['attention_mask'].squeeze(0),  # mask to differentiate padding
            'alias': alias,
            'biomarker': biomarker,
            'column': column,
            'raw_context': raw_context
        }

# === Prepare data and model ===
gene_table = pd.read_csv(INPUT_FILE)
inference_data = gene_table[gene_table['classification'] == 'N']            # isolate all unclassified rows
dataset = ClinicalTrialDataset(inference_data, tokenizer, max_len=128)
loader = DataLoader(dataset, batch_size=16)

# match vocabulary sizes (same as training)
vocab_size = len(tokenizer)
model = ModifiedBertForSequenceClassification(
    bert_model=bert_model,
    num_labels=3,
    alias_vocab=len(tokenizer),
    biomarker_vocab=len(tokenizer),
    column_vocab=len(tokenizer)
)

model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "custom_model_weights.pt"), map_location=device))
model.to(device)
model.eval()

# === Run predictions ===
results = []
with torch.no_grad():
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        alias = batch['alias']
        biomarker = batch['biomarker']  
        original_context = batch['raw_context']
        column = batch['column']

        logits = model(input_ids, attention_mask=attention_mask, alias=alias, biomarker=biomarker, column=column)
        preds = torch.argmax(logits, axis=1).cpu()

        for i in range(len(input_ids)):
            results.append({
                "context": original_context[i],
                "alias": alias[i],
                "biomarker": biomarker[i],
                "column": column[i],
                "predicted_label": int(preds[i].numpy()),
                "logits": logits[i].cpu().numpy().tolist()
            })
# Get number of rows
num_rows = len(results)

yy = pd.DataFrame(results)
# Rename 'predicted_label' to 'classification' for merging purposes
yy = yy.rename(columns={"predicted_label": "classification"})

# Map numeric labels to classification codes: 0→T, 1→I, 2→F
label_map = {0: "T", 1: "I", 2: "F"}
yy["classification"] = yy["classification"].map(label_map)

# Normalize case of key columns in both dataframes
#for col in ["context", "alias", "biomarker", "column"]:
#    gene_table[col] = gene_table[col].astype(str).str.lower()
#    yy[col] = yy[col].astype(str).str.lower()

# Remove duplicates in yy so each (context, alias, biomarker, column) is unique
yy_dedup = yy.drop_duplicates(subset=["context", "alias", "biomarker", "column"])

# Merge and keep original row count by using left join
gene_table_merge = gene_table.merge(yy_dedup, on=["context", "alias", "biomarker", "column"], how="left", suffixes=('', '_predicted'))

# Replace 'n' in classification with predicted if available
gene_table_merge["classification"] = gene_table_merge.apply(
    lambda row: row["classification_predicted"] if row["classification"].lower() == "n" and pd.notnull(row["classification_predicted"]) else row["classification"],
    axis=1
)

# Drop extra column
gene_table_merge = gene_table_merge.drop(columns=["classification_predicted","logits"])
# Get today's date in YYYYMMDD format
today_str = datetime.today().strftime('%Y%m%d')

# Construct output filename
output_filename = f"prediction_{num_rows}_{today_str}.csv"
output_path = os.path.join(OUTPUT_FOLDER, output_filename)
gene_table_merge.to_csv(output_path, index=False)
print(f"Predictions saved to {OUTPUT_FOLDER}")
