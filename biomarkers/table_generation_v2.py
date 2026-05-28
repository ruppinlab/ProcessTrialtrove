import argparse
import pandas as pd
import re
from tqdm import tqdm
from helper import *
from eligible_trials import load_eligible_trials

"""
Runnable script wrapping your original biomarker analysis code with minimal changes.

Usage:
    python analyze_biomarker_origin.py \
      -i eligible_trials.xlsx \
      -g top_genes.txt \
      -f Gene_Family_Entries_20240516.txt \
      -a gene_aliases.txt \
      -b biomarkers_fields2.txt \
      -o derived_data/final_gene_table.csv

Arguments:
    -i, --input       Excel file of eligible trials
    -g, --genes       Tab-delimited gene list with column 'gene'
    -f, --family      Tab-delimited gene family entries
    -a, --aliases     Tab-delimited gene aliases
    -b, --biomarkers  Biomarker fields file
    -o, --output      Output CSV file
"""

def compile_regex_patterns(phrase_set):
    patterns = {}
    for phrase in phrase_set:
        escaped_phrase = re.escape(phrase.lower())
        pattern = rf'(?<!\w){escaped_phrase}(?!\w)'
        patterns[phrase] = re.compile(pattern)
    return patterns

def analyze_biomarker_origin_table(gene, trials_df):
    #global family_to_gene, gene_alias_df

    all_genes = {gene}
    if gene in family_to_gene:
        all_genes = family_to_gene[gene]

    print("Gene list is ",all_genes)
    contains_gene_df = trials_df[trials_df["Oncology Biomarker"].apply(lambda x: len(x.intersection(all_genes)) > 0)]

    print("Number of trials containing gene ", gene, " is ", len(contains_gene_df.index), " out of ", len(trials_df))
    aliases = set()
    aliases.add(gene.lower())

    for member_gene in all_genes:
        aliases.update(get_aliases(member_gene, gene_alias_df))

    regex_patterns = compile_regex_patterns(aliases)
    print("number of regex patterns for ", gene, " is ", len(regex_patterns))
    
    def find_matches(cell_text):
        if not isinstance(cell_text, str): return []
        cell_text = cell_text.lower()
        words = re.findall(r'\b\w+\b', cell_text)
        word_idx, word_pos = 0, []
        for match in re.finditer(r'\b\w+\b', cell_text):
            start, end = match.start(), match.end()
            word_pos.append((start, end, word_idx))
            word_idx += 1

        matches = []
        for phrase, pattern in regex_patterns.items():
            for match in pattern.finditer(cell_text):
                start_char, end_char = match.start(), match.end()
                match_word_index = next((word_idx for (s, e, word_idx) in word_pos if s == start_char), None)
                if match_word_index:
                    start_window, end_window = max(0, match_word_index-9), min(len(words), match_word_index+10)
                    context = " ".join(words[start_window:end_window])
                    matches.append((phrase, context))

        return matches

    results_df = contains_gene_df.map(find_matches)

    exploded_dfs = []
    for col in results_df.columns:
        exploded_col = results_df[[col]].explode(col)
        other_cols = {c: "" for c in results_df.columns if c != col}
        exploded_col = exploded_col.assign(**other_cols)
        exploded_dfs.append(exploded_col)

    exploded_results_df = pd.concat(exploded_dfs).dropna().sort_index()

    reshaped_data = []
    for trial_id, row in exploded_results_df.iterrows():
        for col_name, value in row.items():
            if pd.notna(value) and value != "":
                alias, context = value
                reshaped_data.append([trial_id, context, col_name, gene, alias])

    reshaped_df = pd.DataFrame(reshaped_data, columns=["Trial ID", "context", "column", "biomarker", "alias"])
    reshaped_df = reshaped_df.set_index("Trial ID")

    return reshaped_df

def main():
    parser = argparse.ArgumentParser(description="Run biomarker origin analysis")
    parser.add_argument('-i','--input', required=True)
    parser.add_argument('-g','--genes', required=True)
    parser.add_argument('-f','--family', required=True)
    parser.add_argument('-a','--aliases', required=True)
    parser.add_argument('-b','--biomarkers', required=True)
    parser.add_argument('-d','--output_dup', required=True)    
    parser.add_argument('-o','--output_nodup', required=True)
    args = parser.parse_args()

    global eligible_trials_df, gene_to_family, family_to_gene, gene_alias_df, biomarker_fields
    eligible_trials_df = load_eligible_trials(args.input)
    eligible_trials_df = eligible_trials_df.set_index("Trial ID")
    print("First length of eligible trials", len(eligible_trials_df))

    genes_df = pd.read_csv(args.genes, sep='\t')
    fam_df = pd.read_csv(args.family, sep='\t')
    gene_alias_df = pd.read_csv(args.aliases, sep='\t')
    gene_to_family, family_to_gene = parse_gene_families()
    gene_alias_df = parse_alias_dict()
    biomarker_fields = parse_biomarker_fields()
                

    eight_columns = eligible_trials_df[biomarker_fields + ["Oncology Biomarker"]]
    print("Second length of eligible trials", len(eight_columns))          
    
    gene_tables = []
    for gene in tqdm(genes_df['gene'], desc="processing genes..."):
        print("Analyzing gene", gene)
        gene_table_df = analyze_biomarker_origin_table(gene, eight_columns)
        print("For gene ", gene, " found ", len(gene_table_df), "contexts")
        gene_tables.append(gene_table_df)

    # FIX: filter out empty DataFrame before concatenating to avoid deprecation warning
    non_empty_gene_tables = [df for df in gene_tables if not df.empty]
    if non_empty_gene_tables:
        table_df = pd.concat(non_empty_gene_tables, ignore_index=False).reset_index().rename(columns={"Trial ID": "trial id"})
    else:
        table_df = pd.DataFrame(columns=["trial id", "context", "column", "biomarker", "alias"])
    table_df.to_csv(args.output_dup, index=False, header=True)
    
    dups_df = table_df.groupby(["trial id", "context", "column"])
    mismatched_groups = {key for key, group in dups_df if len(group) > 1}

    collapsed_rows = []
    for key in mismatched_groups:
        group = dups_df.get_group(key)
        longest_alias_row = group.loc[group['alias'].str.len().idxmax()]
        collapsed_rows.append(longest_alias_row)
    collapsed_df = pd.DataFrame(collapsed_rows)

    mask = table_df[["trial id", "context", "column"]].apply(tuple, axis=1).isin(mismatched_groups)
    final_table_df = pd.concat([table_df[~mask], collapsed_df], ignore_index=True)
    final_table_df.sort_values(by=["biomarker", "trial id", "column"], inplace=True)

    final_table_df.to_csv(args.output_nodup, index=False, header=True)

if __name__ == '__main__':
    main()
