from collections import Counter
from heapq import nlargest
import re
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from helper import *
from eligible_trials import load_eligible_trials

# get all the candidate biomarkers to consider (eligible trials - problem reports)
def get_candidate_biomarkers(eligible_trials_df, problem_reports_df):
    """
    Returns a set of candidate biomarkers to consider for analysis.
    
    inputs:
    - eligible_trials_df: DataFrame containing the trial data
    - problem_reports_df: DataFrame containing the problem reports
    
    outputs:
    - candidate_biomarkers: set of candidate biomarkers
    """
    
    # get all unique biomarkers from eligible trials
    unique_biomarkers = set()
    for _, row in eligible_trials_df.iterrows():
        gene_set = row["Oncology Biomarker"]
        unique_biomarkers.update(gene_set)

    # check if there are any problem reports
    if problem_reports_df == None or len(problem_reports_df) == 0:
        return unique_biomarkers

    # get all unique biomarkers from problem reports
    problem_biomarkers = set(problem_reports_df["Oncology Biomarker"].tolist())
    
    # subtract "problematic" biomarkers from all biomarkers in eligible trials
    candidate_biomarkers = unique_biomarkers.difference(problem_biomarkers)
    
    return candidate_biomarkers

# function for getting # of trials mentioning each biomarker
def parse_oncology_biomarker(trials_df, candidate_biomarkers):
    """
    Counts the # of trials mentioning each biomarker.

    inputs:
    - eligible_trials_df: DataFrame containing the trial data
    - candidate_biomarkers: set of biomarkers to consider
    outputs:
    - biomarker_counts: Counter object with the counts of each biomarker
    """
    biomarker_counts = Counter()

    oncology_biomarker_df = trials_df[["Oncology Biomarker"]]

    for _, row in oncology_biomarker_df.iterrows():
        gene_set = row["Oncology Biomarker"]
        
        vis = set()
        for gene in gene_set:
            if gene in candidate_biomarkers:
                family = gene_to_family.get(gene, gene)   # get the family, or the gene itself
                if family not in vis:
                    biomarker_counts[family] += 1
                    vis.add(family)

    return biomarker_counts

def compute_biomarker_trial_frequency(trials_df, candidate_biomarkers):
    """
    Computes biomarker frequency by unique trial ID (not mentions).

    Each biomarker is counted at most once per trial.

    inputs:
    - trials_df: DataFrame with index = trial ID and column "Oncology Biomarker"
    - candidate_biomarkers: set of biomarkers to consider

    outputs:
    - freq_df: DataFrame with columns:
        ['biomarker', 'num_trials']
    """

    trial_level_counts = Counter()

    for trial_id, row in trials_df.iterrows():
        gene_set = row["Oncology Biomarker"]
        seen_in_trial = set()

        # we count families as equal/synonymous
        # store families of genes (so we don't double count)
        for gene in gene_set:
            if gene in candidate_biomarkers:
                family = gene_to_family.get(gene, gene)
                seen_in_trial.add(family)
        
        # count each biomarker once per trial
        for biomarker in seen_in_trial:
            trial_level_counts[biomarker] += 1

    freq_df = (
        pd.DataFrame(trial_level_counts.items(), columns=["biomarker", "num_trials"]) \
            .sort_values("num_trials", ascending=False) \
            .reset_index(drop=True)
    )

    return freq_df

def plot_biomarker_trial_freq(freq_df, top_n=20, output_path="figures/biomarker_trial_frequency.png"):
    """
    Plot biomarker trial frequency for large N (e.g., 100 biomarkers)
    using a log2-scaled x-axis.

    inputs:
    - freq_df: DataFrame from compute_biomarker_trial_frequency
    - top_n: number of top biomarkers to plot
    - output_path: file path to save the figure
    """

    # ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plot_df = freq_df.head(top_n).copy()

    # y positions
    y_pos = np.arange(len(plot_df))

    # dynamic but stable height
    fig_height = max(8, 0.22 * top_n)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.barh(
        y_pos,
        plot_df["num_trials"].values,
        height=0.8
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["biomarker"].values, fontsize=8)

    ax.set_xlabel("Number of Trials (log-2 scale)", fontsize=12)
    ax.set_title(
        f"Top {top_n} Biomarkers by Trial-Level Frequency",
        fontsize=14
    )

    # log2 axis
    ax.set_xscale("log", base=2)

    ax.set_ylim(-0.5, len(plot_df) - 0.5)
    # remove default margins
    ax.margins(y=0)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved biomarker trial-frequency plot to {output_path}")

# precompute regex patters for each alias
def compile_regex_patterns(aliases):
    """
    Compiles regex patterns for each alias.

    inputs:
    - aliases: set of aliases to compile regex patterns for

    outputs:
    - pattern: dictionary mapping each alias to its compiled regex pattern
    """
    
    # compile regex patterns for each alias
    patterns = {}
    for alias in aliases:
        escaped_phrase = re.escape(alias.lower())  # normalize to lower + escape special chars
        pattern = rf'(?<!\w){escaped_phrase}(?!\w)'  # compile regex phrase
        patterns[alias] = re.compile(pattern)
    return patterns

def analyze_biomarker_origin(gene, trials_df):
    """
    Analyzes occurrence of a gene in 8 columns of trials DataFrame.

    inputs:
    - gene: string representation of a gene (more accurately, a biomarker)
    - trials_df: DataFrame containing the trial data in 8 biomarker fields + Oncology Biomarker field

    outputs:
    - results_df: DataFrame with the results of the analysis
    - missing_trials: DataFrame with trials that do not mention the biomarker
    - missing_IDs: list of IDs of trials that do not mention the biomarker
    """

    # verify if gene is a family (i.e. gene = BRCA is actually an analysis on BRCA1 + BRCA2)
    all_genes = {gene}
    if gene in family_to_gene:
        all_genes = family_to_gene[gene]
        
    # get all trials that contain gene in Oncology Biomarker field
    contains_gene_df = trials_df[trials_df["Oncology Biomarker"].apply(lambda x: len(x.intersection(all_genes)) > 0)]

    # get all aliases to look for
    aliases = set()
    aliases.add(gene.lower())

    # get all aliases for gene/family
    for member_gene in all_genes:
        aliases.update(get_aliases(member_gene, gene_alias_df))
    
    # precompute regex patterns for each alias
    regex_patterns = compile_regex_patterns(aliases)

    # function to find occurrence of pattern in a cell's text
    def find_matches(cell_text):
        if not isinstance(cell_text, str): return []   # cell is empty

        # identify matching phrases
        cell_text = cell_text.lower()
        matches = [phrase for phrase, pattern in regex_patterns.items() if pattern.search(cell_text)]
        return matches

    # isolate trials without any mention of gene/alias
    results_df = contains_gene_df.map(find_matches)
    missing_flag = results_df.map(lambda x: x == []).all(axis = 1).tolist()
    missing_trials = results_df[missing_flag]

    # get IDs of missing trials
    missing_IDs = missing_trials.index.tolist()
    
    return results_df, missing_trials, missing_IDs

# method to process a single gene + get metrics like frequency + missing trials
def process_gene(gene, trials_df):
    """
    Processes a single gene and calculates metrics like frequency and missing trials.

    inputs:
    - gene: string representation of a gene (more accurately, a biomarker)
    - trials_df: DataFrame containing the trial data in 8 biomarker fields + Oncology Biomarker field

    outputs:
    - gene: string representation of the gene
    - count: int, number of trials mentioning the gene
    - part_of_family: bool, whether the gene is part of a family
    - missing_num: int, number of trials that do not mention the gene
    - missing_prop: float, proportion of trials that do not mention the gene
    - missing_IDs: list of IDs of trials that do not mention the gene
    """
    try:
        # check if gene in family
        part_of_family = gene in family_to_gene

        # calculate metrics (proportion + missing trial IDs)
        results_df, missing_trials, missing_IDs = analyze_biomarker_origin(gene, trials_df)
        missing_num = len(missing_trials)
        missing_prop = "{:.2f}".format(missing_num / len(results_df)) if len(results_df) > 0 else "0.00"
        return gene, biomarker_counts[gene], part_of_family, missing_num, missing_prop, missing_IDs
    except Exception as e:
        print(f"Error processing gene {gene}: {e}")
        return gene, "ERROR"


# # get top N most frequent biomarkers
def get_top_N_biomarkers(N, output_file):
    """
    Returns the top N most frequent biomarkers.

    inputs:
    - N: int, number of top biomarkers to return
    - output_file: str, path to the output file

    outputs:
    - top_N_genes: list of top N most frequent biomarkers
    """
    if N <= 0:
        print("N must be greater than 0; please provide a valid number.")
        return
    
    top_N_genes = nlargest(N, biomarker_counts, key=biomarker_counts.get)
    print(f"Top {N} genes have been identified. Analyzing missing trials...")

    gene_to_proportion = dict()
    gene_to_missingIDs = dict()
    trials_with_fields_df = eligible_trials_df[biomarker_fields + ["Oncology Biomarker"]]

    # ensure directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # write to a file
    if top_N_genes:
        with open(output_file, "w") as file, open(f"missing_analysis.txt", "w") as missing_file:
            file.write('gene\ttrial_mentions\tfamily\tmissing_trials\tmissing_proportion\n')
            file.flush()  # force write to disk immediately

            for gene in top_N_genes:
                results = process_gene(gene, trials_with_fields_df)
                
                if results[1] == "ERROR":
                    file.write(f"{gene} ERROR\n")
                    file.flush()
                else:
                    gene, count, part_of_family, missing_num, missing_prop, missing_IDs = results

                    # store metrics for later use
                    gene_to_proportion[gene] = missing_prop
                    gene_to_missingIDs[gene] = missing_IDs

                    file.write(f"{gene}\t{count}\t{part_of_family}\t{missing_num}\t{missing_prop}\n")
                    file.flush()

    print(f"Successfully analyzed top {N} genes; results written to {output_file}")

def main():
    """
    Main function to execute script.
    """
    parser = argparse.ArgumentParser(description="Analyze top N biomarkers among eligible trials")
    parser.add_argument("-n", type=int, default=200, help="Number of top biomarkers to return")
    parser.add_argument("--input", "-i", type=str, default="eligible_trials.xlsx", help="Path to eligible trials Excel file")
    parser.add_argument("--output", "-o", type=str, default="top_genes.txt", help="Path to output .txt file")
    args = parser.parse_args()

    # check defaults used
    defaults_used = []
    if "-n" not in sys.argv: defaults_used.append("-n=200")
    if "--input" not in sys.argv and "-i" not in sys.argv: defaults_used.append("input=eligible_trials.xlsx")
    if "--output" not in sys.argv and "-o" not in sys.argv: defaults_used.append("output=top_genes.txt")

    if defaults_used:
        print(f"Using default argument values: {', '.join(defaults_used)}")

    # 1. load the eligible trials DataFrame
    global eligible_trials_df
    eligible_trials_df = load_eligible_trials(args.input)
    if eligible_trials_df is None:
        print(f"Error: Unable to load eligible trials from {args.input}.")
        return
    
    print(f"Loaded {len(eligible_trials_df)} eligible trials from {args.input}")

    # 2. parse gene families, alias dict, problem reports, and biomarker fields
    global gene_to_family, family_to_gene, gene_alias_df, problem_reports, biomarker_fields
    gene_to_family, family_to_gene = parse_gene_families()
    gene_alias_df = parse_alias_dict()
    problem_reports = parse_problem_reports()
    biomarker_fields = parse_biomarker_fields()
    
    # 3. get candidate biomarkers (eligible trials - problem reports)
    candidate_biomarkers = get_candidate_biomarkers(eligible_trials_df, problem_reports)

    # 4. store trial counts for each gene where there is a mention
    global biomarker_counts
    biomarker_counts = parse_oncology_biomarker(eligible_trials_df, candidate_biomarkers)

    # 5. get top N biomarkers
    get_top_N_biomarkers(args.n, args.output)

    # 6. compute trial-level biomarker frequency
    freq_df = compute_biomarker_trial_frequency(
        eligible_trials_df,
        candidate_biomarkers
    )

    # 7. plot top biomarkers by trial frequency
    figure_path = "figures/biomarker_trial_frequency.png"
    output_path = args.output.replace("top_genes.txt", figure_path)
    plot_biomarker_trial_freq(
        freq_df,
        top_n=20,
        output_path=output_path
    )

if __name__ == "__main__":
    main()
