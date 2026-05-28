from collections import defaultdict
import pandas as pd
import os

# ALIAS DICTIONARY
# ================

# read alias dictionary
def parse_alias_dict():
    """
    Reads a file containing gene to alias entries and returns a dataframe:
    - gene_alias_df: dataframe containing gene alias entries
    """
    file = "helper_files/gene_alias_names_20240507_old.txt"
    gene_alias_df = pd.read_csv(file, sep='\t')
    return gene_alias_df

# get set of all aliases for a given gene (union of alias_symbol + prev_symbol)
def get_aliases(gene, gene_alias_df):
    """
    Returns a set of all aliases for a given gene.
    """

    aliases = set()

    # add gene itself
    aliases.add(gene.lower())

    # if gene has no aliases, return alias set as is
    gene_row = gene_alias_df[gene_alias_df["symbol"] == gene]
    if len(gene_row) == 0: 
        return aliases
    
    # if gene has aliases, get the entry in alias file
    gene_row = gene_row.iloc[0]

    # add alias_symbol to set
    if not pd.isna(gene_row["alias_symbol"]):
        aliases.update(gene_row["alias_symbol"].lower().split("|"))
    
    # add prev_symbol to set
    if not pd.isna(gene_row["prev_symbol"]):
        aliases.update(gene_row["prev_symbol"].lower().split("|"))

    return aliases

# GENE FAMILIES
# =============

def parse_gene_families():
    """
    Reads a file containing gene family entries and returns two dictionaries:
    - gene_to_family: maps each gene to its family
    - family_to_gene: maps each family to its genes
    """

    file = "helper_files/Gene_Family_Entries_20240516.txt"

    gene_to_family = defaultdict(list)
    family_to_gene = defaultdict(list)

    with open(file) as file:
        for gene in file:
            gene = gene.strip()
            if "Gene Family" in gene: continue
            else:
                split = gene.split("1")
                gene_stem, remaining_vars = split[0], split[1].split("/")
                remaining_vars.remove('')

                family_to_gene[gene_stem].append(gene_stem+"1")
                gene_to_family[gene_stem+"1"] = gene_stem

                if remaining_vars:
                    for var in remaining_vars: 
                        family_to_gene[gene_stem].append(gene_stem+var)
                        gene_to_family[gene_stem+var] = gene_stem
    
    return gene_to_family, family_to_gene


# BIOMARKER FIELDS
# ===============
def parse_biomarker_fields():
    """
    Returns a list of 8 biomarker fields of interest
    """
    # define the file name and read the file
    file = "helper_files/biomarker_fields2.txt"

    biomarker_fields = []
    with open(file) as file:
        for line in file:
            if line.startswith("The"): continue
            biomarker_fields.append(line.strip())
    
    return biomarker_fields

# PROBLEM REPORTS
# ===============

def parse_problem_reports():
    """
    Reads multiple Excel files containing problem reports and aggregates them into a single DataFrame.
    - returns Excel containing all problem reports
    """

    return None

    # directory = 'problem_reports'
    # output_file = 'problem_reports.xlsx'

    # dataframes = []
    # for file_name in os.listdir(directory):
    #     file_path = os.path.join(directory, file_name)
    #     if "family" in file_path: continue
    #     if os.path.isfile(file_path) and ("20241205" in file_path or "20241208" in file_path):
    #         try:
    #             df = pd.read_excel(file_path)
    #             dataframes.append(df)
    #             print(f"Successfully loaded {file_name}")
    #         except Exception as e:
    #             print(f"Error loading {file_name}: {e}")

    # problem_reports_df = []
    # if dataframes:
    #     problem_reports_df = pd.concat(dataframes, ignore_index=True)

    #     if output_file:
    #         problem_reports_df.to_excel(output_file, index=False)
    #         print(f"Saved aggregated data to {output_file}")
