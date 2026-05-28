import pandas as pd
import os
import argparse
import sys

# create eligible trials from all trials
def generate_eligible_trials(input_file="all_trials.xlsx", output_file="eligible_trials.xlsx"):
    """
    Reads an Excel file containing all trials. Returns a DataFrame with eligible trials
    
    inputs:
    - input_file: str, path to the input Excel file containing all trials from aggregated_trials (default is None)
    - output_file: str, path to the output Excel file containing eligible trials (default is "eligible_trials.xlsx")
    
    outputs:
    - eligible_trials_df saved as an Excel containing the subset of eligible trials
    """

    # check if input file (i.e. all trials) exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist. Please run the aggregation script first.")
        return

    print(f"Reading {input_file}...")
    trials_df = pd.read_excel(input_file)
    print(f"Successfully read {input_file}")
    print("total # of trials:", len(trials_df))

    # drop trials with blank Oncology Biomarker column
    nonempty_biomarker_df = trials_df[trials_df['Oncology Biomarker'].notna()]
    print("# of trials with non-empty Oncology Biomarker Field:", len(nonempty_biomarker_df), f"[{len(trials_df) - len(nonempty_biomarker_df)} dropped]")

    # isolate trials with eligible outcomes (not NA, not indeterminate, not unknown)
    eligibility_mask = (
        nonempty_biomarker_df['Trial Outcomes'].notna() & 
        (nonempty_biomarker_df['Trial Outcomes'] != "Completed, Outcome indeterminate") &
        (nonempty_biomarker_df['Trial Outcomes'] != "Completed, Outcome unknown")
    )

    eligible_trials_df = nonempty_biomarker_df[eligibility_mask]
    print(f"# of trials with eligible outcomes (not NA, not indeterminate):", len(eligible_trials_df), f"[{len(nonempty_biomarker_df) - len(eligible_trials_df)} dropped]")

    # ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # save the eligible trials to a new Excel file
    eligible_trials_df.to_excel(output_file, index=False)
    print(f"Saved eligible trial data to {output_file}")

# load existing eligible trials DataFrame
def load_eligible_trials(input_file):
    """
    Reads eligible trials data. Returns a DataFrame with eligible trials.
    
    inputs:
    - None; the function reads from a predefined file path
    
    outputs:
    - eligible_trials_df: DataFrame containing the trial data
    """

    # check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist.")
        return None
    
    # define file name and read the Excel file
    eligible_trials_df = pd.read_excel(input_file)

    # parse "Oncology Biomarkers" column (as set of ';'-split biomarkers)
    def parse_biomarker(text):
        distinct_biomarkers = text.split(";")
        return set(x.strip() for x in distinct_biomarkers)

    eligible_trials_df.loc[:, "Oncology Biomarker"] = eligible_trials_df["Oncology Biomarker"].apply(parse_biomarker)
    
    return eligible_trials_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a subset of eligible oncology trials")
    parser.add_argument("--input", "-i", type=str, default="all_trials.xlsx", help="Path to input Excel file (all trials)")
    parser.add_argument("--output", "-o", type=str, default="eligible_trials.xlsx", help="Path to output Excel file")
    args = parser.parse_args()

    # check defaults used
    defaults_used = []
    if "--input" not in sys.argv and "-i" not in sys.argv: defaults_used.append("input=all_trials.xlsx")
    if "--output" not in sys.argv and "-o" not in sys.argv: defaults_used.append("output=eligible_trials.xlsx")

    if defaults_used:
        print(f"Using default argument values: {', '.join(defaults_used)}")

    generate_eligible_trials(input_file=args.input, output_file=args.output)