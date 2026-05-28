import pandas as pd
import os
import argparse
import sys

def aggregate_trials(directory="trialtrove_reports", output_file="all_trials.xlsx"):
    """
    Aggregates trial data from multiple Excel files in a given directory into a single Excel file.

    inputs:
    - directory: str, path to the directory containing the trial files
    - output_file: str, path to the output Excel file

    outputs:
    - None; saves an aggregated Excel file to the specified output path
    """
    
    # check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # load all trial files into a list of dataframes
    dataframes = []
    for file_number in range(1,38):
        file_name = f"TrialTrove_Oncology_File{file_number}_20250505.xlsx"
        file_path = os.path.join(directory, file_name)

        if os.path.isfile(file_path) and file_path.endswith('.xlsx'):
            try:
                df = pd.read_excel(file_path)
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    trials_df = []
    if dataframes:
        trials_df = pd.concat(dataframes, ignore_index=True)

        # ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        trials_df.to_excel(output_file, engine="openpyxl", index=False)
        print(f"Saved aggregated data to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate Excel trial files from Trialtrove")
    parser.add_argument("--input", "-i", type=str, default="trialtrove_reports", help="Path to input directory containing trial files")
    parser.add_argument("--output", "-o", type=str, default="all_trials.xlsx", help="Path to output Excel file")
    args = parser.parse_args()

    # check defaults used
    defaults_used = []
    if "--input" not in sys.argv and "-i" not in sys.argv:
        defaults_used.append("input=trialtrove_reports")
    if "--output" not in sys.argv and "-o" not in sys.argv:
        defaults_used.append("output=all_trials.xlsx")

    if defaults_used:
        print(f"Using default argument values: {', '.join(defaults_used)}")

    print(f"Reading {args.input}...")
    aggregate_trials(directory=args.input, output_file=args.output)
    print("Aggregation complete.")