"""
compare_results.py

Reads all analyzed_success_table_summed_*.csv files from a given analysis
results directory, pivots them into a single comparison table (criteria sets
as rows, subsets as columns), and saves the result.

Works for any subset type (phase, disease, biomarker, sponsor, treatment, year)
as long as the files follow the naming convention produced by 03_analyze_data.sh:
    [PREFIX_]analyzed_success_table_summed_YYYYMMDD.csv

Usage:
    python compare_results.py -i analysis_results/phase_analysis [-d YYYYMMDD]

Arguments:
    -i  Directory containing analyzed_success_table_summed_*.csv files
    -d  (Optional) Date string (YYYYMMDD) to restrict which files are read.
        If omitted, the most recent date found in the directory is used.
"""

import os
import re
import argparse
import pandas as pd

FILENAME_PATTERN = re.compile(
    r'^(.+_)?analyzed_success_table_summed_(\d{8})\.csv$'
)

BASELINE_LABEL = "All (Baseline)"


def find_files(input_dir, date_filter=None):
    """Return list of (filepath, label, date) tuples for matching files."""
    matches = []
    for fname in sorted(os.listdir(input_dir)):
        m = FILENAME_PATTERN.match(fname)
        if not m:
            continue
        raw_prefix = m.group(1) or ""
        label = raw_prefix.rstrip("_") if raw_prefix else BASELINE_LABEL
        date = m.group(2)
        if date_filter and date != date_filter:
            continue
        matches.append((os.path.join(input_dir, fname), label, date))
    return matches


def pick_date(matches):
    """Return the most recent date present across all matched files."""
    dates = sorted({date for _, _, date in matches}, reverse=True)
    return dates[0] if dates else None


def load_file(filepath):
    """Load a single analyzed_success_table_summed CSV as a Series indexed by criteria set."""
    df = pd.read_csv(filepath, header=0)
    # File has exactly one data row; transpose to get criteria sets as index
    return df.iloc[0]


def sort_key(label):
    """Sort baseline first, then alphabetically."""
    if label == BASELINE_LABEL:
        return (0, "")
    return (1, label)


def main():
    parser = argparse.ArgumentParser(
        description="Pivot subset analysis results into a single comparison table."
    )
    parser.add_argument("-i", "--input_dir", required=True,
                        help="Directory containing analyzed_success_table_summed_*.csv files")
    parser.add_argument("-d", "--date", default=None,
                        help="Date (YYYYMMDD) of the analysis run to compare. "
                             "Defaults to the most recent date found.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Directory not found: {args.input_dir}")
        return

    all_matches = find_files(args.input_dir)
    if not all_matches:
        print(f"ERROR: No analyzed_success_table_summed_*.csv files found in {args.input_dir}")
        return

    date = args.date if args.date else pick_date(all_matches)
    matches = [(fp, label, d) for fp, label, d in all_matches if d == date]

    if not matches:
        print(f"ERROR: No files found for date {date}")
        return

    print(f"Building comparison table for date: {date}")
    print(f"Found {len(matches)} file(s):")
    for fp, label, _ in sorted(matches, key=lambda x: sort_key(x[1])):
        print(f"  [{label}]  {os.path.basename(fp)}")

    # Build comparison table
    data = {}
    for filepath, label, _ in matches:
        data[label] = load_file(filepath)

    sorted_labels = sorted(data.keys(), key=sort_key)
    comparison_df = pd.DataFrame({label: data[label] for label in sorted_labels})
    comparison_df.index.name = "Criteria Set"

    output_path = os.path.join(args.input_dir, f"comparison_table_{date}.csv")
    comparison_df.to_csv(output_path)
    print(f"\nComparison table saved to: {output_path}")
    print(f"\n{comparison_df.to_string()}")


if __name__ == "__main__":
    main()
