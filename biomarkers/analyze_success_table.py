import argparse
import ast
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os


def sum_down_column(df, column_names):
    summed_values = {}
    for column in column_names:
        num1_sum = denom1_sum = num2_sum = denom2_sum = 0
        for cell in df[column]:
            num1, denom1, _ = cell[0]
            num2, denom2, _ = cell[1]
            num1_sum += num1
            denom1_sum += denom1
            num2_sum += num2
            denom2_sum += denom2
        first_prop = num1_sum / denom1_sum if denom1_sum else 0
        second_prop = num2_sum / denom2_sum if denom2_sum else 0
        summed_values[column] = [((num1_sum, denom1_sum, first_prop), (num2_sum, denom2_sum, second_prop))]
    return pd.DataFrame(summed_values, index=[0])


def chi_square_from_tuple(cell):
    (num1, denom1, _), (num2, denom2, _) = cell
    contingency = np.array([
        [num1, denom1 - num1],
        [num2, denom2 - num2]
    ])
    if (contingency.sum(axis=0) == 0).any() or (contingency.sum(axis=1) == 0).any():
        return None, None
    chi2, p, _, _ = chi2_contingency(contingency, correction=False)
    return p, chi2


def analyze_cell(cell):
    (_, _, prop1), (_, _, prop2) = cell
    p, _ = chi_square_from_tuple(cell)
    if p is None:
        return "invalid test"
    alpha = 0.05 / 8
    if p < alpha:
        prop, group = (prop1, 0) if prop1 > prop2 else (prop2, 1)
        return f"({prop}; {group}; p={p:.4g})"
    return f"not significant (p={p:.4g})"


def analyze_success_table(df):
    return df.apply(lambda col: col.map(analyze_cell)) # applies analyze_cell to each cell per column


def main():
    parser = argparse.ArgumentParser(description="Analyze biomarker success table with chi-square tests.")
    parser.add_argument(
        "-i", "--input_file", required=True,
        help="Input success_table.csv with tuple strings per cell"
    )
    parser.add_argument(
        "-o", "--output_file", default="derived_data/success_counts/analyzed_success_table_summed.csv",
        help="Output CSV path for analyzed results"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Step 1: Read and convert tuples
    df = pd.read_csv(args.input_file)
    df = df.apply(lambda col: col.map(lambda x: ast.literal_eval(x.replace('; ', ', ')) if isinstance(x, str) and x.startswith('(') else x))
    # df = df.applymap(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('(') else x)

    # Step 2: Summarize success counts across biomarkers
    summed_df = sum_down_column(df, df.columns[1:])
    full_succes_table_path = args.output_file.replace("analyzed_success_table_summed", "full_success_table")
    def fmt_triple(t):
        return f"({t[0]}; {t[1]}; {t[2]})"
    def format_entry(val):
        if not isinstance(val, tuple):
            return val
        return f"({fmt_triple(val[0])}, {fmt_triple(val[1])})"
    summed_df.applymap(format_entry).to_csv(full_succes_table_path, index=False)
    print(f"Saved full success table to {full_succes_table_path}:")
    print(summed_df)

    # Step 3: Analyze summed table
    analyzed_df = analyze_success_table(summed_df)
    analyzed_df.to_csv(args.output_file, index=False)
    print(f"Analyzed success table saved to {args.output_file}")


if __name__ == "__main__":
    main()

