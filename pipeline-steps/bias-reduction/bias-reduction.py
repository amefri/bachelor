import pandas as pd
import argparse
import os
import numpy as np
import sys

def calculate_rates(df, sensitive_col, target_col):
    """Calculates selection rates and counts per group."""
    rates = df.groupby(sensitive_col)[target_col].mean().fillna(0)
    counts = df.groupby(sensitive_col).size()
    positive_counts = df[df[target_col] == 1].groupby(sensitive_col).size().fillna(0)
    return rates, counts, positive_counts

def mitigate_bias_oversample(input_csv, output_csv, sensitive_col, target_col):
 
    print(f"--- Bias Mitigation (Oversampling) ---")
    print(f"Input file: {input_csv}")
    print(f"Output file: {output_csv}")
    print(f"Sensitive column: {sensitive_col}")
    print(f"Target column: {target_col}")

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # --- Input Validation ---
    required_cols = [sensitive_col, target_col, 'feature1', 'feature2'] 
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in input CSV: {missing_cols}")
        sys.exit(1)

    if df[sensitive_col].nunique() > 2:
         print(f"Warning: Sensitive column '{sensitive_col}' has more than 2 groups. This script assumes binary groups ('groupA', 'groupB'). Proceeding, but results may be suboptimal.")
         

    if not df[target_col].isin([0, 1]).all():
         print(f"Error: Target column '{target_col}' contains values other than 0 or 1.")
         sys.exit(1)


    # --- Calculate Initial Bias ---
    rates, counts, positive_counts = calculate_rates(df, sensitive_col, target_col)
    print("\n--- Original Data ---")
    print(f"Selection Rates:\n{rates}")
    print(f"Group Counts:\n{counts}")
    print(f"Positive Counts:\n{positive_counts}")

    if len(rates) < 2:
        print("\nError: Less than two groups found in sensitive column. Cannot compare rates.")
        df.to_csv(output_csv, index=False)
        print(f"Original data saved to {output_csv}")
        sys.exit(1)

  
    groupA_rate = rates.get('groupA', 0.0)
    groupB_rate = rates.get('groupB', 0.0)
    groupA_count = counts.get('groupA', 0)
    groupB_count = counts.get('groupB', 0)
    groupA_pos_count = positive_counts.get('groupA', 0)
    groupB_pos_count = positive_counts.get('groupB', 0)

    # --- Determine Which Group to Oversample ---
    if abs(groupA_rate - groupB_rate) < 1e-6: # Check for floating point equality
        print("\nSelection rates are already approximately equal. No oversampling needed.")
        df.to_csv(output_csv, index=False)
        print(f"Original data saved to {output_csv}")
        return

    elif groupA_rate < groupB_rate:
        disadvantaged_group = 'groupA'
        target_rate = groupB_rate
        current_disadvantaged_count = groupA_count
        current_positive_count = groupA_pos_count
    else: # groupB_rate < groupA_rate
        disadvantaged_group = 'groupB'
        target_rate = groupA_rate
        current_disadvantaged_count = groupB_count
        current_positive_count = groupB_pos_count

    print(f"\nDisadvantaged group (lower rate): {disadvantaged_group} (Rate: {rates.get(disadvantaged_group, 0):.4f})")
    print(f"Target rate (from advantaged group): {target_rate:.4f}")

    # --- Calculate Number of Samples to Add ---
    # Formula derived from setting the new rate equal to the target rate:
    # N = (target_rate * current_total_count - current_positive_count) / (1 - target_rate)
    if 1.0 - target_rate < 1e-9: # Avoid division by zero or near-zero
         print("\nWarning: Target rate is 1.0 or very close. Cannot calculate oversampling amount reliably using this method.")
         num_to_add = 0
    else:
         numerator = (target_rate * current_disadvantaged_count - current_positive_count)
         denominator = (1.0 - target_rate)
         # Ensure we only add if the numerator is positive (meaning we are actually behind)
         if numerator > 0:
             num_to_add = int(round(numerator / denominator))
         else:
             num_to_add = 0


    if num_to_add <= 0:
        print(f"\nCalculation suggests {num_to_add} samples to add. No oversampling performed.")
        df_mitigated = df.copy() # Keep original data
    else:
        print(f"\nAttempting to add {num_to_add} positive samples to group '{disadvantaged_group}'...")

        # Get the positive samples from the disadvantaged group
        disadvantaged_positives_df = df[
            (df[sensitive_col] == disadvantaged_group) & (df[target_col] == 1)
        ]

        if disadvantaged_positives_df.empty:
            print(f"\nError: No positive samples found for the disadvantaged group '{disadvantaged_group}'. Cannot oversample.")
            df.to_csv(output_csv, index=False)
            print(f"Original data saved to {output_csv}")
            sys.exit(1)

        # Sample with replacement
        samples_to_add = disadvantaged_positives_df.sample(
            n=num_to_add, replace=True, random_state=42 # Use random_state for reproducibility
        )

        # Append the new samples
        df_mitigated = pd.concat([df, samples_to_add], ignore_index=True)

        print(f"Added {len(samples_to_add)} samples.")

    # --- Shuffle and Save ---
    df_mitigated = df_mitigated.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle rows

    # --- Verify New Rates ---
    final_rates, final_counts, final_positive_counts = calculate_rates(df_mitigated, sensitive_col, target_col)
    print("\n--- Mitigated Data ---")
    print(f"Final Selection Rates:\n{final_rates}")
    print(f"Final Group Counts:\n{final_counts}")
    print(f"Final Positive Counts:\n{final_positive_counts}")
    print(f"Final Total Rows: {len(df_mitigated)}")

    try:
        df_mitigated.to_csv(output_csv, index=False)
        print(f"\nMitigated data successfully saved to {output_csv}")
    except Exception as e:
        print(f"\nError writing mitigated data to CSV: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mitigate demographic parity bias using oversampling.")
    parser.add_argument("--input-csv", required=True, help="Path to the input CSV data file.")
    parser.add_argument("--output-csv", required=True, help="Path to save the mitigated CSV data file.")
    parser.add_argument("--sensitive-col", default="sensitive_attr", help="Name of the sensitive attribute column (default: sensitive_attr).")
    parser.add_argument("--target-col", default="target", help="Name of the binary target column (default: target).")

    args = parser.parse_args()

    mitigate_bias_oversample(
        args.input_csv,
        args.output_csv,
        args.sensitive_col,
        args.target_col
    )