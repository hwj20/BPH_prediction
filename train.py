import re
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.feature_select import process_data, feature_analysis, find_top_feature
from utils.ml_utils import train_all, train_mean


def count_label(df: pd.DataFrame) -> None:
    """
    Print value counts for non-numeric/categorical-like values per column.
    Numeric-like strings (e.g., '12.3', '5') are ignored.
    """
    for col in df.columns.tolist():
        counts = {}
        for v in df[col]:
            # Skip NaNs
            if pd.isna(v):
                continue

            # If value is numeric or numeric-like string, skip
            if isinstance(v, (int, float, np.number)):
                continue
            if isinstance(v, str) and re.match(r"^\s*[\d.+-]", v):
                # string starts with a number sign/dot -> treat as numeric-like
                continue

            # Count categorical-like tokens
            key = v
            counts[key] = counts.get(key, 0) + 1

        print(col)
        print(counts)


def remove_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove measurement units from string fields by extracting numeric values.
    - If value is already numeric, keep it (NaN -> fill with 0 to keep shape).
    - If value is a string, extract the first numeric token (e.g., '12.3 mg/dL' -> 12.3).
    - If no numeric token is found, leave as 0 to maintain array shape.

    NOTE: If you prefer to keep NaNs instead of 0, replace the 0 with np.nan.
    """
    num_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    for col in df.columns.tolist():
        out_vals = []
        any_converted = False

        for v in df[col]:
            if isinstance(v, (int, float, np.number)):
                if not (isinstance(v, float) and math.isnan(v)):
                    out_vals.append(v)
                else:
                    out_vals.append(0.0)  # fill missing numeric with 0.0
                continue

            if isinstance(v, str):
                m = num_pattern.search(v)
                if m:
                    try:
                        out_vals.append(float(m.group(0)))
                        any_converted = True
                        continue
                    except ValueError:
                        pass

            # If not numeric and not convertible, default to 0.0
            out_vals.append(0.0)

        # Only overwrite the column if we actually converted anything
        if any_converted or all(isinstance(x, (int, float, np.number)) for x in out_vals):
            if len(out_vals) != len(df):
                print(col)
                print(len(out_vals))
                print('Length mismatch error!')
            else:
                df[col] = out_vals

    return df



if __name__ == "__main__":
    ANALYSE_FEATURE = False
    LABEL_COL = "is_BPH"   # make sure your utils use the same label name

    # 1) Load raw data
    df = pd.read_csv("data/train.csv", low_memory=False)

    # 2) Optional analysis (won't change data)
    if ANALYSE_FEATURE:
        feature_analysis("age", df)
        input("Press Enter to continue...")

    # 3) Basic cleaning (safe, provided it does not use label statistics)
    df = process_data(df)
    # Optional unit stripping if needed:
    # df = remove_units(df)

    # 4) Split BEFORE feature selection to avoid information leakage
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df[LABEL_COL] if LABEL_COL in df else None)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[LABEL_COL] if LABEL_COL in temp_df else None)

    # 5) Feature selection ONLY on the training set
    #    Assumes find_top_feature(train_df) returns a list of feature names (excluding the label).
    selected_features = find_top_feature(train_df)  # <-- supervised selection on train only
    # Ensure label is present for downstream splits
    selected_with_label = list(selected_features) + [LABEL_COL]

    # 6) Apply the same selected features to train/val/test
    train_df = train_df[selected_with_label].copy()
    val_df   = val_df[selected_with_label].copy()
    test_df  = test_df[selected_with_label].copy()

    # 7) Build numpy arrays
    # Train
    y_train = np.array(train_df[LABEL_COL])
    X_train = np.array(train_df.drop(LABEL_COL, axis=1))
    # Val
    y_val = np.array(val_df[LABEL_COL])
    X_val = np.array(val_df.drop(LABEL_COL, axis=1))
    # Test
    y_test = np.array(test_df[LABEL_COL])
    X_test = np.array(test_df.drop(LABEL_COL, axis=1))

    feature_list = list(train_df.drop(LABEL_COL, axis=1).columns)
    print("Selected features:", feature_list)

    # 8) Train & evaluate
    print('-' * 20 + 'validation' + '-' * 20)
    # train_mean(X_train, y_train, X_val, y_val)
    train_all(X_train, y_train, X_val, y_val)

    print('-' * 20 + 'test' + '-' * 20)
    # train_mean(X_train, y_train, X_test, y_test)
    train_all(X_train, y_train, X_test, y_test)
