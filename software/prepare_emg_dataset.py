#!/usr/bin/env python3
"""Prepare EMG dataset from raw input text.

This script parses `input.txt` lines like:
  17:58:46.521531 | AD: 10.8 | EXG: 858.12

It then computes envelope and rolling window features and outputs a clean,
minimal dataset for training.

Usage:
  python prepare_emg_dataset.py --input input.txt --output input_features.csv
  python prepare_emg_dataset.py --input input.txt --output input_features.csv --label up
  python prepare_emg_dataset.py --input input.txt --output input_features.csv --label up --label-id 2
  python prepare_emg_dataset.py --input input.txt --output input_features_full.csv --full
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATTERN = re.compile(r"(?P<timestamp>[^|]+)\|\s*AD:\s*(?P<bicep>[-\d.]+)\s*\|\s*EXG:\s*(?P<tricep>[-\d.]+)")
EXG_ONLY_PATTERN = re.compile(r"(?P<timestamp>[^|]+)\|\s*EXG:\s*(?P<tricep>[-\d.]+)")

LABEL_ID_MAP = {
    'REST': 0,
    'ELBOW_FLEX': 1,
    'ELBOW_EXTEND': 2,
    'WRIST_PRONATE': 3,
    'GRIP_CLOSE': 4,
    'GRIP_OPEN': 5,
    'up': 0,
    'down': 1,
    'yaw towards left': 2,
    'yaw towards right': 3,
    'roll': 4,
}


def parse_input_text(file_path: Path) -> pd.DataFrame:
    rows = []
    text = file_path.read_text(encoding='utf-8', errors='ignore')
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Try full pattern (AD + EXG)
        match = RAW_PATTERN.search(line)
        if match:
            rows.append({
                'timestamp': match.group('timestamp').strip(),
                'bicep_raw': float(match.group('bicep')),
                'tricep_raw': float(match.group('tricep')),
            })
        else:
            # Try EXG-only pattern
            match = EXG_ONLY_PATTERN.search(line)
            if match:
                rows.append({
                    'timestamp': match.group('timestamp').strip(),
                    'bicep_raw': 0.0,  # No bicep data for EXG-only lines
                    'tricep_raw': float(match.group('tricep')),
                })

    if not rows:
        raise ValueError(f'No valid data found in {file_path}')

    return pd.DataFrame(rows)


def compute_envelopes(df: pd.DataFrame, envelope_window: int = 10) -> pd.DataFrame:
    df2 = df.copy()
    df2['bicep_baseline'] = df2['bicep_raw'].rolling(envelope_window, min_periods=1).median()
    df2['tricep_baseline'] = df2['tricep_raw'].rolling(envelope_window, min_periods=1).median()
    df2['bicep_envelope'] = np.abs(df2['bicep_raw'] - df2['bicep_baseline']).rolling(envelope_window, min_periods=1).mean()
    df2['tricep_envelope'] = np.abs(df2['tricep_raw'] - df2['tricep_baseline']).rolling(envelope_window, min_periods=1).mean()
    df2['ratio_b_to_t'] = df2['bicep_envelope'] / df2['tricep_envelope'].replace(0, np.nan)
    df2['ratio_b_to_t'] = df2['ratio_b_to_t'].fillna(0.0)
    df2['co_activation'] = df2[['bicep_envelope', 'tricep_envelope']].min(axis=1)
    return df2.drop(columns=['bicep_baseline', 'tricep_baseline'])


def compute_window_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df2 = df.copy()
    df2['b_mean'] = df2['bicep_raw'].rolling(window, min_periods=1).mean()
    df2['b_max'] = df2['bicep_raw'].rolling(window, min_periods=1).max()
    df2['b_std'] = df2['bicep_raw'].rolling(window, min_periods=1).std().fillna(0.0)
    df2['b_rms'] = np.sqrt((df2['bicep_raw'] ** 2).rolling(window, min_periods=1).mean())
    df2['b_range'] = df2['bicep_raw'].rolling(window, min_periods=1).max() - df2['bicep_raw'].rolling(window, min_periods=1).min()

    df2['t_mean'] = df2['tricep_raw'].rolling(window, min_periods=1).mean()
    df2['t_max'] = df2['tricep_raw'].rolling(window, min_periods=1).max()
    df2['t_std'] = df2['tricep_raw'].rolling(window, min_periods=1).std().fillna(0.0)
    df2['t_rms'] = np.sqrt((df2['tricep_raw'] ** 2).rolling(window, min_periods=1).mean())
    df2['t_range'] = df2['tricep_raw'].rolling(window, min_periods=1).max() - df2['tricep_raw'].rolling(window, min_periods=1).min()

    df2['ratio_b_t'] = df2['b_mean'] / df2['t_mean'].replace(0, np.nan)
    df2['ratio_b_t'] = df2['ratio_b_t'].fillna(0.0)
    df2['sum_bt'] = df2['b_mean'] + df2['t_mean']
    df2['diff_bt'] = df2['b_mean'] - df2['t_mean']
    df2['coactivation'] = np.minimum(df2['b_mean'], df2['t_mean'])
    return df2


def assign_labels(df: pd.DataFrame, label: str | None, label_id: int | None) -> pd.DataFrame:
    df2 = df.copy()
    if label is not None:
        df2['label'] = label
        if label_id is None:
            df2['label_id'] = LABEL_ID_MAP.get(label, -1)
        else:
            df2['label_id'] = label_id
    else:
        df2['label'] = pd.NA
        df2['label_id'] = -1
    return df2


def build_dataset(input_path: Path, label: str | None, label_id: int | None, full: bool = False) -> pd.DataFrame:
    df = parse_input_text(input_path)
    df = compute_envelopes(df, envelope_window=10)
    df = compute_window_features(df, window=20)
    df = assign_labels(df, label, label_id)

    minimal_cols = [
        'timestamp',
        'bicep_raw', 'tricep_raw',
        'bicep_envelope', 'tricep_envelope', 'ratio_b_to_t', 'co_activation',
        'label',
    ]

    full_cols = [
        'timestamp',
        'bicep_raw', 'tricep_raw',
        'bicep_envelope', 'tricep_envelope', 'ratio_b_to_t', 'co_activation',
        'b_mean', 'b_max', 'b_std', 'b_rms', 'b_range',
        't_mean', 't_max', 't_std', 't_rms', 't_range',
        'ratio_b_t', 'sum_bt', 'diff_bt', 'coactivation',
        'label',
    ]

    return df[full_cols] if full else df[minimal_cols]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build EMG dataset from raw input.txt')
    parser.add_argument('--input', '-i', type=Path, default=Path('input.txt'), help='Raw input text file')
    parser.add_argument('--output', '-o', type=Path, default=Path('input_features.csv'), help='Output CSV file')
    parser.add_argument('--label', '-l', type=str, help='Label name to assign to all rows')
    parser.add_argument('--label-id', type=int, help='Numeric label id for the gesture')
    parser.add_argument('--full', action='store_true', help='Keep all computed features instead of only essential columns')
    parser.add_argument('--append', '-a', type=Path, help='Optional existing CSV to append to')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_dataset(args.input, args.label, args.label_id, args.full)
    if args.append:
        existing = pd.read_csv(args.append)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f'Appended {len(df)} rows to {args.append} and saved {len(combined)} rows to {args.output}')
    else:
        df.to_csv(args.output, index=False)
        print(f'Wrote {len(df)} rows to {args.output}')

    print('\nColumn summary:')
    print(df.columns.tolist())
    print('\nExample rows:')
    print(df.head().to_string(index=False))


if __name__ == '__main__':
    main()
