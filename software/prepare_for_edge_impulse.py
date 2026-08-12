import pandas as pd
import os

# Load your existing dataset
df = pd.read_csv('EMG_TeleOperation_Dataset.csv').dropna()

os.makedirs('ei_upload/training', exist_ok=True)
os.makedirs('ei_upload/testing', exist_ok=True)

# Edge Impulse needs features only — no label column in the data file
# Label comes from the filename
feature_cols = ['bicep_envelope', 'tricep_envelope',
                 'ratio_b_to_t', 'co_activation',
                 'bicep_raw', 'tricep_raw']

label_col = 'label'   # <-- matches your actual dataset (not 'gesture')

for gesture in df[label_col].unique():
    subset = df[df[label_col] == gesture][feature_cols].reset_index(drop=True)
    split = int(len(subset) * 0.8)

    # Save training and testing CSVs named after the gesture
    # (filenames use underscores since Edge Impulse labels don't like spaces)
    safe_name = gesture.replace(' ', '_')

    subset.iloc[:split].to_csv(
        f'ei_upload/training/{safe_name}.csv', index=False)
    subset.iloc[split:].to_csv(
        f'ei_upload/testing/{safe_name}.csv', index=False)
    print(f'{gesture:20s} -> {split:4d} train, {len(subset) - split:4d} test')

print('\nDone. Upload ei_upload/training/ and ei_upload/testing/ folders to Edge Impulse.')
print('Note: class sizes are imbalanced (up to ~2:1) — see message above for counts.')
