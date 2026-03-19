import pandas as pd
import glob
import os

files = sorted(
    f for f in glob.glob(r"A:\Projects\csv-merger\csvs\*.csv")
    if os.path.basename(f).lower() != "merged.csv"
)

if not files:
    raise FileNotFoundError("No CSV files found.")

df = pd.concat(
    (pd.read_csv(f, dtype=str, low_memory=False) for f in files),
    ignore_index=True
)

df.to_csv(r"A:\Projects\csv-merger\merged.csv", index=False)
print("Done.")