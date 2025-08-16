import os
import pandas as pd

# Define the input directory where your CSV files are stored
input_dir = "./"  # change this if your files are in a subfolder

# Get all .csv filenames
csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv") and f.startswith("output_roi")]

# Read all CSVs into a dictionary with filenames as keys
csv_data = {filename: pd.read_csv(os.path.join(input_dir, filename), header=None) for filename in csv_files}

# Optional: show how many files were loaded
print(f"{len(csv_data)} CSV files loaded.")

# Preview one file (first in dict)
first_key = next(iter(csv_data))
print(f"Previewing: {first_key}")
csv_data[first_key].head()
