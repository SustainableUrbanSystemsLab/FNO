import os
import pandas as pd

target_dir = 'train_csv'
files = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
removed = 0
total = len(files)

print(f"Checking {total} files in {target_dir}...")

for f in files:
    path = os.path.join(target_dir, f)
    try:
        # Just read the header
        df = pd.read_csv(path, nrows=0)
        if 'mag_U' not in df.columns:
            os.remove(path)
            removed += 1
            print(f"Removed invalid file: {f} (missing 'mag_U')")
    except Exception as e:
        print(f"Error reading {f}: {e}. Removing file.")
        os.remove(path)
        removed += 1

print(f"\nCleanup complete.")
print(f"Total files checked: {total}")
print(f"Files removed:      {removed}")
print(f"Valid files remaining: {total - removed}")
