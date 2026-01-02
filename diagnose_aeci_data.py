"""
Diagnose AECI Data Structure
=============================
Check how AECI data is actually stored in Experiment B results
"""

import os
import pickle
import numpy as np

# Configuration
try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    RESULTS_DIR = "/content/drive/MyDrive/DisasterAI_results"
except ImportError:
    RESULTS_DIR = "DisasterAI_results"

# Load Experiment B
file_b = os.path.join(RESULTS_DIR, "results_experiment_B.pkl")
with open(file_b, 'rb') as f:
    results_b = pickle.load(f)

# Check one alignment value
align = list(results_b.keys())[0]
res = results_b[align]

print("="*70)
print(f"AECI DATA STRUCTURE DIAGNOSIS (Alignment = {align})")
print("="*70)

# Check all keys
print("\nAvailable keys in results:")
for key in res.keys():
    print(f"  - {key}")

# Check AECI-related data
print("\n" + "="*70)
print("AECI-RELATED DATA:")
print("="*70)

for key in ['aeci', 'aeci_variance', 'component_aeci']:
    if key in res:
        data = res[key]
        print(f"\n{key}:")
        print(f"  Type: {type(data)}")
        if isinstance(data, np.ndarray):
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
            print(f"  Mean: {np.mean(data):.4f}")

            # Show structure
            if data.ndim == 2:
                print(f"  Structure: (runs={data.shape[0]}, values={data.shape[1]})")
                print(f"  Sample row 0: {data[0, :5]}...")
            elif data.ndim == 3:
                print(f"  Structure: (runs={data.shape[0]}, ticks={data.shape[1]}, cols={data.shape[2]})")
                print(f"  Columns in dimension 2: {data.shape[2]}")
                for col_idx in range(data.shape[2]):
                    col_data = data[:, :, col_idx]
                    print(f"    Column {col_idx}: min={np.min(col_data):.4f}, max={np.max(col_data):.4f}, mean={np.mean(col_data):.4f}")
                print(f"  Sample from run 0, tick 0: {data[0, 0, :]}")
                print(f"  Sample from run 0, tick 50: {data[0, 50, :]}")
                print(f"  Sample from run 0, tick -1: {data[0, -1, :]}")
        elif isinstance(data, list):
            print(f"  Length: {len(data)}")
            if len(data) > 0:
                print(f"  First element type: {type(data[0])}")
                if isinstance(data[0], (list, tuple)):
                    print(f"  First element: {data[0]}")
    else:
        print(f"\n{key}: NOT FOUND")

# Check SECI for comparison
print("\n" + "="*70)
print("SECI DATA (for comparison):")
print("="*70)
seci_data = res.get('seci', np.array([]))
print(f"Type: {type(seci_data)}")
print(f"Shape: {seci_data.shape}")
print(f"Sample from run 0, tick -1: {seci_data[0, -1, :]}")

print("\n" + "="*70)
