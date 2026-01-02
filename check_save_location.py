"""
EMERGENCY DIAGNOSTIC: Where are files being saved?
"""
import os
import sys

print("="*70)
print("DIAGNOSTIC: Checking save locations")
print("="*70)

# Check if in Colab
try:
    from google.colab import drive
    print("✓ Running in Colab")
    IN_COLAB = True
except ImportError:
    print("✗ NOT in Colab")
    IN_COLAB = False

# Check Drive mount
if IN_COLAB:
    drive_path = "/content/drive/MyDrive"
    if os.path.exists(drive_path):
        print(f"✓ Drive mounted at: {drive_path}")
    else:
        print(f"✗ Drive NOT mounted at: {drive_path}")
        print("  You need to mount Drive first!")

    # Check if results directory exists
    results_dir = "/content/drive/MyDrive/DisasterAI_results"
    if os.path.exists(results_dir):
        print(f"✓ Results directory exists: {results_dir}")

        # List files
        files = os.listdir(results_dir)
        print(f"\n  Files in directory ({len(files)} total):")
        for f in sorted(files):
            fpath = os.path.join(results_dir, f)
            size = os.path.getsize(fpath) / (1024*1024)  # MB
            print(f"    - {f} ({size:.2f} MB)")
    else:
        print(f"✗ Results directory DOES NOT EXIST: {results_dir}")

# Check local directory
local_dir = "/content/DisasterAI_results"
if os.path.exists(local_dir):
    print(f"\n⚠ Local results directory exists: {local_dir}")
    files = os.listdir(local_dir)
    print(f"  Files in LOCAL directory ({len(files)} total):")
    for f in sorted(files):
        fpath = os.path.join(local_dir, f)
        size = os.path.getsize(fpath) / (1024*1024)  # MB
        print(f"    - {f} ({size:.2f} MB)")
    print("\n  ⚠⚠⚠ FILES MAY BE SAVED LOCALLY, NOT ON DRIVE! ⚠⚠⚠")

# Check current working directory
print(f"\nCurrent directory: {os.getcwd()}")
if os.path.exists("DisasterAI_results"):
    print("⚠ 'DisasterAI_results' exists in current directory")

print("="*70)
