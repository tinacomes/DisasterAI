"""
Quick script to fix Drive connection and locate results
"""

# Step 1: Force unmount current drive
try:
    from google.colab import drive
    import os

    # Unmount current drive
    drive.flush_and_unmount()
    print("✓ Unmounted current drive")
except:
    print("! Not in Colab or already unmounted")

# Step 2: Check if results are still in local Colab storage
print("\n=== Checking for local results in Colab runtime ===")
import subprocess

# Search for any .pkl files
result = subprocess.run(['find', '/content', '-name', '*.pkl', '-type', 'f'],
                       capture_output=True, text=True)
if result.stdout:
    print("✓ Found .pkl files in local storage:")
    print(result.stdout)
else:
    print("⚠ No .pkl files found in /content")

# Search in common directories
possible_dirs = [
    '/content/agent_model_results',
    '/content/DisasterAI_Results',
    '/content/results',
    '/content/drive'
]

print("\n=== Checking common result directories ===")
for dir_path in possible_dirs:
    if os.path.exists(dir_path):
        print(f"✓ Found: {dir_path}")
        # List contents
        try:
            files = os.listdir(dir_path)
            print(f"   Contains {len(files)} files/folders")
            pkl_files = [f for f in files if f.endswith('.pkl')]
            if pkl_files:
                print(f"   PKL files: {pkl_files}")
        except:
            pass
    else:
        print(f"✗ Not found: {dir_path}")

print("\n=== Now remount Drive (choose CORRECT account!) ===")
