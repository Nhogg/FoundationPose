import sys
import os

# 1. Setup path
build_path = os.path.expanduser("~/FP/FoundationPose/mycpp/build")
sys.path.append(build_path)

# 2. Load PyTorch first (This is likely breaking things)
print("Loading PyTorch...")
import torch

# 3. Load your extension
print("Attempting to load mycpp...")
try:
    import mycpp
    print("\nSUCCESS: No conflict detected.")
except Exception as e:
    print(f"\n[CRITICAL ERROR DETECTED]: {e}")
    print("Copy and paste this error message!")
