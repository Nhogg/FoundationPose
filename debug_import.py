import sys
import os

# 1. Add the build folder to python path (mimicking run_demo.py)
build_path = os.path.join(os.getcwd(), 'mycpp', 'build')
sys.path.append(build_path)

print(f"Looking for mycpp in: {build_path}")
print(f"Files in that folder: {os.listdir(build_path)}")

# 2. Try to import explicitly
print("\nAttempting import...")
import mycpp
print("SUCCESS! mycpp imported correctly.")
