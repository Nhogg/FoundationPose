"""
Script to extract frames from npz dictionaries
"""

import numpy as np 
import cv2 
import os
import argparse

def extract_npz(npz_path, output_dir, view_key):
    print(F"Loading {npz_path}")
    try:
        data = np.load(npz_path, allow_pickle=True)

        if view_key not in data:
            print(f"Error: Key '{view_key}' not found.")
            print(f"Available keys are: {list(data.keys())}")
            return

        video_data = data[view_key]
        print(f"Found video shape: {video_data.shape}")

    except Exception as e:
        print(f"Failed to load NPZ: {e}")
        return

    rgb_dir = os.path.join(output_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    print(f"Extracting {len(video_data)} frames to {rgb_dir}...")

    for i, frame in enumerate(video_data):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        filename = f"{i:06d}.png"
        cv2.imwrite(os.path.join(rgb_dir, filename), frame_bgr)

        if i % 50 == 0:
            print(f"Saved frame {i}...")
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_file", help="Path to .npz file")
    parser.add_argument("output_dir", help="Folder where 'rgb' subfolder will be created")
    parser.add_argument("view_key", help="Key to extract (over, side, or low)")
    args = parser.parse_args()

    extract_npz(args.npz_file, args.output_dir, args.view_key)
