import cv2
import numpy as np 
import argparse
import os

def create_box_mask(image_path, x, y, w, h, out_dir):

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return

    height, width = img.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.rectangle(mask, (x, y), (x + w, y + h), (255), thickness=-1)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "000000.png")
    cv2.imwrite(out_path, mask)
    print(f"Success! Mask saved to {out_path}")
    print(f"Tracking region: x={x}, y={y}, w={w}, h={h}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=int, help="Top-left X coordinate")
    parser.add_argument("y", type=int, help="Top-left Y coordinate")
    parser.add_argument("w", type=int, help="Width of box")
    parser.add_argument("h", type=int, help="Height of box")
    args = parser.parse_args()

    create_box_mask("my_data/rgb/000000.png", args.x, args.y, args.w, args.h, "my_data/masks")
