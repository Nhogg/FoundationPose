import trimesh
import argparse
import numpy as np

def shrink(input_path, output_path):
    print(f"Loading {input_path}...")
    mesh = trimesh.load(input_path)
    
    # Check current size
    extents = mesh.extents
    print(f"Current Size (X, Y, Z): {extents}")
    
    if np.max(extents) > 10:
        print("DETECTED GIANT MESH: Likely in millimeters.")
        print("Scaling down by 0.001 (1000x)...")
        mesh.apply_scale(0.001)
        print(f"New Size (X, Y, Z): {mesh.extents}")
    else:
        print("WARNING: Mesh seems small already. Scaling anyway? (Modify script if needed)")
        # Uncomment below if you want to force it regardless of check
        # mesh.apply_scale(0.001)

    print(f"Exporting to {output_path}...")
    mesh.export(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input mesh file")
    parser.add_argument("output", help="Output mesh file")
    args = parser.parse_args()

    shrink(args.input, args.output)
