import os
import argparse
import requests
import numpy as np 
import cv2
import glob
from tqdm import tqdm

def process_directory(input_dir, output_dir, server_url, prompt_mode="everything"):
    # Setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_extensions = ['*.png', '*.jpeg', '*.jpg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    image_files.sort()
    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Target Server: {server_url}")

    # Loop over image dir 
    for img_path in tqdm(image_files, desc="Processing Images..."):
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, base_name + ".npz")

        if os.path.exists(output_path):
            continue

        with open(img_path, 'rb') as f:
            files = {'image': (file_name, f, 'image/png')}

            data = {'mode': prompt_mode}

            try:
                response = requests.post(server_url, files=files, data=data, timeout=30)

                if response.status_code == 200:
                    
                    try:
                        import io 
                        buff = io.BytesIO(response.content)
                        loaded = np.load(buff)
                        mask = loaded['masks']
                    except:
                        mask_array = np.frombuffer(response.content, np.uint8)
                        mask = cv2.imdecode(mask_array, cv2.IMREAD_UNCHANGED)

                    if mask in None:
                        print(f"\nERROR DECODING RESPONSE FOR {file_name}")
                        continue

                    np.savez_compressed(output_path, masks=mask)

                else:
                    print(f"\nSERVER ERROR {response.status_code} for {file_name}: {response.text}")

            except requests.exceptions.ConnectionError:
                print(f"CRITICAL: Could not connect to server. Is it running on localhost:8080?")
                return
            except Exception as e:
                print(f"\nFailed to process {file_name}: {e}")

    print(f"\nDone! Output saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing RGB images")
    parser.add_argument("--output", required=True, help="Directory to save .npz masks")
    parser.add_argument("--url", default="http://localhost:8080/predict", help="Server endpoint URL")
    args = parser.parse_args()

    process_directory(args.input, args.output, args.url)
