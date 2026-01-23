import asyncio
import websockets
import msgpack
import os
import glob
import numpy as np
import argparse
from tqdm.asyncio import tqdm

# We don't need msgpack_numpy because we are sending PATHS, not pixels.
# This makes the script much more robust.

async def process_images(input_dir, output_dir, server_uri, prompt_text):
    # 1. Setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")) + 
                         glob.glob(os.path.join(input_dir, "*.jpg")) +
                         glob.glob(os.path.join(input_dir, "*.jpeg")))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images.")
    print(f"Connecting to {server_uri}...")
    print(f"Using Prompt: '{prompt_text}'")

    # 2. Open WebSocket Connection
    async with websockets.connect(server_uri, max_size=None) as websocket:
        print("Connected!")
        
        for img_path in tqdm(image_files, desc="Processing"):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, base_name + ".npz")
            
            # Resume capability
            if os.path.exists(save_path):
                continue

            try:
                # 3. Construct Payload (VideoRequest Mode)
                # Instead of sending pixels, we send the absolute path to the file.
                # The server loads the file from disk itself.
                abs_path = os.path.abspath(img_path)
                
                payload = {
                    "type": "video",          # Switch to 'video' mode to use path
                    "resource_path": abs_path, # Server opens this file
                    "text": prompt_text,
                    "frame_index": 0          # Treat image as frame 0
                }

                # 4. Send (Pack as Binary MsgPack)
                await websocket.send(msgpack.packb(payload))
                
                # 5. Receive
                response_bytes = await websocket.recv()
                
                # Unpack the response
                # Note: We use strict_map_key=False to avoid str/bytes issues on response
                response_data = msgpack.unpackb(response_bytes, strict_map_key=False)
                
                # 6. Extract Mask
                mask_data = None
                
                # Check for standard keys
                if isinstance(response_data, dict):
                    target_keys = ['masks', 'mask', 'segmentation', 'prediction', 'outputs']
                    for key in target_keys:
                         # Check both string and bytes keys just in case
                        if key in response_data:
                            mask_data = response_data[key]
                            break
                        if key.encode() in response_data:
                            mask_data = response_data[key.encode()]
                            break
                    
                    # Fallback: if dict has no known keys, maybe the dict IS the data?
                    if mask_data is None and 'type' not in response_data:
                         # Sometimes result is just the data wrapped
                         mask_data = response_data
                else:
                    # If it's a direct list/array
                    mask_data = response_data

                # Convert to numpy
                if mask_data is not None:
                    mask_np = np.array(mask_data, dtype=np.uint8)
                    
                    # Flatten dimensions if needed (1, H, W) -> (H, W)
                    if mask_np.ndim == 3:
                        mask_np = mask_np[0]
                    
                    # 7. Save
                    np.savez_compressed(save_path, masks=mask_np)
                else:
                    print(f"Warning: No mask data found in response for {base_name}")

            except Exception as e:
                print(f"\nFailed on {base_name}: {e}")

    print(f"\nDone! Saved masks to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images using SAM WebSocket (Path Mode)")
    
    parser.add_argument("--input", required=True, help="Directory containing RGB images")
    parser.add_argument("--output", required=True, help="Directory to save .npz masks")
    parser.add_argument("--prompt", required=True, help="Text prompt for segmentation")
    parser.add_argument("--url", default="ws://localhost:8080", help="WebSocket server URI")

    args = parser.parse_args()

    try:
        asyncio.run(process_images(args.input, args.output, args.url, args.prompt))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except ConnectionRefusedError:
        print(f"\nError: Could not connect to {args.url}. Is the server running?")
