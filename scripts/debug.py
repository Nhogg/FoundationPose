import asyncio
import websockets
import msgpack
import os
import argparse
import numpy as np
import cv2

# Better prompts for that specific object
PROMPTS = ["brush", "scrub brush", "wooden brush", "bristles", "cleaning tool", "wood"]

async def force_find_object(img_path, server_uri):
    abs_path = os.path.abspath(img_path)
    if not os.path.exists(abs_path):
        print(f"❌ File not found: {abs_path}")
        return

    print(f"🔹 Analyzing: {abs_path}")
    print(f"🔹 Connecting to {server_uri}...")

    async with websockets.connect(server_uri, max_size=None) as websocket:
        print("✅ Connected.")
        
        for prompt in PROMPTS:
            print(f"👉 Asking for: '{prompt}'...")
            
            payload = {
                "type": "video",
                "resource_path": abs_path,
                "text": prompt,
                "frame_index": 0,
                "confidence": 0.1  # Low confidence to make sure we catch it
            }

            # Send
            await websocket.send(msgpack.packb(payload))
            
            # Receive
            response_bytes = await websocket.recv()
            data = msgpack.unpackb(response_bytes, strict_map_key=False)

            # Check for data
            mask_found = None
            if isinstance(data, dict):
                for k in ['masks', 'mask', 'segmentation']:
                    val = data.get(k) or data.get(k.encode())
                    if val is not None:
                        mask_found = val
                        break
            
            if mask_found is not None:
                print(f"🔥 SUCCESS! Found object with prompt: '{prompt}'")
                
                # Save it immediately
                mask_np = np.array(mask_found, dtype=np.uint8)
                if mask_np.ndim == 3: mask_np = np.max(mask_np, axis=0)
                
                # Load original for debug overlay
                orig = cv2.imread(abs_path)
                
                # Resize mask if needed (sometimes model outputs lower res)
                if mask_np.shape != (orig.shape[0], orig.shape[1]):
                    mask_np = cv2.resize(mask_np, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Save NPZ
                np.savez_compressed("my_data/masks.npz", masks=mask_np)
                print("💾 Saved valid mask to: my_data/masks.npz")
                
                # Save Debug Image
                colored = np.zeros_like(orig)
                colored[:, :, 1] = mask_np * 255 # Green
                debug = cv2.addWeighted(orig, 0.7, colored, 0.3, 0)
                cv2.imwrite("final_debug.png", debug)
                print("🖼️  Saved visual check: final_debug.png")
                print("👉 View it: feh final_debug.png")
                return

        print("❌ Failed. The model ignored all prompts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--url", default="ws://localhost:8080")
    args = parser.parse_args()
    
    asyncio.run(force_find_object(args.input, args.url))
