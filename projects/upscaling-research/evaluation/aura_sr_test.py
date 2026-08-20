from aura_sr import AuraSR
from PIL import Image
import argparse
import time
import os
import torch

def run_upscale(orig_path, out_path, device):
    model = AuraSR.from_pretrained("fal/AuraSR-v2").to(device)
    inference_times = []
    max_vram_overall = 0.0
    total_start_time = time.time()
    log_file_path = os.path.join(out_path, "log.txt")

    image_files = sorted([file for file in os.listdir(orig_path) if file.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_files:
        print("No images found!")
        return
        
    os.makedirs(out_path, exist_ok=True)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    for image_name in image_files:
        img_path = os.path.join(orig_path, image_name)
        base_name = os.path.splitext(image_name)[0]

        img = Image.open(img_path).convert("RGB")
        orig_width, orig_height = img.size

        start = time.time()
        img_4x = model.upscale_4x_overlapped(img)
        inf_time = time.time() - start
        
        if device.type == 'cuda':
            max_vram_overall = max(max_vram_overall, torch.cuda.max_memory_allocated() / 1024 ** 2)

        inference_times.append(inf_time)
        
        img_3x = img_4x.resize((orig_width * 3, orig_height * 3), Image.LANCZOS)
        img_3x.save(os.path.join(out_path, f"{base_name}_3x.png"))
        
        img_2x = img_4x.resize((orig_width * 2, orig_height * 2), Image.LANCZOS)
        img_2x.save(os.path.join(out_path, f"{base_name}_2x.png"))

        print(f"Processed {image_name} | Inference time: {inf_time:.4f} seconds")
    
    total_time = time.time() - total_start_time
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    print(f"Average inference time: {avg_inference_time:.4f} seconds")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Max VRAM usage: {max_vram_overall:.2f} MB")

    with open(log_file_path, "w") as f:
        f.write(f"Average inference time: {avg_inference_time:.4f} seconds\n")
        f.write(f"Total time: {total_time:.4f} seconds\n")
        f.write(f"Max VRAM usage: {max_vram_overall:.2f} MB\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--orig", type=str, required=True, help="Path to the original images.")
    args.add_argument("--out", type=str, required=True, help="Path to save upscaled images.")
    args.add_argument("--device", type=str, default="cpu")
    args = args.parse_args()
    
    run_upscale(args.orig, args.out, torch.device(args.device))
