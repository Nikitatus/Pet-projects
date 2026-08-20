import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import re
import argparse
import json

def init_metrics(metric_names=None, device=torch.device("cpu")):
    if metric_names is None:
        metric_names = ["psnr", "ssim", "lpips", "fid"]

    metrics = {}
    if "psnr" in metric_names:
        metrics["psnr"] = PeakSignalNoiseRatio(data_range=1.0).to(device)
    if "ssim" in metric_names:
        metrics["ssim"] = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    if "lpips" in metric_names:
        metrics["lpips"] = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
    if "fid" in metric_names:
        metrics["fid"] = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    return metrics

class ImagePairDataset(Dataset):
    def __init__(self, orig_path, upscaled_path):
        def extract_id(p):
            m = re.match(r"image_(\d+)", p.stem)
            return int(m.group(1)) if m else None

        orig_map = {}
        for p in Path(orig_path).glob("*"):
            fid = extract_id(p)
            if fid is not None:
                orig_map[fid] = p
        
        upscaled_map = {}
        for p in Path(upscaled_path).glob("*"):
            fid = extract_id(p)
            if fid is not None:
                upscaled_map[fid] = p
        
        ids = sorted(set(orig_map.keys()) & set(upscaled_map.keys()))
        
        self.orig_paths = [orig_map[i] for i in ids]
        self.upscaled_paths = [upscaled_map[i] for i in ids]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.orig_paths)

    def __getitem__(self, idx):
        img_orig = Image.open(self.orig_paths[idx]).convert("RGB")
        img_upscaled = Image.open(self.upscaled_paths[idx]).convert("RGB")

        if img_orig.size != img_upscaled.size:
            img_orig = img_orig.resize(img_upscaled.size, resample=Image.LANCZOS)
            
        img_orig = self.transform(img_orig)
        img_upscaled = self.transform(img_upscaled)

        return img_orig, img_upscaled

def update_metrics(originals, upscaled, metrics):
    for metric_name in ["psnr", "ssim", "lpips"]:
        if metric_name in metrics:
            metrics[metric_name](upscaled, originals)

    if "fid" in metrics:
        metrics["fid"].update(originals, real=True)
        metrics["fid"].update(upscaled, real=False)

def evaluate(orig_path, upscaled_path, metric_names, num_workers, device):
    metrics = init_metrics(metric_names=metric_names, device=device)    
    dataset = ImagePairDataset(orig_path, upscaled_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    
    with torch.no_grad():
        for i, (orig, upscaled) in enumerate(dataloader):
            orig = orig.to(device)
            upscaled = upscaled.to(device)
            
            update_metrics(orig, upscaled, metrics)

    results = {metric_name: metric.compute().item() for metric_name, metric in metrics.items()}
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", type=str, required=True, help="Path to original images.")
    parser.add_argument("--upscaled", type=str, required=True, help="Path to upscaled images.")
    parser.add_argument("--metrics", type=str, nargs="+", default=["psnr", "ssim", "lpips", "fid"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    results = evaluate(args.orig, args.upscaled, metric_names=args.metrics, num_workers=args.num_workers, device=torch.device(args.device))
    print(json.dumps(results, indent=4))