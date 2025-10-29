# metrics_eval.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.inception import inception_v3, Inception_V3_Weights

from scipy import linalg
from transformers import CLIPProcessor, CLIPModel


class InceptionWrap(nn.Module):
    """
    Provides:
      - pool3 features (2048-D) for FID
      - logits for Inception Score
    """
    def __init__(self, device: torch.device):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        # Build with pretrained weights; do not override aux_logits in ctor to avoid torchvision guard
        model = inception_v3(weights=weights, transform_input=False)
        # Disable aux branch post-construction to get a single logits tensor
        model.aux_logits = False
        model.AuxLogits = None
        self.model = model.to(device).eval()
        self.device = device
        self.preprocess = T.Compose([
            T.Resize(299, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(299),
            T.ToTensor(),
            T.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ])

    @torch.inference_mode()
    def activations_and_logits(self, pil_images: List[Image.Image], batch: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        feats_list, logits_list = [], []
        for i in tqdm(range(0, len(pil_images), batch), desc="Inception pass"):
            imgs = pil_images[i:i+batch]
            x = torch.stack([self.preprocess(img.convert("RGB")) for img in imgs], dim=0).to(self.device, non_blocking=True)
            captured = []
            def hook(_, __, outp): captured.append(outp)
            h = self.model.avgpool.register_forward_hook(hook)
            logits_batch = self.model(x)
            h.remove()
            feat = captured[0].reshape(captured[0].size(0), -1)
            feats_list.append(feat.cpu().numpy())
            logits_list.append(logits_batch.detach().cpu().numpy())
        feats = np.concatenate(feats_list, axis=0) if feats_list else np.empty((0, 2048), dtype=np.float32)
        logits = np.concatenate(logits_list, axis=0) if logits_list else np.empty((0, 1000), dtype=np.float32)
        return feats, logits

    @torch.inference_mode()
    def activations_and_logits_from_paths(self, image_paths: List[str], batch: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        pil_batch: List[Image.Image] = []
        feats_list, logits_list = [], []
        for idx, path in enumerate(tqdm(image_paths, desc="Inception pass (paths)")):
            with Image.open(path) as im:
                pil_batch.append(im.convert("RGB"))
            if len(pil_batch) == batch or idx == len(image_paths) - 1:
                f, l = self.activations_and_logits(pil_batch, batch=batch)
                feats_list.append(f); logits_list.append(l)
                pil_batch = []
        feats = np.concatenate(feats_list, axis=0) if feats_list else np.empty((0, 2048), dtype=np.float32)
        logits = np.concatenate(logits_list, axis=0) if logits_list else np.empty((0, 1000), dtype=np.float32)
        return feats, logits


def compute_moments(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))

def inception_score_from_logits(logits: np.ndarray, splits: int = 10) -> float:
    if logits.size == 0:
        return float("nan")
    N = logits.shape[0]
    splits = max(1, min(int(splits), int(N)))
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / np.sum(e, axis=1, keepdims=True)
    base = N // splits
    split_sizes = [base] * splits
    for i in range(N % splits):
        split_sizes[i] += 1
    scores, start = [], 0
    for s in split_sizes:
        if s == 0: continue
        p = probs[start:start+s, :]; start += s
        py = np.mean(p, axis=0, keepdims=True)
        kl = p * (np.log(p + 1e-8) - np.log(py + 1e-8))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(scores)) if scores else float("nan")

@dataclass
class CLIPScorer:
    device: torch.device
    hf_model_id: str = "openai/clip-vit-base-patch32"
    # def __post_init__(self):
    #     self.model = CLIPModel.from_pretrained(self.hf_model_id).to(self.device).eval()
    #     self.processor = CLIPProcessor.from_pretrained(self.hf_model_id)
    def __post_init__(self):
        # 使用safetensors格式，避免torch.load的安全问题
        self.model = CLIPModel.from_pretrained(
            self.hf_model_id, 
            use_safetensors=True
        ).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.hf_model_id)
    
    
    
    
    @torch.inference_mode()
    def score(self, images: List[Image.Image], texts: List[str], batch: int = 32) -> float:
        assert len(images) == len(texts)
        sims = []
        for i in tqdm(range(0, len(images), batch), desc="CLIP score"):
            imgs = [im.convert("RGB") for im in images[i:i+batch]]
            txts = texts[i:i+batch]
            proc = self.processor(text=txts, images=imgs, return_tensors="pt", padding=True)
            pixel_values = proc["pixel_values"].to(self.device)
            input_ids = proc["input_ids"].to(self.device)
            attention_mask = proc["attention_mask"].to(self.device)
            img_feat = self.model.get_image_features(pixel_values=pixel_values)
            txt_feat = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            cos = (img_feat * txt_feat).sum(dim=-1)
            sims.append((2.5 * torch.clamp(cos, min=0.0)).cpu())
        sims = torch.cat(sims, dim=0)
        return float(sims.mean().item())
