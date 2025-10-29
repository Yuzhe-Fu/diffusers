# mscoco_local.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json, os, random, shutil, zipfile, argparse
from urllib.request import urlretrieve
from PIL import Image
from glob import glob

@dataclass
class MSCOCOVal2017:
    root: str                        # /home/zs89/EchoFlow/mscoco
    split: str = "val2017"
    captions_json: str = "annotations/captions_val2017.json"
    num_samples: int = 5000
    seed: int = 1234
    pick: str = "random"             # random or first

    def _index(self) -> Tuple[List[str], Dict[int, str], Dict[int, List[str]]]:
        # Load captions JSON
        cap_path = os.path.join(self.root, self.captions_json)
        with open(cap_path, "r") as f:
            ann = json.load(f)

        # Map image id -> file path
        id_to_path = {}
        split_dir = os.path.join(self.root, self.split)
        for img in ann["images"]:
            fn = img.get("file_name", "")
            path = os.path.join(split_dir, fn)
            id_to_path[img["id"]] = path

        # Map image id -> list of captions
        caps: Dict[int, List[str]] = {}
        for a in ann["annotations"]:
            caps.setdefault(a["image_id"], []).append(a.get("caption", "").strip())

        # All existing image paths in split (for FID reference)
        all_paths = sorted(glob(os.path.join(split_dir, "*.jpg")))
        return all_paths, id_to_path, caps

    def load(self) -> Tuple[List[str], List[str], List[str]]:
        rng = random.Random(self.seed)
        all_val_paths, id_to_path, caps = self._index()

        # Build (path, caption) pairs, filtering missing files
        pairs: List[Tuple[str, str]] = []
        for img_id, path in id_to_path.items():
            if not os.path.exists(path):
                continue
            cap_list = caps.get(img_id, [])
            if len(cap_list) == 0:
                caption = "a photo"
            else:
                caption = cap_list[0] if self.pick == "first" else cap_list[rng.randrange(len(cap_list))]
            pairs.append((path, caption))

        # Deterministic shuffle & take num_samples
        rng.shuffle(pairs)
        pairs = pairs[: min(self.num_samples, len(pairs))]
        img_paths_subset = [p for p, _ in pairs]
        prompts_subset = [t for _, t in pairs]
        return img_paths_subset, prompts_subset, all_val_paths

    @staticmethod
    def open_pil(path: str) -> Image.Image:
        with Image.open(path) as im:
            return im.convert("RGB")


def _progress_hook(t):
    # Simple tqdm-like hook for urlretrieve
    last_b = [0]
    def inner(b, bsize, tsize):
        if tsize > 0:
            done = b * bsize
            if done - last_b[0] >= max(5 * 1024 * 1024, tsize // 100):  # 5MB or 1%
                last_b[0] = done
                pct = min(100.0, 100.0 * done / tsize)
                print(f"  downloaded {done/1e6:.1f} MB / {tsize/1e6:.1f} MB ({pct:.1f}%)", end="\r")
    return inner


def ensure_coco_val2017(root: str):
    """
    Ensure MSCOCO val2017 images and captions are present under `root`.
    Downloads and extracts from official URLs if missing:
      - val2017 images (5k)
      - annotations (captions_val2017.json)
    """
    os.makedirs(root, exist_ok=True)
    split_dir = os.path.join(root, "val2017")
    ann_dir = os.path.join(root, "annotations")
    cap_json = os.path.join(ann_dir, "captions_val2017.json")

    need_imgs = not (os.path.isdir(split_dir) and len(os.listdir(split_dir)) >= 1)
    need_ann = not os.path.isfile(cap_json)

    if not need_imgs and not need_ann:
        return

    if need_imgs:
        url = "http://images.cocodataset.org/zips/val2017.zip"
        zip_path = os.path.join(root, "val2017.zip")
        print(f"[COCO] Downloading val2017 images -> {zip_path}")
        urlretrieve(url, zip_path, _progress_hook(None))
        print("\n[COCO] Extracting val2017 ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(root)
        os.remove(zip_path)
        print("[COCO] val2017 ready.")

    if need_ann:
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        zip_path = os.path.join(root, "annotations_trainval2017.zip")
        print(f"[COCO] Downloading annotations -> {zip_path}")
        urlretrieve(url, zip_path, _progress_hook(None))
        print("\n[COCO] Extracting annotations ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(root)
        os.remove(zip_path)
        print("[COCO] annotations ready.")

    # Final sanity
    if not os.path.isdir(split_dir) or not os.path.isfile(cap_json):
        raise RuntimeError("Failed to prepare MSCOCO val2017. Check network and disk space.")


def _cli():
    ap = argparse.ArgumentParser(description="Prepare and verify MSCOCO val2017 locally.")
    ap.add_argument("--root", type=str, required=True, help="Target directory for MSCOCO files")
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--pick", type=str, default="random", choices=["random", "first"]) 
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    ensure_coco_val2017(args.root)
    ds = MSCOCOVal2017(root=args.root, num_samples=args.num_samples, seed=args.seed, pick=args.pick)
    subset_paths, prompts, all_paths = ds.load()
    print(f"[COCO] val2017 ready at {args.root}")
    print(f"[COCO] Total val images: {len(all_paths)}; subset: {len(subset_paths)} prompts")
    print("[COCO] Sample prompts:")
    for i in range(min(5, len(prompts))):
        print(f"  - {prompts[i]}")


if __name__ == "__main__":
    _cli()
