#!/usr/bin/env python3
"""Normalise released .pt checkpoints into the safetensors layout the attack reads.

--keep all         the file already holds exactly the edited matrices (a container change)
--keep cross_attn  the file holds a whole denoiser; keep only the text-conditioned projections

The base model MUST travel with the checkpoint. A wrong base does not crash anything: it turns
the difference between two base models into a fake footprint and drives recovery to zero.
"""
import argparse, glob, os, torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", required=True, help="a .pt file or a directory of them")
ap.add_argument("--dst", default="")
ap.add_argument("--keep", default="all", choices=["all", "cross_attn"])
ap.add_argument("--base_model_id", required=True)
ap.add_argument("--concepts", default="", help="';'-separated ground truth, kept in the metadata")
a = ap.parse_args()

srcs = [a.src] if a.src.endswith(".pt") else sorted(glob.glob(os.path.join(a.src, "*.pt")))
if not srcs:
    raise SystemExit(f"no .pt under {a.src}")
dst = a.dst or os.path.dirname(srcs[0]) or "."
os.makedirs(dst, exist_ok=True)
meta = {"base_model_id": a.base_model_id}
if a.concepts:
    meta["erase_concept"] = a.concepts

for s in srcs:
    sd = torch.load(s, map_location="cpu")
    keep = {k: v.float().contiguous() for k, v in sd.items()
            if a.keep == "all" or k.endswith(("attn2.to_k.weight", "attn2.to_v.weight"))}
    if not keep:
        raise SystemExit(f"{s}: nothing matched --keep {a.keep}")
    out = os.path.join(dst, os.path.basename(s)[:-3] + ".safetensors")
    save_file(keep, out, metadata=meta)
    print(f"  {os.path.basename(s)} -> {os.path.basename(out)}  ({len(keep)} tensors)")
print(f"{len(srcs)} converted, base={a.base_model_id}")
