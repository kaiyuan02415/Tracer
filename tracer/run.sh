#!/usr/bin/env bash
# Example commands. Set CKPT_ROOT to the directory holding the released checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

SD=${SD:-stable-diffusion-v1-5/stable-diffusion-v1-5}
CKPT_ROOT=${CKPT_ROOT:-checkpoints}
mkdir -p results

# --- identify erased concepts and their number (closed vocabulary, 1k ImageNet names)
python tracer/tracer.py --arch sd --model_id "$SD" \
    --ckpt_dir "$CKPT_ROOT/UCE/imagenet1k-K10" \
    --vocab dataset/vocab/imagenet1k_1000.json \
    --groups_csv dataset/groups/imagenet1k_K10.csv \
    --out_json results/uce_k10.json

# --- the K = 1,3,5,10,20 ladder; the size is read per checkpoint from its own row
python tracer/tracer.py --arch sd --model_id "$SD" \
    --ckpt_dir "$CKPT_ROOT/UCE/ladder" \
    --vocab dataset/vocab/imagenet1k_1000.json \
    --groups_csv dataset/groups/imagenet1k_ladder.csv \
    --out_json results/uce_ladder.json

# --- open vocabulary (73,406). Use this for ablations: on the 1k pool most arms sit at the
#     ceiling and the differences are smaller than one selection.
python tracer/tracer.py --arch sd --model_id "$SD" \
    --ckpt_dir "$CKPT_ROOT/ESD/ladder" \
    --vocab dataset/vocab/openset_73406.json \
    --groups_csv dataset/groups/imagenet1k_ladder.csv \
    --rho 1 --out_json results/esd_noshape.json          # w/o spectral shaping

# --- artistic styles: artist vocabulary + art phrasings ("a photo of a {}" is not neutral)
python tracer/tracer.py --arch sd --model_id "$SD" \
    --ckpt_dir "$CKPT_ROOT/UCE/artist-K5" --templates art \
    --vocab dataset/vocab/artists_1734.json \
    --groups_csv dataset/groups/artist_K5.csv

# --- other backbones: nothing changes but --arch and --model_id
python tracer/tracer.py --arch flux --model_id black-forest-labs/FLUX.1-schnell \
    --ckpt_dir "$CKPT_ROOT/UCE-Flux/imagenet1k-K10" \
    --vocab dataset/vocab/imagenet1k_1000.json
python tracer/tracer.py --arch cogvideox --model_id THUDM/CogVideoX-5b \
    --ckpt_dir "$CKPT_ROOT/Refusal/cogvideox-K10" \
    --vocab dataset/vocab/imagenet1k_1000.json

# --- erasure that touches no text-conditioned matrix: read internal activations instead
python tracer/tracer.py --arch flux --model_id black-forest-labs/FLUX.1-dev --internal \
    --ckpt_dir "$CKPT_ROOT/ESD-Flux/imagenet1k-K10" \
    --vocab dataset/vocab/imagenet1k_1000.json

# --- baselines, on the SAME checkpoints and vocabulary
python utils/baselines.py --attack bruteforce --phase score --model_id "$SD" \
    --ckpt "" --vocab dataset/vocab/imagenet1k_1000.json --out results/bf/base.json
for ck in "$CKPT_ROOT"/UCE/imagenet1k-K10/*.safetensors; do
    python utils/baselines.py --attack bruteforce --phase score --model_id "$SD" \
        --ckpt "$ck" --vocab dataset/vocab/imagenet1k_1000.json \
        --out "results/bf/$(basename "$ck" .safetensors).json"
done
python utils/baselines.py --attack bruteforce --phase rank --score_dir results/bf \
    --vocab dataset/vocab/imagenet1k_1000.json \
    --groups_csv dataset/groups/imagenet1k_K10.csv \
    --tracer results/uce_k10.json --out results/bf_uce.json
