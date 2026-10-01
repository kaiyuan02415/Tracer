# Your Unlearning Gives You Away: Identifying Erased Concepts in Diffusion Models

[![Paper](https://img.shields.io/badge/ICLR-2027-b31b1b.svg)](.)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **TRACER** (ICLR 2027).

Attacks on unlearned diffusion models assume the erased concepts are already known. In practice a
provider does not disclose what was removed, so an adversary has no target to attack. **TRACER**
answers the prior question — *which* concepts were erased, and *how many* — from the weights
alone, in seconds. It builds Gram matrices from the weight difference `W' - W`, groups them by the
representation they receive, and greedily selects the candidate set whose span explains the
footprint; the erased-set size is read off the decline in candidate confidence. No image
generation, no knowledge of the unlearning algorithm, no labelled calibration data.

Across text-to-image and text-to-video backbones it reaches **150–137,000×** and **133–20,000×**
speedups over membership-inference and brute-force search, with substantially higher accuracy.

## Installation

Python 3.10, PyTorch 2.6, CUDA 12.4+, one NVIDIA GPU (RTX A6000 in the paper).

```bash
conda create -n tracer python=3.10 -y && conda activate tracer
pip install -r requirements.txt
```

Base models are downloaded from the HuggingFace Hub on first use.

## Usage

One command, one checkpoint directory in, the recovered concepts and their count out:

```bash
python tracer/tracer.py \
    --arch sd --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --ckpt_dir checkpoints/UCE/imagenet1k-K10 \
    --vocab dataset/vocab/imagenet1k_1000.json
```

```
[sd] 10 ckpts | 32 imprints | 1 group(s)
[sd] encoded 1000 candidates in 7.4s
  uce-G10a-N10   n^=10   0.38s  ['beaver', 'mobile home', 'fox squirrel', ...]
```

Pass `--groups_csv` to also score against ground truth; the attack itself never reads it.
Other backbones need nothing but `--arch` and `--model_id`:

```bash
python tracer/tracer.py --arch flux      --model_id black-forest-labs/FLUX.1-schnell  ...
python tracer/tracer.py --arch sd3       --model_id stabilityai/stable-diffusion-3-medium-diffusers ...
python tracer/tracer.py --arch cogvideox --model_id THUDM/CogVideoX-5b ...
python tracer/tracer.py --arch wan       --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers ...
```

| Flag | |
|---|---|
| `--templates art` | artist vocabulary needs art phrasings; `"a photo of a {}"` is not neutral for a painter |
| `--internal` | the erasure touches no text-conditioned matrix (e.g. ESD on FLUX): group per attention module and read internal activations instead |
| `--phrase` | Appendix A.2 — construct multiword candidates instead of ranking the vocabulary |
| `--beta --gamma --rho --offset` | the self-calibration grid; the default is the paper's 24 settings |
| `--horizon` | search horizon `J`; bounds the size reading only, not recall at a known `K` |
| `--rho 1` / `--gamma 0` / `--beta 0` / `--largest_drop_only` | ablation arms |

See [`tracer/run.sh`](tracer/run.sh) for the full set of experiment commands.

## Evaluation

```bash
# brute-force generation probing (Eq. 12) and the adapted MIA (Eq. 13)
python utils/baselines.py --attack bruteforce --phase score --ckpt "" --out results/bf/base.json ...
python utils/baselines.py --attack bruteforce --phase rank  --score_dir results/bf \
       --groups_csv dataset/groups/imagenet1k_K10.csv --tracer results/uce_k10.json ...
```

`--phase rank` reports recall, the collateral diagnostic, and a comparison paired per checkpoint
against a TRACER result file. Both baselines are given every advantage — identical seeded latents,
base-model samples, and credit for the better of the two ranking directions.

Released `.pt` checkpoints can be normalised first with
[`utils/convert.py`](utils/convert.py), which records the base model in the metadata.

## Data

`dataset/groups/*.csv` are `group_id,N,concepts`: one row per released checkpoint, listing the
concepts it jointly erased. `dataset/vocab/*.json` are candidate vocabularies.

| | |
|---|---|
| `imagenet1k_K10.csv`, `imagenet1k_ladder.csv` | the main benchmark and the `K = 1,3,5,10,20` ladder, ten disjoint groups per rung |
| `artist_single.csv`, `artist_K3.csv`, `artist_K5.csv` | artistic styles |
| `imagenet1k_1000.json` | closed vocabulary — used against the baselines, whose cost caps the pool |
| `openset_73406.json` | open vocabulary — used for ablations, where that cap does not apply |
| `artists_1734.json` | artist names |

Numbers from different vocabularies are not comparable; every table must state the pool it used.
Checkpoints must be real joint erasures — summing `K` single-concept weight differences is a
constructive control, not a multi-concept result.

## Citation

```bibtex
@inproceedings{tracer2027,
  title     = {Your Unlearning Gives You Away: Identifying Erased Concepts in Diffusion Models},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2027}
}
```

## Acknowledgements

We evaluate against the official implementations and released checkpoints of ESD, UCE, MACE, FMN,
ConceptPrune, SPEED, ScaPre, STEREO, AdvUnlearn, EraseAnything, EraseFlow, VideoEraser and others,
used without modification. The open vocabulary combines the ImageNet-1k class names with the
[GNU Aspell](https://aspell.net/) word list.

## License

MIT, see [`LICENSE`](LICENSE). Base models, unlearning implementations and evaluation datasets
carry their own licenses.
