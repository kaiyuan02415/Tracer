#!/usr/bin/env python3
"""The two comparison baselines of Appendix A.3, and the paired comparison against TRACER.

bruteforce  render every candidate under W and W', score with CLIP, rank by the drop   Eq. (12)
mia         diffusion-loss MIA carried over candidates: rank by the rise in error      Eq. (13)

Every protocol choice here FAVOURS the baseline and each must be stated in the paper:
  * latents and noise are seeded from (candidate index, sample index) only, never from the
    checkpoint, so every model sees identical samples and sampling noise leaves the difference;
  * the MIA's samples are rendered by the BASE model, so it needs no data beyond the vocabulary.
    A data-rich MIA holding real photographs is a STRONGER attacker than the threat model allows,
    not a fairer one;
  * both ranking directions are scored and the better is credited -- under some erasures the loss
    on the erased concept falls rather than rises, and scoring one direction would report zero;
  * the base-model pass is computed once and amortised, as TRACER's vocabulary encoding is.

Phases:  cache (mia only) -> score (once per model) -> rank
"""
import argparse, glob, json, os, time

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def seeded(ci, j, shape):
    return torch.randn(*shape, generator=torch.Generator("cpu").manual_seed(ci * 10_000 + j))


def apply_ckpt(net, ckpt):
    from safetensors.torch import load_file
    live = net.state_dict()
    sd = load_file(ckpt, device=DEV)
    edit = {}
    for k, v in sd.items():                       # materialise pruning masks, as TRACER does
        if k.endswith(".mask"):
            wk = k[:-5] + ".weight"
            bk = wk if wk in live else wk.replace("unet.", "")
            if bk in live:
                edit[wk] = live[bk].float() * (1.0 - v.float())
        else:
            edit[k] = v
    n = 0
    for k, v in edit.items():
        bk = k if k in live else k.replace("unet.", "").replace("transformer.", "")
        assert bk in live, f"unmatched key {k}"
        live[bk].copy_(v.to(live[bk].dtype))
        n += 1
    return n


def build(arch, model_id):
    if arch == "flux":
        from diffusers import FluxPipeline
        p = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(DEV)
        return p, p.transformer
    if arch == "t2v":
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        p = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
        return p.to(DEV), p.unet
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    p = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    p.scheduler = DDIMScheduler.from_config(p.scheduler.config)
    return p.to(DEV), p.unet


# ------------------------------------------------------------------------- brute force, Eq. 12
def score_bruteforce(a, vocab):
    import open_clip
    pipe, net = build(a.arch, a.model_id)
    pipe.set_progress_bar_config(disable=True)
    if a.ckpt:
        print(f"applied {apply_ckpt(net, a.ckpt)} edited tensors", flush=True)
    torch.cuda.empty_cache()

    clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai", cache_dir=os.environ.get("HF_HOME"))
    clip = clip.to(DEV).eval()
    tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    with torch.no_grad():
        T = clip.encode_text(tok([a.prompt.format(c) for c in vocab]).to(DEV)).float()
        T = T / T.norm(dim=-1, keepdim=True)

    video = a.arch in ("t2v", "cogvideox", "wan")
    res, t0 = {}, time.time()
    for s in range(0, len(vocab), a.batch):
        chunk = list(enumerate(vocab))[s:s + a.batch]
        prompts = [a.prompt.format(c) for _, c in chunk for _ in range(a.n_img)]
        gens = [torch.Generator("cpu").manual_seed(ci * 10_000 + j)
                for ci, _ in chunk for j in range(a.n_img)]
        with torch.no_grad():
            kw = dict(prompt=prompts, num_inference_steps=a.steps, guidance_scale=a.guidance,
                      generator=gens, height=a.height, width=a.width, output_type="pt")
            if video:
                out = pipe(**kw, num_frames=a.frames).frames
                if out.shape[2] in (1, 3):                      # (N,F,C,H,W)
                    F = out.shape[1]
                    px = out.reshape(-1, *out.shape[2:]).float()
                else:                                           # (N,C,F,H,W)
                    F = out.shape[2]
                    px = out.permute(0, 2, 1, 3, 4).reshape(-1, out.shape[1],
                                                            *out.shape[3:]).float()
                per = a.n_img * F
            else:
                px = pipe(**kw).images.float()
                per = a.n_img
            x = torch.nn.functional.interpolate(px, 224, mode="bicubic", antialias=True).clamp(0, 1)
            E = clip.encode_image((x - CLIP_MEAN.to(DEV)) / CLIP_STD.to(DEV)).float()
            sim = (E / E.norm(dim=-1, keepdim=True)) @ T.T
        for b, (ci, c) in enumerate(chunk):
            sl = slice(b * per, (b + 1) * per)
            res[c] = {"idx": ci, "score": float(sim[sl, ci].mean()),
                      "top1": float((sim[sl].argmax(1) == ci).float().mean())}
        if (s // a.batch) % 20 == 0:
            el = time.time() - t0
            print(f"  {s + len(chunk)}/{len(vocab)}  {el:.0f}s  "
                  f"eta {el / (s + len(chunk)) * (len(vocab) - s) :.0f}s", flush=True)
    return res, time.time() - t0


# --------------------------------------------------------------------------------- MIA, Eq. 13
def cache_mia(a, vocab):
    pipe, _ = build("sd", a.model_id)
    pipe.set_progress_bar_config(disable=True)
    out, t0 = {}, time.time()
    per = max(1, 15 // a.n_img)
    for s in range(0, len(vocab), per):
        chunk = list(enumerate(vocab))[s:s + per]
        prompts, lat = [], []
        for ci, c in chunk:
            prompts += [a.prompt.format(c)] * a.n_img
            lat.append(torch.cat([seeded(ci, j, (1, 4, 64, 64)) for j in range(a.n_img)]))
        with torch.no_grad():
            x0 = pipe(prompt=prompts, latents=torch.cat(lat).to(DEV, torch.float16),
                      num_inference_steps=a.steps, guidance_scale=a.guidance,
                      output_type="latent").images
        for b, (_, c) in enumerate(chunk):
            out[c] = x0[b * a.n_img:(b + 1) * a.n_img].detach().half().cpu()
        if (s // per) % 20 == 0:
            print(f"  cached {s + len(chunk)}/{len(vocab)}  {time.time() - t0:.0f}s", flush=True)
    return out


def score_mia(a, vocab):
    from diffusers import DDPMScheduler
    pipe, net = build("sd", a.model_id)
    pipe.set_progress_bar_config(disable=True)
    X0 = torch.load(a.cache, map_location="cpu")
    if a.ckpt:
        print(f"applied {apply_ckpt(net, a.ckpt)} edited tensors", flush=True)
    torch.cuda.empty_cache()

    sched = DDPMScheduler.from_pretrained(a.model_id, subfolder="scheduler")
    TS = torch.tensor(a.timesteps, device=DEV)
    nT, n_img = len(a.timesteps), a.n_img
    per = max(1, 60 // (n_img * nT))
    res, t0 = {}, time.time()
    with torch.no_grad():
        for s in range(0, len(vocab), per):
            chunk = list(enumerate(vocab))[s:s + per]
            t = pipe.tokenizer([a.prompt.format(c) for _, c in chunk], padding="max_length",
                               max_length=77, truncation=True, return_tensors="pt")
            Emb = pipe.text_encoder(t.input_ids.to(DEV))[0]
            xs, es, ts, eh = [], [], [], []
            for b, (ci, c) in enumerate(chunk):
                x0 = X0[c].to(DEV, torch.float16)
                for j in range(n_img):
                    eps = seeded(ci, 100_000 + j, (1, 4, 64, 64)).to(DEV, torch.float16)
                    for k in range(nT):
                        xs.append(x0[j:j + 1]); es.append(eps)
                        ts.append(TS[k]); eh.append(Emb[b:b + 1])
            x0b, epsb, tb = torch.cat(xs), torch.cat(es), torch.stack(ts)
            xt = sched.add_noise(x0b.float(), epsb.float(), tb.cpu()).to(DEV, torch.float16)
            pred = net(xt, tb, encoder_hidden_states=torch.cat(eh)).sample.float()
            err = (pred - epsb.float()).pow(2).mean((1, 2, 3)).view(len(chunk), n_img * nT).mean(1)
            for b, (ci, c) in enumerate(chunk):
                res[c] = {"idx": ci, "score": float(err[b])}
            if (s // per) % 25 == 0:
                print(f"  {s + len(chunk)}/{len(vocab)}  {time.time() - t0:.0f}s", flush=True)
    return res, time.time() - t0


# ------------------------------------------------------------------------------------- ranking
def rank(a, vocab):
    import numpy as np
    truth = {}
    for line in list(open(a.groups_csv))[1:]:
        p = line.strip().split(",")
        if len(p) >= 3:
            truth[p[0]] = [c.strip() for c in p[2].split(";") if c.strip()]

    def load(tag):
        fs = sorted(glob.glob(os.path.join(a.score_dir, f"{tag}*.json")))
        if not fs:
            return None, 0.0
        sc, sec = {}, 0.0
        for f in fs:
            d = json.load(open(f))
            sc.update(d["scores"]); sec += d["seconds"]
        return sc, sec

    base, base_sec = load(a.base_tag)
    if base is None:
        raise SystemExit(f"no base-model scores under {a.score_dir}")
    tr = {}
    if a.tracer:
        tr = {r["ckpt"].replace(".safetensors", ""): r
              for r in json.load(open(a.tracer))["rows"]}

    rows = []
    for gid, cs in sorted(truth.items()):
        tag = next((t for t in [f"{a.prefix}-{gid}-N{len(cs)}", gid] if load(t)[0]), None)
        if tag is None:
            continue
        ed, sec = load(tag)
        if len(ed) != len(vocab):
            print(f"[skip] {gid}: {len(ed)}/{len(vocab)} candidates"); continue
        K, tset = len(cs), set(cs)
        d = np.array([ed[c]["score"] - base[c]["score"] for c in vocab])
        if a.attack == "bruteforce":
            d = -d                                   # a DROP in CLIP score is the evidence
        best = max(((np.argsort(-sgn * d), sgn) for sgn in (1.0, -1.0)),
                   key=lambda o: len({vocab[i] for i in o[0][:K]} & tset))
        order = best[0]
        rk = {vocab[i]: r + 1 for r, i in enumerate(order)}
        row = {"group": gid, "K": K, "gpu_seconds": sec,
               "recall_at_K": len({vocab[i] for i in order[:K]} & tset) / K,
               "recall_at_10": len({vocab[i] for i in order[:10]} & tset) / K,
               "truth_ranks": sorted(rk[c] for c in cs)}
        if a.attack == "bruteforce":
            # Why it fails, and it is not sampling noise: erasure damages concepts nobody asked
            # to erase, and in image space that damage is indistinguishable from the intended
            # erasure. Each of these outranks a real target; more samples will not separate them.
            drop = {c: base[c]["score"] - ed[c]["score"] for c in vocab}
            thr = float(np.mean([drop[c] for c in cs]))
            row["n_collateral"] = sum(1 for c in vocab if c not in tset and drop[c] >= thr)
        rows.append(row)
        print(f"  {gid:14s} recall@{K} {row['recall_at_K']:.2f}  {sec / 3600:.2f} GPU-h")

    if not rows:
        raise SystemExit("nothing to rank")
    m = lambda k: float(np.mean([r[k] for r in rows if k in r]))        # noqa: E731
    print(f"\n=== {a.attack} over {len(rows)} checkpoints ===")
    print(f"  recall@K           {m('recall_at_K'):.3f}")
    print(f"  recall@10          {m('recall_at_10'):.3f}")
    print(f"  median truth rank  {int(np.median([x for r in rows for x in r['truth_ranks']]))}"
          f" of {len(vocab)}")
    print(f"  cost per ckpt      {m('gpu_seconds') / 3600:.2f} GPU-h "
          f"(+ {base_sec / 3600:.2f} GPU-h once)")
    if "n_collateral" in rows[0]:
        print(f"  collateral         {m('n_collateral'):.1f} non-erased candidates lose at "
              f"least as much as the average erased one")

    # paired comparison: only the checkpoints BOTH measured, else the two sides average over
    # different checkpoints and the table silently compares different things
    if tr:
        both = [r for r in rows if r["group"] in tr]
        if both:
            t_rec = float(np.mean([tr[r["group"]].get("recall_at_K", 0) for r in both]))
            t_sec = float(np.mean([tr[r["group"]]["seconds"] for r in both]))
            b_sec = float(np.mean([r["gpu_seconds"] for r in both]))
            print(f"\n=== paired on the same {len(both)} checkpoints ===")
            print(f"  baseline recall@K  {np.mean([r['recall_at_K'] for r in both]):.3f}"
                  f"   {b_sec / 3600:.2f} GPU-h")
            print(f"  TRACER   recall@K  {t_rec:.3f}   {t_sec:.2f} s")
            print(f"  speedup            {b_sec / max(t_sec, 1e-9):.0f}x")
            print(f"  unpaired: baseline-only {len(rows) - len(both)}, "
                  f"TRACER-only {len(set(tr) - {r['group'] for r in rows})}")
    json.dump({"attack": a.attack, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", required=True, choices=["bruteforce", "mia"])
    ap.add_argument("--phase", required=True, choices=["cache", "score", "rank"])
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--arch", default="sd", choices=["sd", "sd3", "flux", "t2v"])
    ap.add_argument("--ckpt", default="", help="score phase; empty = the base model")
    ap.add_argument("--cache", default="", help="mia score phase")
    ap.add_argument("--prompt", default="a photo of a {}")
    ap.add_argument("--n_img", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--timesteps", type=int, nargs="*", default=[100, 300, 500, 700, 900])
    ap.add_argument("--score_dir", default="", help="rank phase")
    ap.add_argument("--groups_csv", default="", help="rank phase")
    ap.add_argument("--base_tag", default="base", help="rank phase")
    ap.add_argument("--prefix", default="uce", help="rank phase: checkpoint filename prefix")
    ap.add_argument("--tracer", default="", help="rank phase: TRACER result json, to pair against")
    a = ap.parse_args()
    vocab = json.load(open(a.vocab))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    if a.phase == "cache":
        if a.attack != "mia":
            raise SystemExit("only --attack mia has a cache phase")
        torch.save(cache_mia(a, vocab), a.out)
    elif a.phase == "score":
        if a.attack == "mia":
            if not a.cache:
                raise SystemExit("--cache is required")
            res, sec = score_mia(a, vocab)
        else:
            res, sec = score_bruteforce(a, vocab)
        json.dump({"ckpt": a.ckpt, "seconds": sec, "scores": res}, open(a.out, "w"))
    else:
        if not (a.score_dir and a.groups_csv):
            raise SystemExit("--score_dir and --groups_csv are required")
        rank(a, vocab)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
