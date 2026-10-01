#!/usr/bin/env python3
"""TRACER -- identify which concepts a diffusion model was made to forget, and how many.

in   base weights W, released checkpoint(s) W', a candidate vocabulary C
out  the recovered concept set S_hat and its size n, per checkpoint

Weights only: the sampling loop is never run and no image is decoded.
--groups_csv is optional and used for scoring only; the attack never reads it.
"""
import argparse, glob, json, os, re, time
from collections import defaultdict
from itertools import combinations

import torch
from safetensors.torch import load_file

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TEMPLATES = {
    "object": ["a photo of a {}", "{}", "an image of a {}", "a picture of a {}",
               "a photo of the {}"],
    "art": ["art by {}", "{}", "painting by {}", "artwork by {}", "style of {}"],
}
TAU = 1e-8                                  # imprint retention threshold
EPS0, KAPPA = 1e-3, 0.25                    # spectral cutoff, confidence tail depth
EPS_C, EPS_Z, EPS_G, EPS_A, EPS_W = 1e-12, 1e-9, 1e-6, 1e-30, 1e-3
HORIZON, WINDOW, ALPHA = 96, 8, 0.30        # J, B, alpha of Eq. (6)
WORD = re.compile(r"[a-z]+")


# ------------------------------------------------------------------------------- backbones
def _idx(mask, off):                         # Eq. (7): position counted back from the end
    return (mask.sum(1) - off).clamp(min=0)


def _seq_enc(tok, enc, maxlen, bs):
    @torch.no_grad()
    def f(prompts, off=2):
        out = []
        for s in range(0, len(prompts), bs):
            b = prompts[s:s + bs]
            t = tok(b, padding="max_length", max_length=maxlen, truncation=True,
                    return_tensors="pt")
            o = enc(t.input_ids.to(DEV))[0].float()
            out.append(o[torch.arange(len(b), device=DEV),
                         _idx(t.attention_mask.to(DEV), off)].cpu())
            del o
        return torch.cat(out)
    return f


def _pooled_enc(tok, enc, bs=256):
    @torch.no_grad()
    def f(prompts, off=2):                   # pooled groups are unaffected by the offset
        out = []
        for s in range(0, len(prompts), bs):
            t = tok(prompts[s:s + bs], padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt")
            out.append(enc(t.input_ids.to(DEV)).pooler_output.float().cpu())
        return torch.cat(out)
    return f


def load_backbone(arch, model_id):
    """-> (denoiser state dict, {input width: encoder}, pipe).

    The only architecture-specific code. Groups are named by the width they receive, so this map
    is what connects a discovered group to its representation.
    """
    if arch == "sd":
        from diffusers import StableDiffusionPipeline as P
        p = P.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None,
                              requires_safety_checker=False).to(DEV)
        return p.unet.state_dict(), {p.text_encoder.config.hidden_size:
                                     _seq_enc(p.tokenizer, p.text_encoder, 77, 512)}, p
    if arch == "t2v":                        # text-to-video UNet
        from diffusers import DiffusionPipeline as P
        p = P.from_pretrained(model_id, torch_dtype=torch.float32).to(DEV)
        return p.unet.state_dict(), {p.text_encoder.config.hidden_size:
                                     _seq_enc(p.tokenizer, p.text_encoder,
                                              p.tokenizer.model_max_length, 512)}, p
    if arch == "flux":
        from diffusers import FluxPipeline as P
        p = P.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(DEV)
        return p.transformer.state_dict(), {
            p.text_encoder_2.config.d_model: _seq_enc(p.tokenizer_2, p.text_encoder_2, 512, 8),
            p.text_encoder.config.hidden_size: _pooled_enc(p.tokenizer, p.text_encoder)}, p
    if arch == "sd3":
        from diffusers import StableDiffusion3Pipeline as P
        p = P.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(DEV)

        @torch.no_grad()
        def cat_pooled(prompts, off=2, bs=256):
            # SD3 feeds its text embedder the CONCATENATION of the two CLIP projections, a width
            # belonging to no single encoder. Registered as one callable of the combined width.
            out = []
            for s in range(0, len(prompts), bs):
                b, parts = prompts[s:s + bs], []
                for tk, te in ((p.tokenizer, p.text_encoder), (p.tokenizer_2, p.text_encoder_2)):
                    t = tk(b, padding="max_length", max_length=77, truncation=True,
                           return_tensors="pt")
                    parts.append(te(t.input_ids.to(DEV))[0].float())
                out.append(torch.cat(parts, -1).cpu())
            return torch.cat(out)

        w = p.text_encoder.config.projection_dim + p.text_encoder_2.config.projection_dim
        return p.transformer.state_dict(), {
            p.text_encoder_3.config.d_model: _seq_enc(p.tokenizer_3, p.text_encoder_3, 256, 16),
            w: cat_pooled}, p
    if arch in ("cogvideox", "wan"):         # transformer video backbones, both T5-conditioned
        from diffusers import CogVideoXPipeline, WanPipeline
        P = CogVideoXPipeline if arch == "cogvideox" else WanPipeline
        n = 226 if arch == "cogvideox" else 512
        p = P.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(DEV)
        return p.transformer.state_dict(), {p.text_encoder.config.d_model:
                                            _seq_enc(p.tokenizer, p.text_encoder, n, 8)}, p
    raise ValueError(arch)


# ------------------------------------------------------------------- footprint, Section 3.2
def materialise(edit, base):
    """Pruning methods publish a binary mask; turn it back into weights before differencing."""
    out = {}
    for k, v in edit.items():
        if k.endswith(".mask"):
            wk = k[:-5] + ".weight"
            bk = wk if wk in base else wk.replace("unet.", "")
            if bk in base:
                out[wk] = base[bk].float().to(v.device) * (1.0 - v.float())
        else:
            out[k] = v
    return out


def discover(edit, base):
    """Every matrix the erasure changed. The rule never looks at the method name."""
    out = []
    for k, v in edit.items():
        bk = k if k in base else k.replace("unet.", "")
        if bk not in base or v.shape != base[bk].shape or v.ndim != 2:
            continue
        d = v.float() - base[bk].float()
        if float(d.norm()) > TAU * float(base[bk].float().norm() + 1e-12):
            out.append({"key": k, "bkey": bk, "dnorm": float(d.norm()),
                        "in": v.shape[1], "out": v.shape[0]})
    return sorted(out, key=lambda s: -s["dnorm"])


def shape_spec(lam, rho):
    """Bracket of Eq. (1). Dropping the noise floor FIRST is not optional: raising every
    eigenvalue to rho=0 turns near-zero ones into 1, so Q I Q^T = I and the operator forgets
    the edit entirely."""
    lr = lam / lam.max().clamp(min=EPS_A)
    return torch.where(lr >= EPS0, lr.clamp(min=1e-12).pow(rho), torch.zeros_like(lr))


def eig(sd, base, imprints):
    """Eigendecompose each imprint's Gram once per checkpoint; every (beta, rho) reuses it."""
    c = {}
    for s in imprints:
        D = (sd[s["key"]].float() - base[s["bkey"]].float()).to(DEV)
        lam, Q = torch.linalg.eigh((D.T @ D).double())
        c[s["key"]] = (lam.flip(0).clamp(min=0).float(), Q.flip(1).float(),
                       float(D.norm().clamp(min=1e-12)))
        del D
    return c


def fuse_group(spec, imprints, dim, beta, rho):
    """Eq. (1), trace-normalised to supply the tr(G_g) denominator of Eq. (3).

    Only imprints that actually receive this representation are summed: an attention module may
    also hold an edited output projection, whose input is the inner dimension and not the hidden
    state the group was captured at.
    """
    G = 0
    for s in imprints:
        if s["in"] != dim:
            continue
        lam, Q, fro = spec[s["key"]]
        G = G + (fro ** (-2.0 * beta)) * (Q @ torch.diag(shape_spec(lam, rho)) @ Q.T)
    if isinstance(G, int):
        raise ValueError(f"no imprint in this group receives a width-{dim} representation")
    return G / torch.diagonal(G).sum().clamp(min=1e-12)


# -------------------------------------------------------------------- encoding, Section 3.3
def whiten(X, gamma):
    """Eq. (2). Uncentered second moment, then unit norm. gamma=0 normalises only."""
    X = X.to(DEV)
    if gamma == 0:
        Z = X.clone()
    else:
        S = (X.T @ X) / len(X)
        S = S + EPS_W * torch.trace(S) / S.shape[0] * torch.eye(S.shape[0], device=DEV)
        ev, V = torch.linalg.eigh(S.double())
        Z = (X.double() @ V @ torch.diag(ev.clamp(min=1e-12).pow(-gamma / 2)) @ V.T).float()
    return (Z / Z.norm(dim=1, keepdim=True).clamp(min=1e-9)).contiguous()


def encode(fn, pool, off, templates):
    """Mean over templates: E[c,t] = mu + a_c + b_t + eps, so averaging t isolates a_c."""
    acc = None
    for tp in templates:
        e = fn([tp.format(w) for w in pool], off)
        acc = e if acc is None else acc + e
    return acc / len(templates)


# ------------------------------------------------------------------- disclosure, Section 3.4
def conf(s, kappa=KAPPA):
    """Eq. (4). Quantile-based because score fields are right-skewed; constant field -> 0."""
    s = s.float()
    m = s.median()
    return float((s.max() - m) / (torch.quantile(s, 1 - kappa) - m + EPS_C))


def lex_index(pool):
    """F(c): token subset/superset families. Pool-only -- no labels, no free parameter."""
    toks = [frozenset(WORD.findall(p.lower())) for p in pool]
    by, post = defaultdict(list), defaultdict(set)
    for i, T in enumerate(toks):
        by[T].append(i)
        for t in T:
            post[t].add(i)

    def family(i):
        T = toks[i]
        if not T or len(T) > 6:
            return []
        out, tl = set(), list(T)
        for r in range(1, len(tl) + 1):
            for c in combinations(tl, r):
                out.update(by.get(frozenset(c), ()))
        out.update(set.intersection(*[post[t] for t in tl]))
        out.discard(i)
        return sorted(out)
    return family


def pursue(items, w, horizon, family, kappa=KAPPA):
    """Eq. (select): one shared greedy over groups that each keep their own space and basis.
    Returns the pick order and the confidence profile a_0..a_J over eligible candidates."""
    bases = [torch.zeros(G.shape[0], 0, device=DEV) for G, _ in items]
    dead = torch.zeros(items[0][1].shape[0], dtype=torch.bool, device=DEV)
    picks, prof = [], []
    for _ in range(horizon + 1):
        mixed = 0
        for j, (G, X) in enumerate(items):
            Q = bases[j]
            r = X - (X @ Q) @ Q.T if Q.shape[1] else X
            v = ((r @ G) * r).sum(1).clamp(min=0) / (r * r).sum(1).clamp(min=1e-12)
            mixed = mixed + w[j] * (v - v.mean()) / (v.std() + EPS_Z)
        mixed = mixed.masked_fill(dead, -1e30)
        live = mixed[mixed > -1e29]
        if live.numel() < 8:
            break
        prof.append(max(conf(live, kappa), 0.0))
        i = int(mixed.argmax())
        picks.append(i)
        dead[i] = True
        if family:
            f = family(i)
            if f:
                dead[torch.tensor(f, device=DEV, dtype=torch.long)] = True
        for j, (_, X) in enumerate(items):
            q = X[i:i + 1]
            q = q - (q @ bases[j]) @ bases[j].T if bases[j].shape[1] else q
            bases[j] = torch.cat([bases[j], (q / (q.norm() + 1e-12)).T], 1)
    return picks, torch.tensor(prof)


def detect_size(prof, horizon=HORIZON, largest_drop_only=False):
    """Eq. (6): r_i = log a_{i-1} - log a_i, largest decline. Some profiles RISE first, because
    selecting dominant candidates makes the remaining targets more distinguishable; there the
    largest decline can land past the erased set, so take the first sufficiently large one."""
    v = prof.float().clamp(min=0)
    m = min(horizon, v.numel() - 1)
    if m < 1:
        return 1
    r = torch.log(v[:m].clamp(min=EPS_A)) - torch.log(v[1:m + 1].clamp(min=EPS_A))
    if not largest_drop_only:
        k = min(WINDOW, m)
        if k >= 3 and int((v[1:k + 1] > v[:k]).sum()) >= k - 1 and float(r.max()) > 0:
            thr = ALPHA * float(r.max())
            for i in range(m):
                if float(r[i]) >= thr:
                    return i + 1
            return m
    return int(torch.argmax(r).item()) + 1


# -------------------------------------- text-agnostic groups: read the internal activations
def capture(pipe, pool, mods, templates, sigmas, batch):
    """For erasures that never touch a text-conditioned matrix. Hooks to_k (processors are
    called with keyword args only, so a hook on the attention sees an empty positional tuple);
    to_k and to_v of one block share an input, other blocks do not. Single forwards, no decode."""
    tr, buf, hooks = pipe.transformer, {}, []

    def get(root, dotted):
        for p in dotted.split("."):
            root = root[int(p)] if p.isdigit() else getattr(root, p)
        return root

    for g, path in mods.items():
        hooks.append(get(tr, path).register_forward_hook(
            lambda m, inp, out, g=g: buf.__setitem__(g, inp[0].detach().float().mean(1))))
    acc = {g: [] for g in mods}
    try:
        for tp in templates:
            for sg in sigmas:
                part = {g: [] for g in mods}
                for s in range(0, len(pool), batch):
                    b = pool[s:s + batch]
                    pe, pooled, tids = pipe.encode_prompt(
                        prompt=[tp.format(w) for w in b], prompt_2=None,
                        max_sequence_length=256, device=DEV)
                    gen = torch.Generator("cpu").manual_seed(0)
                    kw = dict(hidden_states=torch.randn(len(b), 1024, 64, generator=gen)
                              .to(DEV, tr.dtype) * sg,
                              encoder_hidden_states=pe.to(tr.dtype),
                              pooled_projections=pooled.to(tr.dtype),
                              img_ids=torch.zeros(1024, 3, device=DEV, dtype=tr.dtype),
                              txt_ids=tids.to(DEV).to(tr.dtype),
                              timestep=torch.full((len(b),), float(sg), device=DEV,
                                                  dtype=tr.dtype), return_dict=False)
                    # guidance-distilled variants take (timestep, guidance, pooled); undistilled
                    # ones have no guidance argument -- the kwarg must be conditional
                    if getattr(tr.config, "guidance_embeds", False):
                        kw["guidance"] = torch.full((len(b),), 3.5, device=DEV, dtype=tr.dtype)
                    tr(**kw)
                    for g in mods:
                        part[g].append(buf[g].cpu())
                for g in mods:
                    acc[g].append(torch.cat(part[g]))
    finally:
        for h in hooks:
            h.remove()
    return {g: torch.stack(v).mean(0) for g, v in acc.items()}


# ------------------------------------------------------- Appendix A.2: multiword candidates
def phrase_search(pipe, G, beam, lam, max_words, gamma):
    """Build the phrase from single words instead of looking it up. The length is an OUTPUT:
    a longer phrase wins only if it beats the shorter one by more than the penalty it pays."""
    te, tok = pipe.text_encoder.eval().to(DEV), pipe.tokenizer
    bos, eos = tok.bos_token_id, tok.eos_token_id
    inv = {v: k for k, v in tok.get_vocab().items()}
    words = sorted((i, w[:-4]) for i, w in inv.items()
                   if w.endswith("</w>") and w[:-4].isalpha() and len(w[:-4]) >= 2)
    wid, id2w = [i for i, _ in words], {i: w for i, w in words}

    @torch.no_grad()
    def rep(seqs):
        n, L = len(seqs), len(seqs[0])
        t = torch.full((n, L + 2), eos, dtype=torch.long)
        t[:, 0] = bos
        t[:, 1:L + 1] = torch.as_tensor(seqs, dtype=torch.long)
        out = []
        for s in range(0, n, 8192):          # causal encoder: read the last content token
            out.append(te(t[s:s + 8192].to(DEV))[0][:, L].float())
        E = torch.cat(out)
        Xc = E - E.mean(0, keepdim=True)     # centred here, unlike the main pipeline
        if gamma:
            S = (Xc.T @ Xc) / max(1, len(Xc) - 1)
            S = S + 1e-6 * float(torch.diagonal(S).mean()) * torch.eye(S.shape[0], device=DEV)
            ev, V = torch.linalg.eigh(S.double())
            ev = (0.95 * ev.clamp(min=0) + 0.05 * ev.clamp(min=0).mean()).clamp(min=1e-12)
            Xc = (Xc.double() @ V @ torch.diag(ev.pow(-gamma / 2)) @ V.T).float()
        return Xc / (Xc.norm(dim=1, keepdim=True) + 1e-9)

    cand, X, best = [(i,) for i in wid], None, None
    X = rep(cand)
    for L in range(1, max_words + 1):
        f = ((X @ G) * X).sum(1)
        u = (f - f.mean()) / (f.std() + EPS_Z) - lam * (L - 1)        # Eq. (10)
        top = torch.topk(u, min(beam, u.numel())).indices.tolist()
        kept = [(float(u[j]), cand[j]) for j in top]
        if best is None or kept[0][0] > best[0]:                      # Eq. (11)
            best = kept[0]
        else:
            break
        if L == max_words:
            break
        nxt = [s[:p] + (t,) + s[p:] for _, s in kept for p in range(len(s) + 1) for t in wid]
        cand = list(dict.fromkeys(nxt))
        X = rep(cand)
        torch.cuda.empty_cache()
    return " ".join(id2w.get(t, "?") for t in best[1])


# ----------------------------------------------------------------------------------- driver
def read_groups(path):
    t = {}
    for line in list(open(path))[1:]:
        p = line.strip().split(",")
        if len(p) >= 3:
            t[p[0]] = [c.strip() for c in p[2].split(";") if c.strip()]
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="sd",
                    choices=["sd", "sd3", "flux", "t2v", "cogvideox", "wan"])
    ap.add_argument("--model_id", required=True, help="the base the checkpoints were built on")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--vocab", required=True, help="JSON list of candidate strings")
    ap.add_argument("--groups_csv", default="", help="optional ground truth, for scoring only")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--templates", default="object", choices=list(TEMPLATES))
    # self-calibration grid; the default is the paper's 24 settings
    ap.add_argument("--beta", type=float, nargs="*", default=[0.0, 1.0])
    ap.add_argument("--gamma", type=float, nargs="*", default=[0.0, 1.0])
    ap.add_argument("--rho", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    ap.add_argument("--offset", type=int, nargs="*", default=[2, 1])
    ap.add_argument("--horizon", type=int, default=HORIZON, help="bounds the size reading only")
    ap.add_argument("--internal", action="store_true",
                    help="erasure touches no text-conditioned matrix: group per attention "
                         "module and read internal activations instead")
    ap.add_argument("--sigmas", type=float, nargs="*", default=[1.0], help="--internal only")
    ap.add_argument("--batch", type=int, default=16, help="--internal only")
    ap.add_argument("--phrase", action="store_true",
                    help="Appendix A.2: build multiword candidates instead of ranking the vocab")
    ap.add_argument("--no_lexical", action="store_true")
    ap.add_argument("--largest_drop_only", action="store_true")
    a = ap.parse_args()

    tpl = TEMPLATES[a.templates]
    net_sd, encoders, pipe = load_backbone(a.arch, a.model_id)
    base = {k: v.detach().float().cpu() for k, v in net_sd.items()}
    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "*.safetensors")))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {a.ckpt_dir}")
    pool = json.load(open(a.vocab))
    truth = read_groups(a.groups_csv) if a.groups_csv else {}
    pidx = {w.lower(): i for i, w in enumerate(pool)}

    # ---- discover the groups from the checkpoint, rather than declaring them
    sd0 = materialise(load_file(ckpts[0], device="cpu"), base)
    imprints = discover(sd0, base)
    del sd0
    groups = defaultdict(list)
    for s in imprints:
        if a.internal:
            if s["key"].endswith(".weight") and ".attn." in s["key"]:
                groups[s["key"].split(".attn.")[0] + ".attn"].append(s)
        elif s["in"] in encoders:
            groups[s["in"]].append(s)
    groups = dict(groups)
    if not groups:
        raise SystemExit("no edited matrix receives a text encoding -- rerun with --internal")
    print(f"[{a.arch}] {len(ckpts)} ckpts | {len(imprints)} imprints | {len(groups)} group(s)",
          flush=True)

    # ---- one-time candidate encoding, amortised over every checkpoint
    t0 = time.time()
    offsets = [0] if a.internal else a.offset
    if a.internal:
        raw = capture(pipe, pool, {g: g + ".to_k" for g in groups}, tpl[:2], a.sigmas, a.batch)
        X = {(g, ga, 0): whiten(raw[g], ga) for g in groups for ga in a.gamma}
        del raw
    else:
        X = {}
        for g in groups:
            for off in offsets:
                E = encode(encoders[g], pool, off, tpl)
                for ga in a.gamma:
                    X[(g, ga, off)] = whiten(E, ga)
                del E
    t_enc = time.time() - t0
    torch.cuda.empty_cache()
    print(f"[{a.arch}] encoded {len(pool)} candidates in {t_enc:.1f}s", flush=True)

    family = None if (a.no_lexical or a.phrase) else lex_index(pool)
    theta = [(b, g, r, o) for b in a.beta for g in a.gamma for r in a.rho for o in offsets]
    rows = []

    for ck in ckpts:
        gid = next((g for g in truth if g in os.path.basename(ck)), os.path.basename(ck)[:-12])
        t0 = time.time()
        sd = materialise(load_file(ck, device="cpu"), base)
        spec = eig(sd, base, imprints)
        del sd

        # ---- self-calibrate on confidence alone: no labels, no validation set
        best = (-1.0, None, None)
        for (be, ga, rh, of) in theta:
            items, fields = [], []
            for g, members in groups.items():
                x = X[(g, ga, of)]
                G = fuse_group(spec, members, x.shape[1], be, rh)
                items.append((G, x))
                fields.append(((x @ G) * x).sum(1).clamp(min=0))
            w = torch.tensor([max(conf(f), EPS_G) for f in fields])
            w = (w / w.sum()).tolist()
            c = conf(sum(wi * (f - f.mean()) / (f.std() + EPS_Z)
                         for wi, f in zip(w, fields)))                 # Eq. (5)
            if c > best[0]:
                best = (c, items, w)
            del fields
        bconf, items, w = best

        if a.phrase:
            pred = [phrase_search(pipe, items[0][0], 8, 0.05, 3, 1.0)]
            n = 1
        else:
            picks, prof = pursue(items, w, a.horizon, family)
            n = detect_size(prof, a.horizon, a.largest_drop_only)
            pred = [pool[i] for i in picks[:n]]

        row = {"ckpt": os.path.basename(ck), "n_hat": n, "predicted": pred,
               "conf": bconf, "seconds": time.time() - t0}
        if gid in truth:
            tset = {pidx[c.lower()] for c in truth[gid] if c.lower() in pidx}
            K = len(truth[gid])
            got = {pidx[c.lower()] for c in pred if c.lower() in pidx}
            row |= {"truth": truth[gid], "K": K,
                    "exact_set": int(got == tset), "exact_size": int(n == K),
                    "recall_at_K": (len(tset & set(picks[:K])) / K
                                    if not a.phrase else float(got == tset))}
            print(f"  {gid:14s} n^={n:<3d} (true {K})  exact={row['exact_set']}  "
                  f"{row['seconds']:.2f}s  {pred[:6]}", flush=True)
        else:
            print(f"  {gid:14s} n^={n:<3d}  {row['seconds']:.2f}s  {pred[:6]}", flush=True)
        rows.append(row)
        del spec, items
        torch.cuda.empty_cache()

    print(f"\n{len(rows)} checkpoints | encoding {t_enc:.1f}s once | "
          f"{sum(r['seconds'] for r in rows) / len(rows):.2f}s each")
    if truth and "exact_set" in rows[0]:
        m = lambda k: 100 * sum(r[k] for r in rows) / len(rows)        # noqa: E731
        # exact size must be read BESIDE recall: recall does not penalise over-estimation, so an
        # estimator returning a huge set scores well on recall while being useless
        print(f"  identification accuracy  {m('exact_set'):.1f}%\n"
              f"  exact size accuracy      {m('exact_size'):.1f}%\n"
              f"  recall@K                 {m('recall_at_K') / 100:.3f}")
    if a.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
        json.dump({"arch": a.arch, "model_id": a.model_id, "ckpt_dir": a.ckpt_dir,
                   "n_pool": len(pool), "encode_s": t_enc, "rows": rows},
                  open(a.out_json, "w"), indent=1)
        print(f"wrote {a.out_json}")


if __name__ == "__main__":
    main()
