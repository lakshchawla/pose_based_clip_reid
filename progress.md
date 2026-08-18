# PCR2 Progress Log

This file is the source of truth for what has been built, why, and what's left. Code stays
comment-light on purpose — major implementation steps, decisions, and their rationale are logged
here instead. Append new dated entries at the bottom of **Progress Log** as work continues; don't
rewrite history, add to it.

---

## Plans

### Plan A (original handoff, `PLAN.md`) — staged Milestone 1 / Milestone 2

Written by a peer session before implementation started. Full text preserved in `PLAN.md` at the
repo root. Summary: port SpCL's baseline self-paced UDA strategy in two stages —

- **Milestone 1**: validate the ported SpCL strategy (HybridMemory, DSBN, trainer, plain Jaccard
  re-ranking) end-to-end using BPBReID as a single-branch (`bn_foreg` only) drop-in encoder, to
  isolate "did the port work" from "does part-based help."
- **Milestone 2**: generalize to BPBReID's full K part-embeddings + visibility scores — new
  `PartHybridMemory`, `compute_bpb_pairwise_distance`, part-based Jaccard base distance,
  part-aware evaluator. M1 was meant to stay intact as a fallback.
- Training hyperparameters were left implicitly config-driven (SpCL-argparse style, not specified
  as a hard requirement either way).
- Explicitly out of scope: CA-Jaccard / camera-aware memory, part-based triplet loss variants, EMA
  teacher-student.

### Plan B (superseding direction, given directly by the user mid-conversation) — what was actually built

The user overrode Plan A's staging before any code was written. Effective spec, in full:

1. **Merge M1+M2 — build the full part-based version directly**, skip the single-branch
   `bn_foreg`-only milestone. BPBReID's K learnable part embeddings + visibility scores are used
   from the start.
2. **Keep `PartHybridMemory`** (an earlier draft instruction to drop the memory bank and replace it
   with a classifier-head + pseudo-label loss was explicitly retracted by the user one message
   later — see conversation history, not re-litigated here).
3. **Explicit BPBReID-style similarity/matching function**: `compute_bpb_pairwise_distance` in
   `pcr/utils/part_distance.py`, ported from `torchreid/metrics/distance.py`'s
   `compute_distance_matrix_using_bp_features`. This one function is the single source of truth
   for part-based distance, reused by both Jaccard re-ranking and the evaluator.
4. **No separate config file for training hyperparameters.** Every hyperparameter used in
   `examples/train_uda.py` is an `argparse` flag with its default set directly in that file — no
   YAML/config-loader layer. (The one exception, `BPBReIDModelCfg`, is a plain dataclass describing
   frozen *model architecture* to satisfy `BPBreID.__init__`, not a training-hyperparameter config.)
5. **Learnable part attention, no masks anywhere in pcr2.** `learnable_attention_enabled=True`
   means the pixel-to-part classifier is learned end-to-end from the contrastive gradient alone;
   PifPaf masks are never loaded or used in this repo (they're only relevant to the *external*
   source-pretraining step, which is out of scope here and stays in the bpbreid repo). This was
   explicitly called out by the user because PifPaf masks only exist for the source domain, not the
   target — learnable attention sidesteps the need for them on either side.
6. **Weights are saved** via a ported `save_checkpoint` (best + latest).
7. **Use a coding subagent** for the actual file-writing, after the orchestrating session did its
   own grounding (reading the real SpCL/BPBReID source to nail exact signatures) and wrote a
   detailed, file-by-file, source-referenced spec — then verified the subagent's output against
   that spec by reading the actual files, not by trusting its self-report.
8. **Minimal comments in code.**

Deferred / explicitly out of scope for this build (carried over from Plan A, still holds):
- CA-Jaccard / camera-aware or camera-proxy memory — traditional Jaccard re-ranking only.
- Part-based triplet loss variants / BodyPartAttentionLoss pixel-supervision — the part-based
  memory-bank contrastive (InfoNCE-style) loss is the sole training signal, matching baseline
  SpCL's own reliance on `HybridMemory` alone with no auxiliary triplet term.
- EMA teacher-student — the memory bank itself is the running feature cache, no separate teacher
  network.
- Source-domain pretraining code — stays external, reuses bpbreid's own
  `torchreid/scripts/main.py` + its existing yaml configs unmodified.

---

## Progress Log

### 2026-08-13 — Full repo built (Plan B, merged M1+M2), verified

**Repo tree** (`/home/lakshh/workspace/reid/pcr2`):

```
pcr2/
  setup.py, README.md, PLAN.md, progress.md
  examples/train_uda.py
  pcr/
    __init__.py, trainers.py, evaluators.py
    models/{__init__,bpbreid_encoder,dsbn,hm}.py
    utils/{__init__,meters,logging,serialization,osutils,jaccard_rerank,part_distance}.py
    utils/data/{__init__,base_dataset,preprocessor,sampler,transforms}.py
    datasets/{__init__,market1501,dukemtmc}.py
    evaluation_metrics/{__init__,ranking}.py
```

**What each non-trivial file does:**

- `pcr/models/bpbreid_encoder.py` — `BPBReIDModelCfg` (plain dataclass, ~13 attrs `BPBreID`
  actually reads) + `BPBReIDEncoder(nn.Module)`. Wraps `torchreid.models.bpbreid.BPBreID` with a
  placeholder `num_classes=1` (BPBReID's own ID classifier heads are never used — only embeddings +
  visibility). `forward(images) -> (Tensor[B, 1+K, D], Tensor[B, 1+K])`: stacks `bn_foreg` +
  `bn_parts` embeddings (index 0 = foreground, 1..K = parts), L2-normalizes each branch
  independently, stacks matching visibility scores. This *is* the module's `forward` (not a
  separately named method) so `nn.DataParallel` gathers it fine as a tuple-of-tensors return.
  Gotcha logged here so it isn't rediscovered: `parts_num=5` is correct for the
  market1501/dukemtmc bpbreid configs even though the yaml's own explicit `masks.parts_num` default
  is 1 — `masks.preprocess: 'five_v'` auto-resolves it to 5 via
  `torchreid/data/masks_transforms/__init__.py::compute_parts_num_and_names` at config-parse time
  in the *original* bpbreid repo. Verified by reading that file directly, not assumed.

- `pcr/utils/part_distance.py::compute_bpb_pairwise_distance(qf, qf_vis, gf=None, gf_vis=None, ...)`
  — ported from `torchreid/metrics/distance.py::compute_distance_matrix_using_bp_features` +
  `_compute_body_parts_dist_matrices`, and `masked_mean`/`replace_values` from
  `torchreid/utils/tensortools.py`. Bool visibility -> outer-product mutual-visibility mask +
  "max observed distance + 1" sentinel for zero-mutual-visibility pairs. Continuous visibility ->
  sqrt-product soft gate. `gf=None` computes self-distance (used by Jaccard re-ranking).
  Line-by-line cross-checked against the original during verification.

- `pcr/models/hm.py::PartHM` (autograd Function) + `PartHybridMemory` — SpCL's `HM`/`HybridMemory`
  generalized from `[N,D]` to `[N,M,D]` feature buffers. Per-branch similarity
  `einsum('bmd,nmd->bmn', ...)`; momentum update in `backward` skips a branch entirely (leaves the
  stored slot untouched) when that branch is invisible for the sample, rather than blending in a
  garbage embedding, then L2-renormalizes only the branches that were actually updated.
  `PartHybridMemory.forward` combines the M per-branch similarities into one `[B,N]` via a
  *query-visibility-only* weighted average (the memory bank has no per-slot visibility bookkeeping,
  so it can't do the full two-sided gating `compute_bpb_pairwise_distance` does — this is a
  deliberate, documented distinction, not an oversight) — then the masked-softmax + NLL loss over
  cluster/class slots is unchanged from SpCL's original. One real bug caught and fixed during
  verification: SpCL's original in-place `sim /= self.temp` throws under current PyTorch autograd
  when applied to a custom `Function`'s output view; changed to out-of-place `sim = sim / self.temp`.

- `pcr/utils/jaccard_rerank.py::compute_jaccard_distance(base_dist, k1, k2)` — plain (non-FAISS,
  non-camera-aware) k-reciprocal algorithm ported unchanged math-wise from
  `spcl/utils/faiss_rerank.py`; the one structural change is that it now takes a **precomputed**
  `[N,N]` base distance matrix directly (built by the caller via `compute_bpb_pairwise_distance`)
  instead of computing `2-2*f@f.T` internally via FAISS GPU search — this is what lets the same
  function serve both a flat-vector and a part-based base distance without touching this file.

- `pcr/trainers.py::PCRTrainer_UDA`, `pcr/evaluators.py`, `pcr/models/dsbn.py` — straightforward
  generalizations/verbatim ports; `DSBN2d/DSBN1d/convert_dsbn/convert_bn` needed no changes at all
  (fully generic over any BatchNorm1d/2d in the model tree). Trainer's device-reshape/split trick
  now applies to `(f_out, vis)` tuples instead of a flat tensor. Evaluator's `pairwise_distance` now
  defers to `compute_bpb_pairwise_distance` instead of a raw `2-2*x@y.T`.

- `examples/train_uda.py` — every hyperparameter is an inline `argparse` default (dataset choice,
  batch size, height/width — 384x128, BPBReID's own convention, not SpCL's 256x128 — DBSCAN eps/
  eps-gap, Jaccard k1/k2, memory momentum/temp, optimizer lr/weight-decay/epochs/iters/step-size,
  `--checkpoint-path` for the externally-pretrained BPBReID checkpoint). Source-center
  initialization does a per-branch, visibility-weighted mean over each pid's member embeddings
  (falls back to a plain unweighted mean for any branch with zero visible members for that pid, to
  avoid div-by-zero). Target-instance init writes extracted part-embeddings straight into the
  memory. Per-epoch: base distance -> Jaccard re-rank -> three-way DBSCAN -> R_indep/R_comp
  self-paced reliability filtering -> pseudo-labeled dataset -> `PCRTrainer_UDA.train(...)`,
  ported near-verbatim from SpCL's `main_worker`. Two resolved ambiguities, logged with rationale
  directly in that file as one-line comments: (a) the epoch-level base distance for target memory
  slots treats every slot/branch as visible (`all-ones` visibility) rather than re-extracting
  target visibility every epoch, since the memory bank has no per-slot visibility bookkeeping
  anyway and re-extracting would cost a full extra forward pass per epoch; (b) SpCL's own
  `--min-samples` argparse flag is defined but never actually wired into its `DBSCAN(...)` calls
  (hardcoded `min_samples=4` regardless) — matched that actual (not documented) behavior rather
  than the flag's apparent intent.

- Generic plumbing ported near-verbatim from SpCL (adjusted only for `spcl.` -> `pcr.` import
  paths): `meters.py`, `logging.py`, `serialization.py`, `utils/data/{__init__,preprocessor,
  sampler,transforms}.py`, `evaluation_metrics/ranking.py`, `datasets/{market1501,dukemtmc}.py`.
  Three files not in the original file-tree spec were added because the ported code above
  transitively needs them (`utils/__init__.py`'s `to_numpy`/`to_torch` for `ranking.py`,
  `utils/osutils.py`'s `mkdir_if_missing` for logging/serialization/dukemtmc,
  `utils/data/base_dataset.py`'s `BaseImageDataset` for the dataset classes) — matches SpCL's own
  module boundaries, not scope creep.

**Verification performed** (by the orchestrating session, reading the actual files — not taking the
building subagent's self-report at face value):
- Read `hm.py`, `part_distance.py`, `jaccard_rerank.py`, `bpbreid_encoder.py`, `evaluators.py`,
  `trainers.py`, `dsbn.py`, `models/__init__.py`, `datasets/__init__.py`, `dukemtmc.py`, `setup.py`,
  `README.md`, and all of `train_uda.py` end-to-end, cross-checked against the real SpCL/BPBReID
  source.
- `python -m py_compile` on every `.py` file in the repo — clean.
- `import pcr` and every submodule import cleanly in the `torchreid` conda env (has torch 2.11 +
  numpy/scipy/sklearn/PIL) — confirmed directly, not just via the subagent's claim.
- Removed one dead import (`copy_state_dict`, unused since this build has no `--resume` flag)
  flagged by the editor's static analysis after the build.

**Known environment gotcha, not a code bug** — flagged to the user, not yet resolved: the existing
`torchreid` conda env has **deep-person-reid's** unrelated `torchreid` package on the path, not
bpbreid's. `from torchreid.models.bpbreid import BPBreID` fails there with a clean
`ModuleNotFoundError` (verified directly) rather than silently importing the wrong package — but
actually running `train_uda.py` needs a env with `pip install -e /home/lakshh/workspace/reid/bpbreid`
run into it, likely a fresh env since both packages claim the `torchreid` name.

**Not yet done / open follow-ups:**
- No actual training run performed (no GPU/dataset exercised in this environment) — only static
  verification (compile, import, and a synthetic-tensor shape/invariant check the build subagent
  ran and reported, not independently re-run by the orchestrating session).
- Source BPBReID checkpoint pretraining (external, via bpbreid's own `main.py`) has not been run.
- `--resume`-from-checkpoint support was dropped from `train_uda.py` relative to SpCL's original
  (no such flag exists yet) — add if resuming interrupted UDA runs turns out to be needed.
- CA-Jaccard / camera-aware clustering remains explicitly out of scope per the user's original
  scope note in Plan A, "not a hard rule ... may be revisited later."

### 2026-08-18 — Training-readiness sanity check (logic/design review, not a code-correctness pass)

Reviewed whether the BPBReID-in-SpCL integration is *logically* sound enough to attempt training,
beyond the earlier "does it compile/import" check. Traced actual data/gradient flow rather than
re-reading files in isolation.

**Confirmed sound:**
- HRNet backbone (`torchreid/models/hrnet.py:611`) only dereferences `hrnet_pretrained_path` when
  `pretrained=True`; `BPBReIDEncoder` always builds with `pretrained=False`, so the empty default
  path never causes a crash.
- `load_pretrained_weights`'s shape-mismatch-skip behavior is exactly why the placeholder
  `num_classes=1` works: the external checkpoint's classifier heads (sized for real num_classes)
  get silently skipped on load, everything else (backbone, pixel-classifier, pooling, dim-reduce)
  loads correctly.
- DSBN's `[source-half, target-half]` batch-split assumption holds through BPBReID's extra
  branches exactly as it does in vanilla SpCL — established once before pooling, unaffected by the
  added classifier heads or the M-dimension carried through the trainer's reshape.
- Gradient path into the attention module (`pixel_classifier -> softmax -> parts_masks -> weighted
  pooling -> bn_parts -> memory loss`) is fully differentiable; visibility is used only as a
  multiplicative weight (stop-gradient when binary), matching GiLt's own convention, not a bug.
- DataParallel + the custom `PartHM` autograd Function don't conflict — same pattern as vanilla
  SpCL's `HM`, memory bank lives outside the DataParallel-wrapped encoder.

**Real risk flagged (methodological, not a bug):** dropping GiLt's ID+triplet loss and all
pixel-supervision means the attention module's *only* training signal is whatever leaks back
through the memory-bank contrastive loss. De-risked by the fact the source checkpoint already has
mask-supervised attention (UDA phase fine-tunes, doesn't learn attention from scratch) — but there's
no anchor loss keeping attention well-behaved if pseudo-labels get noisy mid-training. Recommend
logging per-epoch mean part-visibility rate during the first real runs to watch for attention
collapse.

**Minor, non-blocking inefficiency:** unused identity-classifier heads (global/background/
foreground/concat/parts) still run forward every step and sit in the optimizer's param groups with
permanently-zero gradient — wasted compute/memory, not incorrect. Not fixed yet.

**Still unverified:** no actual forward/backward pass has been run (no GPU/dataset in this
environment) — only static and synthetic-tensor checks so far. Recommended next step:
`--setup-only` dry run, then a short smoke run (few iters) watching for NaN loss and sane
visibility stats, before committing to a full run.
