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

### 2026-08-18 — Dummy training run: environment setup + two real bugs found and fixed

Ran an actual `--setup-only` dry run and then a short smoke training run (`-ds market1501 -dt
dukemtmc-reid`, low batch size for the local 8GB RTX 4060) against real data
(`/home/lakshh/workspace/reid/datasets/{market1501,dukemtmc-reid}`, both present with the expected
`bounding_box_train/query/bounding_box_test` layout and correct image counts). No source-pretrained
BPBReID checkpoint exists yet (external pretraining hasn't been run), so this smoke run used
`--checkpoint-path ""` (randomly-initialized backbone) purely to exercise the pcr2 pipeline's
mechanics — not to produce a meaningful model.

**Environment**: no conda env had bpbreid's own `torchreid` installed — the existing `torchreid`
env's `torchreid` package resolved to an unrelated `deep-person-reid` editable install. Cloned that
env to a new `pcr2-run` env (non-destructive — left the original `torchreid` env untouched), swapped
in bpbreid's `torchreid` (`pip install -e /home/lakshh/workspace/reid/bpbreid`), then
`pip install -e /home/lakshh/workspace/reid/pcr2`. Confirmed `torchreid.models.bpbreid` imports
correctly from the bpbreid path afterward.

**Bug 1 — `convert_dsbn`/`convert_bn` crash on parameter-free submodules (`pcr/models/dsbn.py`).**
`assert not next(model.parameters()).is_cuda` raises `StopIteration` (uncaught) whenever recursion
reaches a submodule whose entire subtree has zero parameters. Never surfaced in vanilla SpCL (every
intermediate container in a plain ResNet holds a Conv/BN somewhere), but BPBReID's pooling-head
wrappers (`GlobalAveragePoolingHead`/`GlobalWeightedAveragePoolingHead`, used for the foreground/
background/parts attention pooling heads) have exactly one child, `self.normalization = nn.Identity()`
when `normalization='identity'` (this repo's config) — a genuinely empty parameter subtree. Fixed by
using `next(model.parameters(), None)` and treating "no params" as "not cuda" (nothing to guard
against). Caught immediately by the `--setup-only` dry run, first failure point.

**Bug 2 — `compute_bpb_pairwise_distance` OOMs at real UDA scale (`pcr/utils/part_distance.py`).**
The build agent's implementation intentionally dropped bpbreid's own gallery-batching
(`batch_size_pairwise_dist_matrix`), reasoning the repo "always runs on modest re-ID gallery sizes."
That assumption is false for this use case: the target-domain self-distance for Jaccard re-ranking
runs over the *entire* target training set every epoch (16,522 images for DukeMTMC), not a query/
gallery eval set. Reproduced directly with a synthetic `[16522, 6, 512]` self-distance: the original
unbatched implementation tried to allocate 6.1GB for one intermediate tensor and OOM'd outright on
this 8GB card (confirmed via `torch.cuda` before touching the training script at all). Rewrote with
chunked processing over the gallery dimension (default `batch_size=1024`), a running-max accumulator
so the "sentinel = max observed distance + 1" value for zero-mutual-visibility pairs stays a true
global max rather than a per-chunk one, and in-place writes into a pre-allocated `[Nq, Ng]` output
tensor instead of accumulating a Python list of chunks before `torch.cat` (that intermediate list-
then-cat approach alone re-inflated peak memory to ~6GB — tightened to write-in-place +
`masked_fill_` instead, bringing peak down to ~3.4GB for the same 16522x16522 case). Verified
numerically sane (no NaN, correct shape, plausible distance range) before re-running the full
pipeline.

Both fixes were caught by actually running the code against real data/hardware, not by re-reading
files — this is exactly the gap the earlier static-only verification pass (compile + import +
synthetic-shape checks) couldn't close.

**Bug 3 — training-step OOM at `-b 8` (not caught by the earlier fixes, only surfaces on an actual
backward pass).** First real training attempt (`-b 8 -j 4 --num-instances 4`) got all the way
through dataset load, source/target feature extraction, Jaccard distance (13s, no OOM — bug 2's fix
holds), and DBSCAN (206-215 clusters found), then OOM'd inside `trainer.train`'s `loss.backward()`
on the very first iteration: `Tried to allocate 1.76 GiB ... this process has 5.89 GiB memory in
use`. Root cause, not a bug in the code so much as a sizing miscalculation on my part: `args.batch_size`
is *per domain* (source and target each get their own full batch, per `get_train_loader` — same
convention as vanilla SpCL), so `-b 8` meant 16 images jointly through HRNet32 at 384x128 per
iteration, not 8. Neither the earlier `--setup-only` dry run nor the feature-extraction passes
would ever have caught this: both run under `torch.no_grad()`, which never allocates the
backward-pass activation memory that's actually the expensive part. **Retried at `-b 4
--num-instances 2` (joint batch of 8) — ran clean, no OOM, through both epochs plus an
end-of-training evaluation pass.**

**Dummy run outcome (`-ds market1501 -dt dukemtmc-reid`, `-b 4 -j 4 --num-instances 2 --iters 10
--epochs 2 --eval-step 2 --checkpoint-path ""`, i.e. a randomly-initialized backbone — no real
source-pretrained checkpoint exists yet, this run only validates pipeline mechanics):**

- Total wall time: 11m 5s (RTX 4060, 8GB). Most of it is the four full-dataset feature-extraction
  passes this run does (source-center init, target-instance init, one mid-training eval, one final
  eval) — each takes ~2-3 min at this batch size, dominating over the actual 10-iteration training
  steps (~3-5s each).
- Epoch 0: 206 clusters found from DBSCAN, 10,689 of 16,522 target-train images left as
  singleton/un-clustered "outliers" (expected — the encoder is untrained, so pseudo-labels are
  effectively derived from random-feature noise, not real identity structure).
  ```
  Epoch: [0][2/10]  Loss_s 9.344 (9.426)   Loss_t 9.905 (9.660)
  Epoch: [0][4/10]  Loss_s 10.400 (9.717)  Loss_t 10.672 (9.891)
  Epoch: [0][6/10]  Loss_s 10.385 (9.895)  Loss_t 11.207 (10.131)
  Epoch: [0][8/10]  Loss_s 11.658 (10.320) Loss_t 11.066 (10.353)
  Epoch: [0][10/10] Loss_s 12.250 (10.632) Loss_t 11.941 (10.644)
  ```
- Epoch 1: re-clustered (206 clusters, 10,739 outliers — stable cluster count, as expected with an
  untrained encoder producing near-static features between epochs).
  ```
  Epoch: [1][2/10]  Loss_s 11.780 (11.421) Loss_t 11.202 (11.134)
  Epoch: [1][4/10]  Loss_s 11.776 (11.221) Loss_t 9.604 (10.581)
  Epoch: [1][6/10]  Loss_s 9.061 (10.692)  Loss_t 9.661 (10.325)
  Epoch: [1][8/10]  Loss_s 10.496 (10.521) Loss_t 10.548 (10.303)
  Epoch: [1][10/10] Loss_s 10.031 (10.457) Loss_t 9.668 (10.176)
  ```
- No NaN/Inf at any point. No clear downward loss trend across these 20 total iterations — expected
  and not a red flag: with a randomly-initialized backbone and only 2x10 iterations, there's no
  reason to expect convergence; the value of this run is confirming the mechanics (data flow,
  clustering, memory update, backward, checkpointing, eval) run cleanly end-to-end, not producing a
  usable model.
- Final eval: Mean AP 0.2%, top-1 0.4%, top-5 1.3%, top-10 1.9% — consistent with near-random
  (expected, same reason as above).
- `checkpoint.pth.tar` and `model_best.pth.tar` (411MB each) written correctly to
  `logs/dummy_run/` — confirms weight-saving works end-to-end.

**Batch-size guidance for this GPU (RTX 4060, 8GB) confirmed empirically, not guessed:** `-b 8`
(joint batch 16) OOMs during backward; `-b 4` (joint batch 8) runs clean with headroom. Recommend
`-b 4` through `-b 8` as the range to try for a real run on similar 8GB hardware, starting low and
increasing only if memory allows — `-b 8` specifically is confirmed to fail at height=384/width=128
with the HRNet32 backbone on this card.

**Environment note for future runs:** the working env is `pcr2-run` (conda), not `torchreid` (which
has an unrelated `deep-person-reid` `torchreid` package installed) — `conda activate pcr2-run`
before running `examples/train_uda.py`.

### 2026-08-19 — `train_usl.py`: ICE's unsupervised (single-domain) strategy, hard-instance
### contrastive loss only

Built a second training driver replicating ICE's USL pipeline
(`/home/lakshh/workspace/reid/ICE`, `examples/unsupervised_train.py` + `ice/trainers.py::
ImageTrainer` + `ice/loss/contrastive.py::ViewContrastiveLoss`), per direct instruction: single
target dataset + pseudo-labels only (no source domain), and only ICE's InfoNCE-based hard-instance
contrastive loss — its cluster-classification loss, cross-camera loss, and teacher-student KL
consistency term are all dropped. Confirmed via the log line ICE itself prints
(`Loss_hard_instance`) that `ViewContrastiveLoss` *is* the hard-instance contrastive loss (Eq. 9,
"L_h_ins", in the ICE paper) before building anything.

**New files:**
- `pcr/loss/contrastive.py::PartViewContrastiveLoss` — ICE's `ViewContrastiveLoss` generalized to
  BPBReID's `[B, M, D]` per-branch embeddings. The hard-mining/InfoNCE math (hardest same-pseudo-ID
  positive by lowest similarity, all different-ID samples as negatives, cross-entropy over
  `[positive, negatives]`) is unchanged from ICE's original — the only new part is combining M
  per-branch `[B,B]` similarity matrices into one via visibility-gated mean (mirrors
  `part_distance.py`'s bool/continuous convention), with one deliberate deviation from that file's
  own sentinel convention: zero-mutual-visibility branch-pairs fall back to `0` similarity here, not
  `part_distance.py`'s "max distance + 1" — for a *distance*, an extreme sentinel means "very far
  apart" (correct); for a *similarity* feeding hard-*positive* mining (which picks the **lowest**
  similarity as "hardest"), an extreme low sentinel would make every zero-visibility pair look like
  the hardest positive regardless of real difficulty, so this loss uses a neutral 0 fallback instead.
  Verified with a synthetic test: correct shapes, finite loss and gradients under bool visibility,
  continuous visibility, and the fully-invisible edge case (all three checked directly, not assumed).
- `pcr/trainers_usl.py::ICEUSLTrainer` — separate trainers file (per instruction), EMA teacher-student
  update + ShuffleBN (ported from ICE's `_update_ema_variables`/`get_shuffle_ids`) driving only
  `PartViewContrastiveLoss`. No DSBN here — DSBN exists specifically for the UDA joint source+target
  batch split in `pcr/trainers.py`; single-domain USL has no domain split, so plain BatchNorm is
  correct and simpler.
- `examples/train_usl.py` — per-epoch: extract via the EMA model → `compute_bpb_pairwise_distance`
  self-distance → `compute_jaccard_distance` → single DBSCAN pass → pseudo-labeled dataset with
  **DBSCAN outliers dropped entirely** (ICE's own, simpler USL approach — no SpCL-style
  R_indep/R_comp self-paced filtering, no outlier-as-singleton-class handling) → `ICEUSLTrainer.train`.
  All hyperparameters inline as argparse defaults, matching this repo's established no-separate-config
  convention. `-dt` only, no `-ds` anywhere in the script.
- Supporting additions to existing shared files (not new files): `pcr/utils/data/preprocessor.py`
  gained `Preprocessor2View` (two independently strong-augmented views of the same image, no index —
  this pipeline keeps no per-instance memory slots, so it doesn't need one); `pcr/utils/data/
  sampler.py` gained `MoreCameraSampler` (ported from ICE, ICE's own preferred PK sampler for this
  strategy); `pcr/utils/data/transforms.py` gained `GaussianBlur` (dropped earlier as "ICE-specific,
  unused" when only `train_uda.py` existed — now genuinely needed); `pcr/utils/lr_scheduler.py`
  (new file) has `WarmupMultiStepLR`, ported near-verbatim from `ice/utils/lr_scheduler.py`.

**Verified, not just written:** full repo `py_compile` + import sweep in the `pcr2-run` env passed.
Ran an actual dummy training smoke test (`-dt market1501`, `-b 4 --num-instances 2`, 2 epochs x 10
iters, no checkpoint — random-init backbone, mechanics-only check, same discipline as the UDA
dummy run): completed in 7m58s, no crash, no NaN, `Loss_hard_instance` printed and finite every
logged iteration (range ~1.2-1.8), checkpoint saved (144MB) and end-of-training eval ran cleanly
(mAP 0.3%, expected near-random for a random-init encoder). One observation, not a bug: DBSCAN's
cluster count collapsed from 335 IDs (epoch 0) to 9 IDs (epoch 1) — a known sensitivity of
density-based clustering near its `eps` threshold when run on a genuinely unstructured (random-init)
feature space; expect far more stable epoch-to-epoch clustering once trained from a real checkpoint.
No batch-size OOM issue reproduced at `-b 4` (this pipeline's per-iteration memory profile is
similar in shape to the UDA one — same encoder/resolution, two forward passes per step instead of
one joint DSBN-split pass — so the same `-b 4`-safe / `-b 8`-risky guidance from the UDA dummy run
likely applies here too, though not separately re-confirmed at `-b 8`).

### 2026-08-19 15:58 CST — Plan: BPA loss, body-part triplet loss, horizontal-stripes option,
### hard/soft triplet loss

**Clarification asked directly: does anything currently in this repo use camera-aware or
camera-agnostic memory? No.** Neither `pcr/models/hm.py::PartHybridMemory` (used by
`train_uda.py`) nor `pcr/trainers_usl.py::ICEUSLTrainer` / `pcr/loss/contrastive.py::
PartViewContrastiveLoss` (used by `train_usl.py`) has any camera dimension, per-camera proxy
memory, or cross-camera loss term. This was a deliberate exclusion both times: PLAN.md's original
scope note ruled out CA-Jaccard/camera-aware clustering from the start, and `ICEUSLTrainer` was
built by explicitly stripping ICE's own `percam_memory`/`loss_cam` mechanism out of
`ice/trainers.py::ImageTrainer`, keeping only the hard-instance contrastive term (confirmed while
researching this plan: even in ICE's own upstream repo, `loss_cam` is computed but never actually
added to the backprop loss — `loss = loss_ccl + loss_vcl + loss_kl` — so ICE's shipped USL script
doesn't really use it either despite building the memory; the sibling `mod_ice` repo does wire a
real camera-aware memory (`HybridCameraMemory`/`CrossDomainMemory`) into its own UDA trainer, but
that's a different repo we haven't touched).

**Research grounding for this plan** (exact files/signatures, not paraphrase — full detail
available on request, summarized per-item below):
- BPA loss: `torchreid/losses/body_part_attention_loss.py::BodyPartAttentionLoss`, invoked from
  `torchreid/engine/image/part_based_engine.py::ImagePartBasedEngine.combine_losses()`.
- PifPaf masks on disk: confirmed present at
  `/home/lakshh/workspace/reid/datasets/market1501/masks/{pifpaf,pifpaf_maskrcnn_filtering}/<split>/<img>.npy`
  — market1501 only; no masks directory exists for the dukemtmc-reid copy on disk, consistent with
  "PifPaf masks only available for the source dataset."
- Body-part triplet loss: `torchreid/losses/part_averaged_triplet_loss.py` and siblings, dispatched
  via `torchreid/losses/__init__.py::init_part_based_triplet_loss(name, **kwargs)` — 7 named
  combination strategies (averaged / max / min / max_min / random_max_min / inter_parts /
  intra_parts-individual).
- Horizontal stripes: `torchreid/models/bpbreid.py::BPBreID.__init__`'s real, correctly-spelled
  `horizontal_stripes` parameter. **Confirmed bug in bpbreid's own shipped code**: its `pcb()`/
  `bot()` factory functions (bottom of the same file) pass a *typo'd* kwarg,
  `horizontal_stipes=True` (missing the second "r"), which `**kwargs` silently swallows without
  error — those factories never actually enable stripe mode. Not something to replicate; pcr2's
  `BPBReIDEncoder` already constructs `BPBreID` directly rather than through those factories, so
  this is avoided by construction as long as the correct spelling is used.
- Hard/soft triplet: `ice/loss/triplet.py::TripletLoss` (batch-hard mining on `[N,D]`+labels) and
  `SoftTripletLoss` (distillation: batch-hard-mines indices from one embedding set, then
  cross-entropy between that set's and a *second* embedding set's softmaxed ap/an distances —
  needs two forward passes of the same inputs, e.g. student vs.\ teacher). **Both are dead code in
  ICE's own shipped trainer** — never wired into `ImageTrainer`. The sibling `mod_ice` repo's
  `ice/uda_trainer.py::UDATrainer` does use hard `TripletLoss`, applied only to source real labels
  (`self.triplet_loss(f_out_s, pids_s)`); `SoftTripletLoss` is unused everywhere in both repos —
  there's no existing precedent to copy for exactly how hard+soft should pair up, see the open
  question below.

**Additional things this plan needs that weren't named explicitly, declared up front:**
1. A source-only, masks-aware dataset/preprocessor path. Every dataset/preprocessor class in pcr2
   today (`pcr/datasets/*.py`, `pcr/utils/data/preprocessor.py`) is deliberately mask-free. BPA
   loss needs source images paired with their PifPaf mask file — this is new data-loading surface,
   not just a new loss file.
2. A decision on *how* to load and group those masks. Two options, not both: (a) reuse torchreid's
   own `torchreid.utils.tools.read_masks` + `torchreid.data.masks_transforms` (five-part grouping,
   etc.) directly by import — pcr2 already hard-depends on torchreid for the encoder, so this adds
   coupling but not a new dependency, and avoids re-porting a moderately complex
   albumentations-based pipeline; or (b) port a minimal, pcr2-native mask loader/grouper. This plan
   recommends (a) for the reasons above; noted as a decision point, not silently assumed.
3. A way to get `pixels_cls_scores` (BPA loss's core input) out of the encoder at all.
   `BPBReIDEncoder.forward()` today discards it (`embeddings, visibility_scores, _, _, _, _ =
   self.model(images)`), and every existing caller (`pcr/trainers.py`, `pcr/trainers_usl.py`,
   `pcr/evaluators.py`, memory init in both `examples/*.py`) depends on `forward()` staying a clean
   `(f_out, vis)` 2-tuple under `nn.DataParallel`. This is a real architectural fork, not a detail:
   - **Option A (recommended):** extend `forward()`'s return to `(f_out, vis, pixels_cls_scores)`
     always, updating every call site to unpack three values (most will just discard the third).
     Correct under multi-GPU `DataParallel` (which only scatters/gathers through `forward()`
     itself), consistent with how this repo already tolerated similar tuple-shape churn earlier
     (single-vector to per-part `(f_out, vis)`).
   - **Option B:** add a separately-named method (e.g. `forward_with_pixel_scores`) called only for
     the source slice when BPA loss is enabled. Less invasive, but a non-`forward` method bypasses
     `DataParallel`'s scatter/gather entirely — silently loses multi-GPU parallelism for that call
     specifically, or breaks outright on multi-GPU depending on device placement. Only reasonable
     if this repo commits to single-GPU for BPA-loss runs.
4. A widened but still config-file-free hyperparameter surface. Each new loss is an *optional*,
   independently-weighted term (`weight=0` disables it, mirroring GiLt's own convention) — this
   means several new inline argparse flags per script, not a new config layer.
5. Reuse, not reinvention, of this repo's existing visibility-gating convention
   (`pcr/utils/part_distance.py::masked_mean`/`replace_values`) wherever a new loss needs
   per-branch visibility masking — with the same caveat already logged for
   `PartViewContrastiveLoss`: the "-1 / +1 sentinel" convention is correct for *distances*, wrong
   for anything where "no mutual visibility" should read as *neutral*, not *extreme*. Each new loss
   below states which convention it actually needs.

**Per-loss plan:**

**1. Body Part Attention (BPA) loss — new `pcr/loss/body_part_attention_loss.py`.**
Port `BodyPartAttentionLoss` near-verbatim (it's self-contained: cross-entropy/focal/dice over a
flattened `[N*Hf*Wf, K]` pixel classification, plus label smoothing — no bpbreid-internal `Writer`
dependency to strip, unlike `part_distance.py`). Depends on items 1-3 above (masks-aware source
loader + `pixels_cls_scores` exposed from the encoder).
- **Domain scope: source only, and `train_uda.py` only.** `train_usl.py` has no source domain and
  no masks anywhere — BPA loss must never be referenced there. This is the sharpest instance of
  "respecting source/target segregation" in this whole plan.
- **Pipeline placement:** inside `PCRTrainer_UDA.train`'s per-iteration loop
  (`pcr/trainers.py`), computed only on the source half of the joint DSBN batch, using Option A's
  `pixels_cls_scores_s` slice and the batch's loaded ground-truth masks (interpolated to
  `pixels_cls_scores`'s spatial size, argmax'd to integer targets, exactly as
  `part_based_engine.py::combine_losses` does). Added into `loss_s` alongside the existing
  `self.memory(f_out_s, s_targets, vis_s)` term: `loss_s = memory_loss_s + bpa_weight * bpa_loss`.
  Never touches `loss_t`.
- **Mutually exclusive with horizontal-stripes mode** (item 3 below): when `horizontal_stripes=True`,
  `pixels_cls_scores` is `None` by construction (`bpbreid.py` forward's stripe branch never builds
  a pixel classifier score map) — BPA loss must be a no-op (or raise clearly) whenever stripes mode
  is active, not silently skipped without explanation.
- **New CLI surface (`train_uda.py` only):** `--bpa-weight` (default 0, i.e. off unless opted in;
  mirrors bpbreid's own yaml default of 0.35 when explicitly enabled), `--masks-dir` (e.g.
  `pifpaf_maskrcnn_filtering`, matching the on-disk directory confirmed above).

**2. Body part triplet loss — new `pcr/loss/part_triplet_loss.py`.**
Port `PartAveragedTripletLoss` as the default combination strategy (matches bpbreid's own "SOTA
weights" default), with the other 6 combination strategies as selectable variants behind one
`--part-triplet-strategy` choice — this is a real, reasonably self-contained piece of math
(distance-matrix-per-part -> combine -> batch-hard mine -> margin loss), doesn't need masks at
all, operates directly on the `(f_out, vis)` this repo's encoder already returns everywhere.
- **Domain scope: both, and both scripts.** No masks needed — just part embeddings + a label per
  sample (real ID for source, pseudo-label for target/USL). No segregation concern here beyond the
  labels themselves already being handled correctly by each pipeline's existing pseudo-labeling
  step.
- **Pipeline placement, `train_uda.py`:** an additional term inside `PCRTrainer_UDA.train`,
  computed on `f_out_s`/`vis_s` against `s_targets` **and** `f_out_t`/`vis_t` against
  `t_indexes + source_classes`, each weighted by `--part-triplet-weight` and added to `loss_s`/
  `loss_t` respectively — same additive pattern as BPA loss, just applied to both halves instead
  of only source.
- **Pipeline placement, `train_usl.py`:** an additional term inside `ICEUSLTrainer.train`, computed
  on the student's `q`/`q_vis` (not the EMA teacher's `k`) against the batch's pseudo-labels, added
  to the existing `PartViewContrastiveLoss` term behind its own `--part-triplet-weight`.
- **Visibility convention:** reuse the *distance*-appropriate sentinel from
  `part_distance.py` (max-observed+1 / +inf depending on combination strategy) since this loss
  really is computing distances for hard-mining, unlike `PartViewContrastiveLoss`'s similarity
  case — the two shouldn't share one convention, and this file should say why in a short comment
  the way `contrastive.py` already does.

**3. Horizontal-stripes option — `BPBReIDModelCfg` change, not a new loss file.**
Add `horizontal_stripes: bool = False` and `num_stripes: int = 6` (matching bpbreid's own PCB
configs' stripe count) to `pcr/models/bpbreid_encoder.py::BPBReIDModelCfg`; thread
`horizontal_stripes=model_cfg.horizontal_stripes` **directly** into the `BPBreID(...)` constructor
call in `BPBReIDEncoder.__init__` (the correctly-spelled parameter — sidesteps bpbreid's own
`pcb()`/`bot()` typo bug by never going through those factories, which this repo already doesn't).
When enabled, `model_cfg.masks.parts_num` should be read as "number of stripes," and
`learnable_attention_enabled` should be forced `False` (mirroring bpbreid's *intent* for `pcb()`,
implemented correctly here instead of copying its bug).
- **Domain scope: both scripts** — this is an encoder architecture choice, orthogonal to which
  training strategy runs on top of it. `PartHybridMemory`, `compute_bpb_pairwise_distance`,
  `PartViewContrastiveLoss`, and the new part-triplet loss all already read `M` from tensor shape
  at runtime rather than hardcoding it, so none of them need changes to support a different M
  from stripes instead of learned/mask-derived parts.
- **New CLI surface (both scripts):** `--horizontal-stripes` (flag) + `--num-stripes` (int,
  default 6). Must stay consistent with whatever the loaded checkpoint was pretrained with, same
  caveat already logged for `parts_num=5` in `bpbreid_encoder.py`'s docstring.
- **Interaction:** disables BPA loss (see above) since `pixels_cls_scores` is `None` in this mode;
  compatible with body-part triplet loss and the existing memory/contrastive losses without changes.

**4. Hard triplet loss + soft triplet loss — new `pcr/loss/part_triplet_loss.py` (hard, shares the
file with item 2 — it's the same `[N,D]`-style batch-hard math, just without the part dimension)
and `pcr/loss/soft_part_triplet_loss.py` (soft).**
- **Hard triplet — domain scope: source only, `train_uda.py` only**, mirroring `mod_ice`'s actual
  (if unused-elsewhere) precedent: real ground-truth labels are exactly what batch-hard mining
  wants to be confident about, and source is the only domain with real labels here. Placement:
  same additive spot in `PCRTrainer_UDA.train` as BPA loss, on `f_out_s`
  (flattened/`bn_foreg`-branch or reuse the part-triplet loss's own per-part variant — this repo
  already has both a flat and a part-based path available, worth picking one rather than building
  both; recommend reusing item 2's part-based version here for consistency rather than adding a
  second, flat-vector-only triplet implementation).
- **Soft triplet — open design question, not silently resolved:** `SoftTripletLoss`'s actual shape
  (distillation between *two* forward passes of the same input) needs a second model to distill
  from. `train_usl.py` already has exactly that (the EMA teacher) — a soft triplet term there,
  between the student's `q` and teacher's `k`, pseudo-labeled, is a natural, low-risk fit alongside
  the existing `PartViewContrastiveLoss`. `train_uda.py` has **no EMA teacher today** — `PCRTrainer_UDA`
  has only one live model. Three options, presented rather than picked:
  (a) add an EMA teacher to `train_uda.py` specifically to support this (real architecture
  addition, mirrors `train_usl.py`'s pattern);
  (b) use `PartHybridMemory`'s own stored per-slot features as the "soft" reference instead of a
  second live model (no new model needed, but changes what "soft" means from teacher-distillation
  to memory-consistency);
  (c) skip soft triplet in `train_uda.py` entirely and treat it as USL-only, since that's where it
  fits without any new infrastructure.
  This plan does not choose between them — flagging for a decision before implementation, since
  each has different infrastructure cost and slightly different meaning.

**Suggested build order** (not started — planning only, per this session's request): (1) source-only
masks-aware preprocessor + BPA loss, since it's the most self-contained new *data* path and
exercises the `pixels_cls_scores` exposure decision (item 3) that a couple of the other pieces
implicitly assume is settled; (2) body-part triplet loss (both scripts) — no new data plumbing,
highest reuse of what already exists; (3) horizontal-stripes option — small, isolated config change,
useful to land before soft-triplet since it's a prerequisite-free architecture toggle; (4) hard
triplet (source, `train_uda.py`) — small once (2)'s triplet math exists to reuse; (5) soft triplet
— last, blocked on the open design question above being resolved first.

### 2026-08-19 16:58 CST — Implemented: GiLt loss + BPA loss + CrossEntropyLabelSmooth + view
### contrastive loss, in `train_usl.py` only

Per direct instruction: dropped camera-aware/agnostic memory from scope entirely (confirmed
already absent, see the clarification above), scoped this whole pass to `examples/train_usl.py`
only (`train_uda.py` untouched, still on the plain `PartHybridMemory` strategy), and implemented
first: GiLt loss, BPA loss, `CrossEntropyLabelSmooth`, and the (already-existing) view contrastive
loss "as in ICE." Horizontal stripes and hard/soft-triplet-with-source/target-segregation from
the earlier plan are *not* part of this pass.

**Key realization that shaped the design:** BPA loss needs masks, and masks are about part
*segmentation*, not identity — PifPaf masks are auto-generated by a pose model, so a dataset
having them doesn't violate "unsupervised" (no ID labels used). This means BPA loss only needs
"does the `-dt` dataset have a `masks/` directory," not a source/target split — resolved the
apparent tension with the earlier plan's UDA-only BPA scoping. Also realized GiLt's own native
id loss (BPBreID's fixed `BNClassifier` heads, sized at model-construction time) is a poor fit
for a per-epoch-reclustering setting: DBSCAN's cluster count and numbering both change every
epoch, so a persistent classifier layer would keep reassigning what each output neuron "means."
Resolved by making the id component classify against **per-epoch cluster centers** instead
(exactly ICE's own `ccloss` mechanism) — which is why "GiLt" and "CrossEntropyLabelSmooth as in
ICE" ended up being the *same* id component here, not two separate mechanisms, once adapted to
this setting.

**New files:**
- `pcr/loss/crossentropy.py::CrossEntropyLabelSmooth` — ported near-verbatim from `ice/loss/
  crossentropy.py`.
- `pcr/loss/body_part_attention_loss.py::BodyPartAttentionLoss` — ported from bpbreid's
  `torchreid/losses/body_part_attention_loss.py`, with two deliberate deviations: `monai` (only
  needed for the unused `'fl'/'dl'` loss-type variants) is imported lazily instead of at module
  level, so the default cross-entropy path doesn't require installing it; accuracy is a plain
  argmax-equality mean instead of `torchmetrics.Accuracy`, avoiding that package's version-coupled
  constructor API.
- `pcr/loss/part_triplet_loss.py::PartTripletLoss` — ported from bpbreid's
  `part_averaged_triplet_loss.py::PartAveragedTripletLoss` (the "averaged" combination strategy,
  bpbreid's own default). Reuses `masked_mean`/`replace_values` from `pcr/utils/part_distance.py`
  for the visibility-gated branch combination, but keeps its own epsilon-stabilized distance
  computation rather than reusing `part_distance.py`'s — that file's distances never backprop
  (eval/clustering only), this one does, and plain `sqrt` has an infinite gradient exactly at
  distance 0 (which real self-pairs hit).
- `pcr/loss/gilt_loss.py::PartGiLtLoss` — new, not a direct port (see "key realization" above): id
  component = `CrossEntropyLabelSmooth` against per-epoch foreground-branch (branch 0) cluster
  centers; triplet component = `PartTripletLoss` across all M branches. Weights default to
  bpbreid's own `foreg: id=1,tr=1 / parts: id=0,tr=1` convention (this repo's encoder only exposes
  foreground+parts branches, no separate global/concat_parts, so those two weight settings are
  all that's needed).

**Modified files:**
- `pcr/models/bpbreid_encoder.py` — added `forward_full()` returning `(f_out, vis,
  pixels_cls_scores)`, alongside the existing `forward()` (unchanged, still a clean 2-tuple).
  Deliberately chose "Option B" from the earlier plan (separate method, not an always-3-tuple
  `forward()`): smaller blast radius, keeps `train_uda.py`/`pcr/evaluators.py`/`pcr/trainers.py`
  completely untouched, matches this message's "only in the USL script" scope. Documented tradeoff
  in the method's own docstring: bypasses `nn.DataParallel`'s scatter/gather, so it only runs
  correctly single-device — fine for this repo's dev GPU, would need revisiting for real multi-GPU
  BPA-loss training.
- `pcr/utils/data/preprocessor.py` — added `PreprocessorMasked`: pairs view-1 with its PifPaf mask,
  applying resize/pad/crop/flip identically to both via shared random parameters (hand-rolled with
  plain `torchvision.transforms.functional`, not bpbreid's own albumentations-based mask pipeline —
  avoids a new dependency). Colour-only augmentation (blur, erasing) stays image-only, applied
  after, since it doesn't touch pixel *positions*. Channel grouping (36 raw PifPaf channels -> 5
  parts) and the synthetic background channel reuse bpbreid's own `torchreid.data.masks_transforms.
  {CombinePifPafIntoFiveVerticalParts,AddBackgroundMask}` directly (imported lazily) rather than
  re-deriving their channel-mapping tables by hand — verified these classes work standalone via
  `.apply_to_mask(tensor)` without needing a full yacs cfg object, confirmed directly before relying
  on it.
- `pcr/trainers_usl.py::ICEUSLTrainer` — added optional `bpa_criterion`/`bpa_weight` (constructor)
  and `gilt_criterion`/`centers` (per-`.train()`-call, since GiLt's id component needs rebuilding
  every epoch to match that epoch's cluster count — `vcl_criterion`/`bpa_criterion` don't, so they
  stay constructor args). `Loss_gilt_id`/`Loss_gilt_tri`/`Loss_bpa` now print alongside
  `Loss_hard_instance` when active.
- `examples/train_usl.py` — added `compute_cluster_centers()` (per-branch, visibility-weighted
  mean per cluster, same pattern as `train_uda.py`'s source-center init, computed fresh every
  epoch from the same `cf`/`cf_vis` already extracted for DBSCAN); split the view-1 transform into
  geometric (paired with the mask) vs. photometric (image-only) pieces; new CLI: `--gilt-id-weight`/
  `--gilt-triplet-weight` (both default **1.0**, i.e. GiLt is on by default in this build, matching
  the instruction to actually add it rather than leave it as an inert opt-in), `--gilt-tau-c`,
  `--gilt-triplet-margin`, `--masks-dir` (default `''`, i.e. **off** by default — most datasets
  don't have masks; `dukemtmc-reid` confirmed has none on disk, `market1501` does), `--masks-suffix`,
  `--bpa-weight` (default 0.35, matching bpbreid's own yaml).

**Real bug found and fixed while verifying (not a code-review guess — an actual crash on a real
run):** `IterLoader.next()`'s `except: self.iter = iter(self.loader); return next(self.iter)`
retry isn't guarded against a *second* consecutive `StopIteration` — and DBSCAN's clustering can
legitimately collapse to very few clusters (already observed twice now, independent of GiLt/BPA:
UDA's own dummy run epoch-to-epoch, and the plain-VCL USL dummy run earlier), which combined with
`MoreCameraSampler` (yields exactly `num_ids*num_instances` indices) and `drop_last=True` means a
degenerate epoch can leave the DataLoader unable to assemble even one batch — crashing the whole
run with an uncaught `StopIteration` on the very first `.next()` call. First repro run (`-dt
market1501`, masks + GiLt + BPA + VCL all active) hit exactly this at epoch 1 (11 samples, 1 id,
`1*num_instances(2)=2 < batch_size(4)`). Fixed in `train_usl.py`'s epoch loop: skip training (and
`lr_scheduler.step()`) for any epoch where `num_ids * num_instances < batch_size`, log why, and
re-cluster next epoch instead of crashing.

**Verified, in order:**
1. Synthetic tests (shapes, finite values, real gradients) for `PartTripletLoss` (bool/continuous/
   no visibility), `PartGiLtLoss`, and `BodyPartAttentionLoss` — all passed before touching the
   training script.
2. `PreprocessorMasked` tested directly against a real image+mask pair on disk
   (`market1501/bounding_box_train/0002_c1s1_000451_03.jpg` + its `.npy` mask) — confirmed shapes
   (`img1 [3,384,128]`, `mask [6,384,128]`) and that the mask sums to 1 per pixel as expected.
3. Full dummy training run (`-dt market1501`, real masks, `-b 4`, no checkpoint — mechanics-only,
   same discipline as every prior dummy run): first attempt crashed on the `IterLoader` bug above;
   after the fix, a second run (3 epochs x 8 iters) completed clean in ~10 minutes — all four loss
   terms (`Loss_hard_instance`, `Loss_gilt_id`, `Loss_gilt_tri`, `Loss_bpa`) finite every logged
   iteration across all 3 epochs, no NaN, checkpoint saved (144MB), end-of-training eval ran
   (mAP 0.3%, expected near-random for a random-init encoder over 3x8 iterations).

**Not done in this pass** (per explicit scope): `train_uda.py` untouched; horizontal-stripes option;
hard triplet (source, UDA) / soft triplet (open design question) from the earlier plan's items 3-4.

### 2026-08-19 — Fixed: backbone was never actually getting ImageNet-pretrained weights

Root cause: `BPBReIDEncoder.__init__` always called `BPBreID(..., pretrained=False, ...)`
unconditionally, regardless of backbone or whether a full checkpoint was given — so any run
without an external bpbreid checkpoint (in particular `train_usl.py` with no `--checkpoint-path`,
which is a supported/default mode) started from a **fully random** backbone, not ImageNet-init.
Both `hrnet32` and `resnet50` were already correctly registered in torchreid's model factory and
already threaded through `BPBReIDModelCfg.backbone` — the gap was specifically the hardcoded
`pretrained=False`, not missing backbone support.

Researched both backbones' actual weight-sourcing before changing anything (see the research
agent's findings, summarized here since they directly shaped the fix):
- **resnet50**: `torchreid/models/resnet.py::init_pretrained_weights` uses
  `torch.utils.model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')`
  — a real, stable, auto-downloading path, caching under `~/.cache/torch/hub/checkpoints/` (or
  `~/.cache/torch/checkpoints/` on older torch). Verified directly: constructed a resnet50-backed
  `BPBReIDEncoder` with `pretrained=True`, confirmed the exact expected file
  (`resnet50-19c8e357.pth`, 102MB) was loaded from cache and a forward pass produced correctly-
  shaped output.
- **hrnet32**: no stable URL exists anywhere in bpbreid's repo or docs for the raw ImageNet-only
  weights (`hrnetv2_w32_imagenet_pretrained.pth`) — `default_config.py`'s own comment says
  "download on our Google Drive" without giving a link/ID, `README.md`'s Google Drive link is for
  full *fine-tuned* bpbreid checkpoints (unconfirmed whether the raw ImageNet file is even in
  there), and there's no download script anywhere in the repo. **Did not fabricate a URL or guess
  a gdown ID** — auto-downloading against an unverified link risks silently fetching the wrong
  file. Instead, added a fail-fast check.

**Fix (`pcr/models/bpbreid_encoder.py`):**
- `pretrained = not checkpoint_path` — ImageNet-init the backbone whenever no full bpbreid
  checkpoint is going to overwrite it anyway (a full checkpoint's matching-shape weights get
  loaded over it afterward via the existing `load_pretrained_weights` call regardless, so
  requesting ImageNet init in that case would just be wasted download/compute).
- When `backbone == 'hrnet32'` and `pretrained` is true, check for
  `<hrnet_pretrained_path>/hrnetv2_w32_imagenet_pretrained.pth` *before* attempting model
  construction; raise `FileNotFoundError` with copy-pasteable instructions (both official sources,
  the exact expected path, and "use `--backbone resnet50` instead" as the zero-setup alternative)
  rather than letting it fail deep inside bpbreid's own `HighResolutionNet.load_param`. Verified
  this actually fires correctly (file genuinely absent in this environment).
- `BPBReIDModelCfg.hrnet_pretrained_path` default changed from `''` to `'pretrained_models'`,
  matching bpbreid's own yacs default.
- `--backbone` (`hrnet32`/`resnet50`) added to both `examples/train_uda.py` and `examples/
  train_usl.py`, threaded into `BPBReIDModelCfg(backbone=args.backbone)`. Default stays `hrnet32`
  in both (no behavior change for existing users/configs that already assume it, e.g. the
  `parts_num=5` gotcha tied to bpbreid's hrnet32-based yaml configs is unaffected by this — that's
  a masks-config axis, independent of backbone choice) — `resnet50` is the new, fully zero-setup
  option for anyone without a pretrained checkpoint or the manually-downloaded HRNet file. In
  `train_uda.py`, `--checkpoint-path` stays `required=True` (unchanged design: source pretraining
  is external, per this repo's original PLAN.md) so the ImageNet-fallback path mainly matters for
  `train_usl.py`, where `--checkpoint-path` is optional by design.

**Verified, not assumed:** both the hrnet32 fail-fast error and the resnet50 real-weights-loaded
path were actually run and their output inspected (not just read as "should work" from the code).

### 2026-08-19 — Vendored resnet50's backbone code into pcr2 directly

Per direct instruction: copied bpbreid's `torchreid/models/resnet.py` into `pcr/models/resnet.py`
rather than continuing to rely solely on the externally-installed torchreid package for it.
Checked first, not assumed: this file's only imports are `torch` and `torch.utils.model_zoo` —
zero references to `torchreid.losses`/`torchreid.utils`/anything else in that package, so there
was no actual dependency tree to bring along despite the request anticipating one ("models losses
utils etc") — copying the single file is the complete, correct vendoring, not a partial one.

**Not wired into the training pipeline** — `BPBReIDEncoder(backbone='resnet50')` still resolves
`'resnet50'` through torchreid's own internal model registry (that's an implementation detail
inside bpbreid's own `BPBreID.__init__` -> `models.build_model(...)`, not something this repo's
code controls), so this vendored copy doesn't change what the existing `--backbone resnet50` flag
does. It's a separate, standalone module for direct use (e.g. a plain non-part-based baseline
encoder, if one gets built later) — flagged to the user as not yet integrated, since the request
was scoped to "copy... and test it," not "make BPBreID use this copy instead."

**Verified rigorously, not just "imports cleanly":**
- `from pcr.models.resnet import resnet50; resnet50(num_classes=751, loss='softmax',
  pretrained=True, last_stride=1)` — real ImageNet weights load correctly through this vendored
  file specifically.
- Checked via **exact tensor equality** against the downloaded checkpoint
  (`~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth`), not just "non-random-looking stats"
  (which is what the earlier verification of the *external* torchreid resnet50 relied on) —
  `torch.equal(model.conv1.weight, ref['conv1.weight'])` and the same for a deep layer
  (`layer4.2.conv3.weight`) both returned `True`.
- Forward pass verified in both modes the backbone is actually used in: standard mode returns a
  global-pooled `[2, 2048]` feature; `model.loss = 'part_based'` mode (how BPBreID's own backbone
  wrapper drives it) returns the raw `[2, 2048, 16, 8]` feature map instead, matching what
  `BPBreID.forward()` expects from `self.backbone_appearance_feature_extractor(images)`.

### 2026-08-19 20:23 CST — Full USL logic audit: 1 critical bug, 1 monitoring gap, 2 medium fixes

Per direct instruction: re-read the entire USL path fresh (not from memory) and verified every
hypothesis empirically before reporting or fixing anything, rather than trusting the earlier
build-time reasoning at face value.

**🔴 CRITICAL, fixed — the EMA "teacher" was never actually a copy of the student.**
`examples/train_usl.py::create_model` built `model`/`model_ema` via two *separate*
`BPBReIDEncoder(...)` constructor calls. Verified empirically before touching anything: **15 of
409 parameter tensors differed** between the two at initialization, including
`pixel_classifier.classifier.weight` (the part-attention mechanism itself) and every dim-reduce
layer. This breaks the entire teacher-student premise every MoCo/BYOL/ICE-style EMA setup depends
on (the teacher must start byte-identical to the student, only diverging via the momentum update
afterward) -- and it's especially damaging here because `model_ema` both (a) supplies the view-
contrastive loss's `k` target and (b) generates the features used for **clustering every epoch**.
At `alpha=0.999` the initial divergence takes ~3000 iterations (7+ epochs at default settings) to
substantially wash out, so early pseudo-labels for a real run would come from a teacher with no
real relationship to the student. Root cause: this is the *same* pattern ICE's own
`unsupervised_train.py::create_model` uses (two independent `models.create(...)` calls, only
forced into agreement via `copy_state_dict` from the *same* checkpoint when `args.init != ''`) --
so this bug is latent in ICE's own reference script too whenever no init checkpoint is given, not
something introduced fresh here. **Fix**: build one model, `model_ema = copy.deepcopy(model)`.
Verified: re-ran the same diff check after the fix, 0/409 tensors differ.

**🟠 HIGH, fixed — no diagnostic existed for part-attention collapse.**
`learnable_attention_enabled=True` means the pixel-to-part classifier is trained from indirect
gradient alone whenever BPA loss isn't active or is being outweighed by other terms -- a known
failure mode for weakly-supervised attention. This exact risk was flagged in this project's very
first sanity-check pass, long before any of the USL work, with "monitor visibility stats" proposed
as a mitigation -- and never implemented until now. **Fix**: `ICEUSLTrainer.train` now tracks
`q_vis.float().mean(dim=0)` (mean per-branch visibility rate) across the epoch and prints it at
epoch end. Verified it actually prints correctly on a real run:
`Epoch 0 mean part-visibility rate per branch (branch 0 = foreground): ['1.00', '1.00', '1.00',
'1.00', '1.00', '1.00']` -- a legitimate baseline for a fresh/untrained pixel classifier (nothing
has learned to differentiate branches yet); a real trained run should show these diverge to more
varied, occlusion-dependent values, and the diagnostic is now in place to actually observe that
instead of only ever finding out indirectly from a bad final mAP.

**🟡 MEDIUM, fixed — resize interpolation mismatch between the masked and unmasked img1 paths.**
`pcr/utils/data/preprocessor.py::PreprocessorMasked` used `TF.resize(img, [h,w])` (defaults to
bilinear) for view-1's geometric pass, while `get_view2_transform`'s `T.Resize(...,
interpolation=3)` (bicubic) is used for view-2 and for the *entire* unmasked-path view-1. Whenever
BPA loss was enabled, the student's primary training view silently got a different resize kernel
than the teacher's view -- an avoidable pipeline inconsistency. Fixed to explicitly match
(`interpolation=InterpolationMode.BICUBIC`).

**🟡 MEDIUM, fixed (per explicit direction) — GiLt id-loss's raw scale tracked num_ids, not
learning progress.** `CrossEntropyLabelSmooth`'s chance-level baseline is ~`log(num_clusters)`,
and DBSCAN's cluster count is directly observed to swing wildly epoch to epoch on an
under-trained encoder (335 -> 9 -> 1 in an earlier dummy run) for reasons unrelated to model
quality. With a fixed `gilt_id_weight`, this term's pull on the total gradient balance shifted
inconsistently for reasons having nothing to do with training progress. **Fix** (user chose
"normalize by log(num_ids)" over "leave as-is" or "skip"): `PartGiLtLoss` now divides the id-loss
by `log(max(num_clusters, 2))` before weighting it into the total loss, while still returning/
logging the *unnormalized* value (the interpretable "how confident is the classifier" number).
Verified on a real run: printed `Loss_gilt_id` values (5.24 at 198 clusters, 4.94 at 143 clusters)
closely track `log(num_ids)` (5.29, 4.96) as expected -- confirms the raw diagnostic stayed
interpretable while the actual backprop term is now the normalized one.

**Investigated and explicitly ruled out** (checked rigorously, not just assumed correct, so they
don't need to be re-litigated in a future pass): visibility-score dtype (verified genuinely
`torch.bool`, not a float lookalike, so the bool/continuous branches in `contrastive.py`/
`part_triplet_loss.py` both correctly hit their intended path); the 0-similarity fallback for
zero-mutual-visibility branch pairs in `PartViewContrastiveLoss._combine_similarity` (traced
through both the hardest-positive selection and the InfoNCE denominator with `T=0.1`'s actual
dynamic range -- confirmed benign on both sides, not just asserted); diagonal (self-pair)
inclusion in `PartViewContrastiveLoss` (intentional -- matches MoCo/ICE's own instance-
discrimination design, q[i] vs k[i] being the same physical image under two augmentations is a
genuinely valid positive candidate, not an oversight); `PartGiLtLoss`'s `tau_c=0.5` (matches ICE's
own default exactly, not invented here). Also noted, informational only, not fixed: ShuffleBN
(`_get_shuffle_ids`) is correct code but is inert on this project's current single-GPU dev setup --
its whole purpose is decorrelating BN statistics *across* GPUs in `nn.DataParallel`, and batch
statistics on one GPU are invariant to element ordering regardless of shuffling. Will start doing
real work automatically once run on 2+ GPUs; not a bug to fix now.

**Verified end-to-end after all four fixes landed together**, not just individually: a real
training run (`-dt market1501`, `--backbone resnet50`, real masks, GiLt + BPA + VCL all active,
2 epochs) completed clean, no NaN, no crash, checkpoint saved, visibility diagnostic printing
correctly each epoch.
