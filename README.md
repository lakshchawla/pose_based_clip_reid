# PCR

PCR combines three techniques into one pipeline, per `reid_pipeline_plan.md`:

1. **BPBreID** part-attention feature pooling (K learnable body parts + visibility scores, no
   external masks needed at forward time).
2. **CLIP-ReID**-style two-stage prompt learning -- a frozen CLIP text encoder regularizes
   training via per-(identity, body-part) learnable prompts (Stage 1), then the backbone is
   fine-tuned against those frozen prompts' text embeddings (Stage 2).
3. **SPCL**'s self-paced hybrid-memory strategy (Jaccard/k-reciprocal re-ranking -> DBSCAN ->
   self-paced reliability filtering -> contrastive memory-bank loss -> DSBN) for unsupervised
   domain adaptation, or an ICE-style single-domain unsupervised alternative.
4. **Part-relational attention** -- bidirectional self-attention across a person's K body-part
   tokens, on both the visual side (VisualAttentionBlock) and the text side (TextualAttentionBlock),
   in Stage 1/2 only (Stage 3 is untouched by this). See `pcr/models/relation_blocks.py` and
   `progress.md` for the design.

Repo layout mirrors SpCL's own shape (`examples/` + a `pcr/` package with `models/`, `loss/`,
`utils/`, `datasets/`, `evaluation_metrics/`), not BPBReID/torchreid's. Every hyperparameter for
the pre-existing UDA/USL drivers is an inline argparse default (no config files); the two newer
CLIP training stages are YAML-driven instead -- see `configs/`.

## Install

```bash
pip install -e .
pip install -e /path/to/bpbreid           # provides the `torchreid` package
                                            # pcr/models/bpbreid_encoder.py imports
pip install ftfy regex git+https://github.com/openai/CLIP.git   # frozen CLIP text encoder
```

## Pipeline stages

```
Stage 1                cache_text_anchors.py    Stage 2                    Stage 3
(train_relational        (build Stage 2's       (train_relational          (train_uda.py or
 _prompts.py)              alignment target)     _finetune.py)              train_usl.py)
per-part CLIP prompt   -->                   --> supervised backbone   --> domain adaptation
+ relation blocks,                               finetune + CLIP           (UDA) or single-
backbone frozen                                  alignment + VAB           domain unsupervised
```

Run the whole thing with `examples/run_pipeline.py`, or any subset of stages directly.

### Stage 1 -- per-part CLIP prompt learning + relational attention

```bash
python examples/train_relational_prompts.py --config configs/stage1_relational_prompts.yaml
```

Frozen BPBreID encoder (ImageNet-init by default, or set `model.checkpoint_path` to an
externally-pretrained BPBreID checkpoint) + frozen CLIP text encoder. Trainable: `PromptLearner`
(per-(identity, branch) learnable prompt context, owning a `TextualAttentionBlock` that mixes the K
part branches' context together before any part's prompt is built) and a `VisualAttentionBlock`
(mixes the K part branches' pooled visual features the same way). The training set is filtered
first by a visibility-index threshold (`visibility.lambda_v_min`) -- see
`pcr/utils/visibility_filter.py`. Produces `prompt_learner.pth` and `vab.pth` under
`logging.logs_dir`.

```bash
python examples/cache_text_anchors.py --config configs/stage1_relational_prompts.yaml
```

Run once, after Stage 1 finishes, against the *same* config. Loads `prompt_learner.pth` and
builds `text_prototypes.pth` (a frozen per-identity, per-branch text-embedding lookup table) --
the only one of Stage 1's outputs Stage 2 actually reads.

### Stage 2 -- supervised backbone finetune

```bash
python examples/train_relational_finetune.py --config configs/stage2_relational_finetune.yaml
```

Set `stage1.prompt_dir` in the config to Stage 1's `logging.logs_dir` (this is where both
`text_prototypes.pth` and `vab.pth` are read from). Trains the full BPBreID encoder with real
identity labels: id loss (foreground branch only) + triplet loss + an alignment loss against
Stage 1's frozen text prototypes, plus an optional BPA (body-part-attention) loss if
`data.masks_dir` is set (masks are source-domain-only on disk for Market1501; DukeMTMC-reID has
none). `VisualAttentionBlock` continues training here (loaded from Stage 1's `vab.pth`, jointly
with the now-unfrozen backbone) -- unlike `PromptLearner`/`TextualAttentionBlock`, which are frozen
and discarded after Stage 1. No per-branch visibility gating inside the loss loop here; the same
`visibility.lambda_v_min` filter Stage 1 uses is applied to this stage's own training set instead.
Produces a checkpoint directly loadable by Stage 3's `--checkpoint-path`, unchanged (Stage 3 is
untouched by VisualAttentionBlock -- its own weights save separately and aren't consumed yet).

### Stage 3 -- domain adaptation

Either UDA (source labeled + target unlabeled, SpCL-style self-paced hybrid memory):

```bash
python examples/train_uda.py \
    -ds dukemtmc-reid -dt market1501 \
    --checkpoint-path /path/to/stage2_or_external_checkpoint.pth.tar \
    --data-dir /path/to/datasets \
    --logs-dir logs/duke2market
```

...or USL (single target domain, no source, ICE-style EMA teacher-student):

```bash
python examples/train_usl.py \
    -dt market1501 \
    --checkpoint-path /path/to/stage2_or_external_checkpoint.pth.tar \
    --data-dir /path/to/datasets \
    --logs-dir logs/market_usl
```

`--data-dir` should contain `<data-dir>/market1501/{bounding_box_train,query,bounding_box_test}`
and `<data-dir>/dukemtmc-reid/DukeMTMC-reID/{bounding_box_train,query,bounding_box_test}`.
`--checkpoint-path` is required for `train_uda.py`, optional (ImageNet-init fallback) for
`train_usl.py`. Both accept `--setup-only` for a dry run that builds the model/memory/optimizer
and exits before training, to sanity-check shapes. Both also accept `--gilt-id-weight`/
`--gilt-triplet-weight`/`--masks-dir`/`--bpa-weight` to enable GiLt (id+triplet) and BPA loss
terms alongside the base memory/contrastive loss (`train_uda.py`'s id loss is source-only via a
persistent classifier; `train_usl.py`'s classifies against per-epoch cluster centers instead,
since its pseudo-label numbering changes every epoch).

### Orchestrator

```bash
python examples/run_pipeline.py \
    --stage1-config configs/stage1_relational_prompts.yaml \
    --stage2-config configs/stage2_relational_finetune.yaml \
    --stage3 uda \
    -ds dukemtmc-reid -dt market1501 --data-dir /path/to/datasets --logs-dir logs/full_run
```

Runs the selected stages in order (Stage 1 automatically triggers `cache_text_anchors.py`
afterward, against the same `--stage1-config`), stopping on the first failure. Any argument this
script doesn't recognize (like `-ds`/`-dt`/`--data-dir`/`--logs-dir` above) is passed straight
through to the Stage 3 script; `--checkpoint-path`/`--backbone` are auto-filled from Stage 2's
checkpoint and config unless already given explicitly. Omit `--stage1-config`/`--stage2-config`
to skip those stages (e.g. to run Stage 3 alone against an externally-pretrained checkpoint);
pass `--stage3 none` to stop after Stage 2.

### Alternative: external BPBreID source pretraining

Skipping Stages 1-2 entirely and pretraining with bpbreid's own pipeline instead is still
supported -- any checkpoint works for `--checkpoint-path`, this repo doesn't require it to come
from Stage 2:

```bash
cd /path/to/bpbreid
python torchreid/scripts/main.py --config-file configs/bpbreid/bpbreid_market1501_train.yaml
```

(or `bpbreid_dukemtmc_train.yaml` for the other source domain). If PifPaf masks aren't
available, pass `loss.part_based.weights.pixls.ce=0` on the CLI so the pixel-supervision loss is
skipped and the attention trains unsupervised, purely from the ID/triplet gradient.

## Fusion module

`pcr/models/fusion_head.py::FusionHead` fuses BPBreID's per-branch embeddings into a single
L2-normalized descriptor (fixed-weight or learned-gate visibility gating), for ANN/FAISS-style
retrieval. It's a standalone module, not wired into any training/eval script -- every script
above matches via `pcr/utils/part_distance.py::compute_bpb_pairwise_distance`'s native part-wise
distance instead, which is the default (and generally more accurate) matching strategy.
