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
Stage 1 (train_prompts.py)  -->  Stage 2 (train_finetune.py)  -->  Stage 3 (train_uda.py or train_usl.py)
per-part CLIP prompt            supervised backbone finetune         domain adaptation (UDA) or
learning, backbone frozen       + CLIP alignment loss                single-domain unsupervised (USL)
```

Run the whole thing with `examples/run_pipeline.py`, or any subset of stages directly.

### Stage 1 -- per-part CLIP prompt learning

```bash
python examples/train_prompts.py --config configs/stage1_prompt_learning.yaml
```

Frozen BPBreID encoder (ImageNet-init by default, or set `model.checkpoint_path` to an
externally-pretrained BPBreID checkpoint) + frozen CLIP text encoder; only a per-(identity,
branch) learnable prompt context trains. Produces `prompt_learner.pth` and `text_prototypes.pth`
(a frozen per-identity, per-branch text-embedding lookup table) under `logging.logs_dir`.

### Stage 2 -- supervised backbone finetune

```bash
python examples/train_finetune.py --config configs/stage2_backbone_finetune.yaml
```

Set `stage1.prompt_dir` in the config to Stage 1's `logging.logs_dir`. Trains the full BPBreID
encoder with real identity labels: id loss + triplet loss + an alignment loss against Stage 1's
frozen text prototypes, plus an optional BPA (body-part-attention) loss if `data.masks_dir` is
set (masks are source-domain-only on disk for Market1501; DukeMTMC-reID has none). Produces a
checkpoint directly loadable by Stage 3's `--checkpoint-path`, unchanged.

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
    --stage1-config configs/stage1_prompt_learning.yaml \
    --stage2-config configs/stage2_backbone_finetune.yaml \
    --stage3 uda \
    -ds dukemtmc-reid -dt market1501 --data-dir /path/to/datasets --logs-dir logs/full_run
```

Runs the selected stages in order, stopping on the first failure. Any argument this script
doesn't recognize (like `-ds`/`-dt`/`--data-dir`/`--logs-dir` above) is passed straight through to
the Stage 3 script; `--checkpoint-path`/`--backbone` are auto-filled from Stage 2's checkpoint and
config unless already given explicitly. Omit `--stage1-config`/`--stage2-config` to skip those
stages (e.g. to run Stage 3 alone against an externally-pretrained checkpoint); pass
`--stage3 none` to stop after Stage 2.

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
