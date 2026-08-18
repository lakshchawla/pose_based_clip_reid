# PCR

PCR = BPBReID's part-attention feature pooling (K learnable body parts + visibility scores,
no external masks needed at forward time) trained under SpCL's baseline self-paced-clustering
UDA strategy (Jaccard/k-reciprocal re-ranking -> DBSCAN -> self-paced reliability filtering ->
contrastive memory-bank loss -> DSBN), for cross-domain person re-ID between Market1501 and
DukeMTMC-reID.

Repo layout mirrors SpCL's own shape (`examples/` + a `pcr/` package with `models/`, `utils/`,
`datasets/`, `evaluation_metrics/`), not BPBReID/torchreid's.

## Install

```bash
pip install -e .
pip install -e /path/to/bpbreid   # provides the `torchreid` package this repo's
                                   # pcr/models/bpbreid_encoder.py imports
```

## 1. Source pretraining (external, unmodified)

Source pretraining is done entirely with bpbreid's own training pipeline, outside this repo:

```bash
cd /path/to/bpbreid
python torchreid/scripts/main.py --config-file configs/bpbreid/bpbreid_market1501_train.yaml
```

(or `bpbreid_dukemtmc_train.yaml` for the other source domain). If PifPaf masks aren't
available, pass `loss.part_based.weights.pixls.ce=0` on the CLI so the pixel-supervision
loss is skipped and the attention trains unsupervised, purely from the ID/triplet gradient --
pcr2 never uses masks anyway (`learnable_attention_enabled=True`).

This produces a checkpoint (`.pth.tar`) that `examples/train_uda.py --checkpoint-path` loads
into the encoder.

## 2. UDA training

```bash
python examples/train_uda.py \
    -ds dukemtmc-reid -dt market1501 \
    --checkpoint-path /path/to/source_pretrained.pth.tar \
    --data-dir /path/to/datasets \
    --logs-dir logs/duke2market
```

`--data-dir` should contain `<data-dir>/market1501/{bounding_box_train,query,bounding_box_test}`
and `<data-dir>/dukemtmc-reid/DukeMTMC-reID/{bounding_box_train,query,bounding_box_test}`.

Use `--setup-only` for a dry run that builds the model/memory/optimizer and exits before
training, to sanity-check shapes.
