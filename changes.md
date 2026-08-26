# Changes

Pending/proposed changes that have been identified and discussed but deliberately not yet
implemented -- held for later decision or action. Once a change here is actually made, it moves
out of this file and gets its own entry in `progress.md` (which records what was done, not what's
still pending).

## Changes to be done

### 1. Stage 1's per-part loss: symmetric (i2t + t2i) vs. the algorithm's literal single-direction InfoNCE

`examples/train_relational_prompts.py`'s per-part loss (lines ~270-273) computes `SupConLoss`
symmetrically for each part k:

```python
loss = loss + supcon(visual_k, part_text, b_labels, b_labels) \
    + supcon(part_text, visual_k, b_labels, b_labels)
```

"Algorithm 1 -- Stage 1: Prompt + Relation Learning" (as supplied) step 15 specifies one direction
only: `InfoNCE(relation_feats[:,k], t_i^k)`. The symmetric version is CLIP-ReID's own established
convention (and this repo's file docstring already documents it as a deliberate, ported choice),
so it wasn't changed during the 2026-08-26 Algorithm-1 rewrite -- but it is a real deviation from
the algorithm's literal wording, doubling the number of loss terms per part (2 instead of 1).

**Decision needed**: keep symmetric (current behavior, matches CLIP-ReID/CLIP's own convention),
or switch to single-direction only (visual->text) to match the algorithm literally.

### 2. (Minor, pre-existing, lower priority) `VisualAttentionBlock`'s output isn't renormalized before `SupConLoss`

`VisualAttentionBlock.forward` returns `part_tokens + tanh(gate) * relation_out` -- a residual sum
that can drift away from unit norm as `gate` moves off zero during training. `SupConLoss` computes
raw dot-product logits scaled by a fixed `temperature=1.0`, implicitly assuming its inputs are
already close to unit-normalized (a true cosine similarity). This predates the Algorithm-1 rewrite
(not introduced by it) and the algorithm text doesn't specify normalization either way, so it's
not a strict violation -- just a place where the loss's temperature calibration could quietly
drift as training progresses.

**Decision needed**: leave as-is (consistent with how `CosineAlignLoss`/other losses in this repo
already treat VAB's output, per `train_relational_finetune.py`'s identical pattern -- notably,
`CosineAlignLoss` itself is scale-invariant since cosine similarity ignores norm, so this concern
is really only about `SupConLoss`'s raw dot-product logits), or add an explicit `F.normalize` on
the visual side before `SupConLoss` in Stage 1 (and/or Stage 2).
