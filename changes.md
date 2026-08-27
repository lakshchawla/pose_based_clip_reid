# Changes

Pending/proposed changes that have been identified and discussed but deliberately not yet
implemented -- held for later decision or action. Once a change here is actually made, it moves
out of this file and gets its own entry in `progress.md` (which records what was done, not what's
still pending).

## Changes to be done

### 1. Stage 1's per-part loss is still symmetric (i2t + t2i), now via `InfoNCELoss`

`SupConLoss` has been replaced with `pcr/loss/clip_infonce_loss.py::InfoNCELoss` (per direct user
request), resolving the loss-*type* half of what this entry used to flag. The loss is still
computed symmetrically internally (`InfoNCELoss.forward` returns `loss_i2t + loss_t2i`), while
"Algorithm 1"'s own step 15 specifies one direction only (`InfoNCE(relation_feats[:,k], t_i^k)`).
Kept symmetric since that wasn't part of the loss-type request and matches this repo's established
convention (CLIP-ReID/CLIP's own symmetric training loop) -- but it's still a real deviation from
the algorithm's literal wording, doubling the number of loss terms per part (2 instead of 1).

**Decision needed**: keep symmetric (current behavior), or switch to single-direction only
(visual->text) to match the algorithm literally.

### 2. `VisualAttentionBlock` (and `TextualAttentionBlock`) still perform fully unmasked
self-attention over possibly-invisible part tokens

The upstream image-level visibility filter (`pcr/utils/visibility_filter.py::filter_by_visibility`,
now deleted) has been replaced by per-part *weighting* inside Stage 1/2's own loss functions
(`InfoNCELoss`, `CosineAlignLoss`, and a loose boolean exclusion in `PartTripletLoss`) -- every
image, including ones where every part reads as invisible, now enters training; see `progress.md`'s
entry on this refactor for the full mechanism. `VisualAttentionBlock`/`TextualAttentionBlock`'s own
bidirectional self-attention over the K part tokens (`pcr/models/relation_blocks.py`) is
deliberately left untouched by this refactor: both still mix all K tokens completely
unconditionally, with no visibility-awareness of any kind. A genuinely-occluded part's
near-garbage pooled feature is therefore still blended, at full and equal weight, into every
other part's post-attention representation *before* any loss ever sees a token or gets a chance
to discount it -- the loss-level weighting introduced by this refactor only discounts a poorly-
visible part's own direct contribution to InfoNCE/align/triplet, it cannot undo contamination VAB
already mixed into its neighboring parts upstream of the loss.

This was a deliberate scope limit confirmed directly with the user, not an oversight: fixing it
would mean threading per-branch visibility into VAB/TAB's own attention computation (e.g. an
additive per-key attention-score bias proportional to `log(visibility)`, or something like
`nn.TransformerEncoderLayer`'s own `src_key_padding_mask` mechanics adapted to a soft score rather
than a hard pad mask), which is out of scope here.

**Decision needed**: whether/how to make VAB (and, symmetrically, TAB) visibility-aware in a
future pass, or whether loss-level weighting alone is judged sufficient once BPAM itself is
properly trained (60 epochs, per the original plan, not the current 1-epoch smoke test that
produced the training-collapse signature motivating this whole refactor) and visibility scores
stop being degenerate in the first place.

## Red flags found in review (2026-08-27, plain language)

A code-focused review and a research-methodology review were run against Stage 0, 1, and 2 as they
stand today. Below is what they found, explained simply -- what's wrong, why it matters, and what
to do about it. This section adds to the "Changes to be done" items above; it doesn't replace them
(those are still valid, just written in more technical language). Concrete code-level fixes with
math and paper references are in `IMPROVEMENT_PLAN.md`.

### 1. ~~(Most serious) The alignment loss can "cheat" by hiding a part instead of improving it~~ -- FIXED 2026-08-28

Stage 2's `CosineAlignLoss` now detaches its visibility weight before using it
(`weights.detach().clamp(...)`), closing the gradient path that let the model lower a part's own
visibility score instead of actually improving that part's alignment. See `progress.md`'s entry on
this fix for the full before/after and verification. `InfoNCELoss` (Stage 1) still has the same
weight-detach applied only implicitly (via its no-grad caching design, not the loss class itself)
-- carrying the explicit `.detach()` over there too remains a cheap, not-yet-done follow-up.

### 2. A body part can go completely silent in training with no further warning

**What's wrong**: A body part's triplet loss is skipped for a batch when every image in that batch
reads that part as "not visible enough." When that happens, Python prints one warning -- and then,
by default, never prints that same warning again for the rest of the run, no matter how many more
times it happens. The training log still looks normal (numbers just quietly stop including that
part), so nobody would notice one part effectively dropped out of training.

**Where**: `pcr/loss/part_triplet_loss.py`'s `warnings.warn(...)`, called from
`examples/train_relational_finetune.py`'s `compute_losses()`.

**Why it matters**: this is exactly the shape of the pathology already suspected in this repo (one
body-part branch reading as "invisible" almost everywhere, dataset-wide, because Stage 0's
part-attention head is currently only smoke-tested for 1 epoch instead of the intended 60). If it
recurs in a real run, nothing in the log would reveal it.

**Fix**: log how often this happens (a per-part "how many batches had zero valid triplets" counter,
printed periodically) instead of relying on a one-time warning. See `IMPROVEMENT_PLAN.md`
section 2.

### 3. A leftover scratch note sitting inside Stage 0's file

At the bottom of `examples/train_bpa_segmentation.py`, after the code that actually runs, there is
a stray quoted paragraph suggesting a completely different way to train Stage 0. It doesn't
execute or do anything, but it contradicts the file's own docstring and reads like an
accidentally-committed note. Harmless, but worth deleting (or moving into this file's own "Changes
to be done" list if the idea is worth keeping).

### 4. ~~The alignment loss has no way to say "not like this other person"~~ -- FIXED 2026-08-28

`CosineAlignLoss` is now a softmax classification against each branch's full identity-prototype
table (every other identity acts as an implicit negative), restoring the repulsion term the earlier
pure-regression version was missing -- see `progress.md`'s entry on this fix. One new thing to
watch, found while smoke-testing the fix: this loss's raw magnitude is now much larger than the old
`[0, 2]`-bounded regression (real smoke values were ~35-50 summed over 5 parts, vs. the old
version's max of ~10), so `loss.align_weight` (currently `0.5` in
`configs/stage2_relational_finetune.yaml`) may need retuning down once real training starts, or the
align term will dominate the total loss more than "Algorithm 2" intended.

### 5. Stage 1's "who counts as a negative example" pool is much smaller than the original design

**What's wrong**: Stage 1's `InfoNCELoss` avoids a same-identity-treated-as-negative bug by only
comparing the *unique* identities present in the current batch. With this repo's current batch
settings (32 images, 4 per identity), that's only about 8 identities compared against each other
per training step -- far fewer than, e.g., comparing against all ~751 identities in the training
set. Fewer negatives per step is a weaker, noisier training signal (telling 8 people apart is a
much easier task than telling 751 apart).

**Where**: `pcr/loss/clip_infonce_loss.py`; `configs/stage1_relational_prompts.yaml`'s
`data.batch_size` / `data.num_instances`.

**Fix**: see `IMPROVEMENT_PLAN.md` section 4 for two options (bigger PK batches, or a small memory
bank of past batches' text anchors to widen the negative pool without a bigger batch).

### 6. Unmasked attention blocks spread bad information into good parts, not just their own loss term

This is the mechanism behind why item 2 in "Changes to be done" above (VAB/TAB doing fully
unmasked self-attention) is worse than it first sounds. Self-attention doesn't just fail to help an
occluded part -- it actively blends that part's near-garbage feature into every *other*, perfectly
visible part's representation, because attention mixes all K parts together unconditionally. A
single badly-occluded part (say, a bag blocking someone's legs) can quietly degrade the "head,"
"torso," and "arms" features too, even though those are perfectly visible in the same photo.
Down-weighting a loss term (what the earlier visibility refactor did) only reduces how much that
bad part's *own* mistake counts in the loss -- it does nothing to undo the contamination that part
already spread to its neighbors before the loss ever saw them.

**Fix**: see `IMPROVEMENT_PLAN.md` section 5 for a concrete, visibility-weighted attention fix.
