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

