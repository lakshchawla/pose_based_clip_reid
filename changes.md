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

