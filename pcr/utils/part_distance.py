"""BPBReID-style part-based pairwise distance.

Ported from bpbreid's torchreid/metrics/distance.py
(compute_distance_matrix_using_bp_features + _compute_body_parts_dist_matrices) and
torchreid/utils/tensortools.py (masked_mean/replace_values). Dropped: Writer telemetry
hooks (bpbreid-internal debugging, not needed here), the use_gpu-driven .cuda() calls and
gallery batching (this repo always runs on modest re-ID gallery sizes, so a single dense
[Nq, Ng] pass is simpler and numerically identical to the batched original).

This is THE explicit part-based matching function reused by jaccard_rerank.py (as the base
distance for k-reciprocal re-ranking) and evaluators.py (query-gallery matching at eval time).
"""
import torch
from torch.nn import functional as F


def replace_values(input, mask, value):
    return input * (~mask) + mask * value


def masked_mean(input, mask):
    """output -1 where the mean couldn't be computed (no visible branch in common)."""
    valid_input = input * mask
    mean_weights = mask.sum(0)
    mean_weights = mean_weights + (mean_weights == 0)  # avoid division by 0
    pairwise_dist = valid_input.sum(0) / mean_weights
    invalid_pairs = (mask.sum(dim=0) == 0)
    return replace_values(pairwise_dist, invalid_pairs, -1)


def _compute_body_parts_dist_matrices(qf, gf, metric='euclidean'):
    """qf: [Nq, M, D], gf: [Ng, M, D] -> [M, Nq, Ng] per-branch distance matrices."""
    qf = qf.transpose(1, 0)  # [M, Nq, D]
    gf = gf.transpose(1, 0)  # [M, Ng, D]
    if metric == 'euclidean':
        dot_product = torch.matmul(qf, gf.transpose(2, 1))
        qf_square_sum = qf.pow(2).sum(dim=-1)
        gf_square_sum = gf.pow(2).sum(dim=-1)
        distances = qf_square_sum.unsqueeze(2) - 2 * dot_product + gf_square_sum.unsqueeze(1)
        distances = F.relu(distances)
        distances = torch.sqrt(distances)
    elif metric == 'cosine':
        distances = 1 - torch.matmul(qf, gf.transpose(2, 1))
    else:
        raise ValueError('Unknown distance metric: {}. Use "euclidean" or "cosine"'.format(metric))
    return distances


def _bool_visibility_distance(qf, gf, qf_vis, gf_vis, dist_combine_strat, metric):
    body_part_dist = _compute_body_parts_dist_matrices(qf, gf, metric)  # [M, Nq, Ng]

    qf_vis_t = qf_vis.t()  # [M, Nq]
    gf_vis_t = gf_vis.t()  # [M, Ng]
    valid_mask = qf_vis_t.unsqueeze(2) * gf_vis_t.unsqueeze(1)  # [M, Nq, Ng] bool, mutual visibility

    if dist_combine_strat == 'max':
        masked_dist = replace_values(body_part_dist, ~valid_mask, -1)
        pairwise_dist, _ = masked_dist.max(dim=0)
    elif dist_combine_strat == 'mean':
        pairwise_dist = masked_mean(body_part_dist, valid_mask)
    else:
        raise ValueError('Body parts distance combination strategy "{}" not supported'.format(dist_combine_strat))

    # sentinel: pairs with zero mutual visibility across every branch get "max observed + 1"
    max_value = body_part_dist.max() + 1
    invalid_mask = (pairwise_dist == -1)
    pairwise_dist = replace_values(pairwise_dist, invalid_mask, max_value)
    return pairwise_dist


def _soft_visibility_distance(qf, gf, qf_vis, gf_vis, dist_combine_strat, metric):
    body_part_dist = _compute_body_parts_dist_matrices(qf, gf, metric)  # [M, Nq, Ng]

    qf_vis_t = qf_vis.t()  # [M, Nq]
    gf_vis_t = gf_vis.t()  # [M, Ng]
    soft_mask = torch.sqrt(qf_vis_t.unsqueeze(2) * gf_vis_t.unsqueeze(1))  # [M, Nq, Ng]

    pairwise_dist = masked_mean(body_part_dist, soft_mask)

    max_value = body_part_dist.max() + 1
    invalid_mask = (pairwise_dist == -1)
    pairwise_dist = replace_values(pairwise_dist, invalid_mask, max_value)
    return pairwise_dist


def compute_bpb_pairwise_distance(qf, qf_vis, gf=None, gf_vis=None, dist_combine_strat='mean', metric='euclidean'):
    """Part-based pairwise distance between two sets of BPBReID embeddings.

    qf, gf: [N, M, D] per-branch (foreground + K parts) embeddings.
    qf_vis, gf_vis: [N, M] visibility, either bool (hard) or float in [0, 1] (soft).
    If gf/gf_vis are omitted, computes the self-distance qf vs qf (used by Jaccard re-ranking).

    Returns: Tensor[Nq, Ng] distance matrix.
    """
    if gf is None:
        gf, gf_vis = qf, qf_vis

    if qf_vis.dtype == torch.bool and gf_vis.dtype == torch.bool:
        return _bool_visibility_distance(qf, gf, qf_vis, gf_vis, dist_combine_strat, metric)
    else:
        qf_vis = qf_vis.float()
        gf_vis = gf_vis.float()
        return _soft_visibility_distance(qf, gf, qf_vis, gf_vis, dist_combine_strat, metric)
