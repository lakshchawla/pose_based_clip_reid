"""Per-(identity, body-part-branch) generalization of CLIP-ReID's PromptLearner
(../CLIP-ReID/model/make_model_clipreid.py, lines 191-239). One learnable context-vector sequence
per (identity, branch) pair -- branch 0 = foreground (matches BPBReIDEncoder.forward's convention,
pcr/models/bpbreid_encoder.py), branches 1..K = the K learned body parts -- sharing one fixed
template ("a photo of a X X X X person.") and one frozen prefix/suffix embedding buffer across all
identities and branches, per reid_pipeline_plan.md section 3.1's prompt_i_k design.

n_ctx serves two purposes that CLIP-ReID's original code keeps as two separately-named variables
(n_ctx, n_cls_ctx) always set to the same literal 4: the number of "X" placeholder tokens in the
fixed template (which determines where the frozen prefix ends and suffix begins), and the length
of the learnable context spliced into their place. One name here instead of two, since they must
always be equal for the prefix/suffix slicing to line up and splitting them buys nothing.

Deviation from CLIP-ReID's original, found by actually running the training loop (Stage 3's real
smoke run), not by inspection: CLIP-ReID creates cls_ctx directly in clip_model.dtype (fp16 on
GPU). `torch.amp.GradScaler.step()` hard-errors ("Attempting to unscale FP16 gradients") on an
fp16 leaf parameter -- mixed-precision training requires fp32 master weights, an fp16 trainable
parameter isn't the supported pattern regardless of what CLIP-ReID's own (evidently never run
under a GradScaler this strict) code does. Fixed here: cls_ctx is created and stored in fp32
always, cast to the frozen buffers' dtype only inside forward() when assembling the prompt
embedding -- standard fp32-master/fp16-compute practice, and autograd handles the cast's backward
correctly (the incoming gradient is cast back to fp32 for accumulation into the fp32 parameter).
"""
import clip
import torch
import torch.nn as nn


class PartPromptLearner(nn.Module):
    def __init__(self, num_identities, num_branches, clip_text_encoder, n_ctx=4, device='cuda'):
        super(PartPromptLearner, self).__init__()
        ctx_dim = clip_text_encoder.embed_dim
        dtype = clip_text_encoder.dtype

        ctx_init = "A photo of a " + " ".join(["X"] * n_ctx) + " person."
        tokenized_prompts = clip.tokenize(ctx_init).to(device)
        with torch.no_grad():
            embedding = clip_text_encoder.token_embedding(tokenized_prompts).type(dtype)

        # fp32 master weights, cast to the frozen buffers' dtype only in forward() -- see the
        # module docstring for why (GradScaler forbids fp16 leaf parameters)
        cls_vectors = torch.empty(num_identities, num_branches, n_ctx, ctx_dim, dtype=torch.float32)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)
        self.prompt_dtype = dtype

        # not trained, but must move with the module (.cuda()/.to()) -- registered as buffers
        # rather than plain attributes, unlike CLIP-ReID's own hardcoded .cuda() call
        self.register_buffer('tokenized_prompts', tokenized_prompts)  # [1, 77]
        self.register_buffer('token_prefix', embedding[:, :n_ctx + 1, :])
        self.register_buffer('token_suffix', embedding[:, n_ctx + 1 + n_ctx:, :])

        self.num_identities = num_identities
        self.num_branches = num_branches
        self.n_ctx = n_ctx

    def forward(self, labels, branch_idx):
        """labels: [B] identity indices. branch_idx: int in [0, num_branches). Returns
        [B, 77, ctx_dim] assembled prompt embeddings for that one branch, splicing the
        per-(identity,branch) learnable context between the frozen prefix/suffix."""
        cls_ctx = self.cls_ctx[labels, branch_idx].type(self.prompt_dtype)  # [B, n_ctx, ctx_dim]
        b = labels.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        return torch.cat([prefix, cls_ctx, suffix], dim=1)
