"""Frozen CLIP text tower, faithful to CLIP-ReID's TextEncoder
(../CLIP-ReID/model/make_model_clipreid.py, lines 31-48) -- only the text-side transformer,
positional_embedding, ln_final, text_projection and token_embedding are kept. CLIP's own visual
encoder is never loaded/used anywhere in this pipeline: BPBreID's backbone is the sole visual
encoder (pcr/models/bpbreid_encoder.py), CLIP contributes only a frozen semantic anchor via text.

Loaded via the official openai/CLIP package (`pip install git+https://github.com/openai/CLIP.git`,
done in pcr2-run), not CLIP-ReID's own vendored fork under CLIP-ReID/model/clip/ -- that fork is
patched specifically for CLIP-ReID's own ReID visual-encoder resolution/stride interpolation,
irrelevant here since the visual tower is never built.

ViT-B/32 is the default arch: its text embedding dim is 512, exactly matching
BPBReIDModelCfg.dim_reduce_output (pcr/models/bpbreid_encoder.py) -- no projection layer needed
between BPBreID's part embeddings and CLIP's text embeddings for the alignment losses.
"""
import clip
import torch
import torch.nn as nn


class ClipTextEncoder(nn.Module):
    def __init__(self, clip_arch='ViT-B/32', device='cuda'):
        super(ClipTextEncoder, self).__init__()
        clip_model, _ = clip.load(clip_arch, device=device, jit=False)
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.embed_dim = self.text_projection.shape[1]

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, prompts, tokenized_prompts):
        """prompts: [B, 77, ctx_dim] assembled prompt embeddings (from PartPromptLearner -- NOT
        raw token ids, the embedding lookup already happened). tokenized_prompts: [1, 77] integer
        token ids from the shared template, used only to locate each sequence's EOT position via
        argmax (EOT has the highest token id in CLIP's vocabulary); broadcasts against the batch
        since every prompt shares the same template structure regardless of its learnable
        context. Ported from CLIP-ReID's TextEncoder.forward.
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
