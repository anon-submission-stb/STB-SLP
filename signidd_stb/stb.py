# coding: utf-8
"""Spatio-Temporal Block (STB).

A single MixSTE-style block (depth 1): spatial self-attention over the joint tokens of each frame, followed by
temporal self-attention over the frames of each joint token. The output has the same shape as the input
(B, T, J, C) -> (B, T, J * C), so the block can be inserted in front of an existing denoiser.

Parameter names follow MixSTE (https://github.com/JinluZhang1126/MixSTE) so that trained checkpoints load directly.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ["MixSTE2"]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn_mask = None
        if mask is not None:
            # Boolean mask (True = attend). Padded query rows attend to themselves, so no row is fully masked.
            # A fully masked row with an additive -1e9 mask gives unstable gradients in SDPA, which matters
            # when the backbone back-propagates into padded frames (e.g. FGDM's GCN).
            attn_mask = mask.bool() | torch.eye(N, dtype=torch.bool, device=x.device)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.attn_drop.p if self.training else 0.0, scale=self.scale)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        return x + self.mlp(self.norm2(x))


class MixSTE2(nn.Module):
    """STB with one spatial and one temporal transformer block.

    Args:
        num_frame: maximum sequence length (size of the temporal positional embedding)
        num_joints: number of tokens per frame
        in_chans: channels per token (input and output)
        embed_dim_ratio: token embedding size
        depth: kept for checkpoint compatibility; STB uses depth 1
    """

    def __init__(self, num_frame=500, num_joints=50, in_chans=3, embed_dim_ratio=32, depth=1,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None):
        super().__init__()
        assert depth == 1, "STB uses a single spatial and a single temporal block"
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio

        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        block = partial(Block, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
        self.STEblocks = nn.ModuleList([block(dim=embed_dim_ratio)])
        self.TTEblocks = nn.ModuleList([block(dim=embed_dim)])
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, in_chans))

    def STE_forward(self, x):
        """Spatial attention among the joint tokens of each frame."""
        f = x.shape[1]
        x = rearrange(x.to(self.Spatial_patch_to_embedding.weight.dtype), 'b f n c -> (b f) n c')
        x = self.pos_drop(self.Spatial_patch_to_embedding(x) + self.Spatial_pos_embed)
        x = self.Spatial_norm(self.STEblocks[0](x))
        return rearrange(x, '(b f) n c -> (b n) f c', f=f)

    def TTE_forward(self, x, mask):
        """Temporal attention over the frames of each joint token (padded frames are masked)."""
        x = self.pos_drop(x + self.Temporal_pos_embed[:, :x.shape[1], :])
        return self.Temporal_norm(self.TTEblocks[0](x, mask))

    def forward(self, x, trg_mask):
        """x: (B, T, J, C); trg_mask: (B, 1, T) or (B, T, T) boolean padding mask. Returns (B, T, J * C)."""
        b, f, n, _ = x.shape
        x = self.STE_forward(x)
        mask = (trg_mask & trg_mask.transpose(-1, -2)).repeat_interleave(n, dim=0)
        x = self.TTE_forward(x, mask)
        x = self.head(rearrange(x, '(b n) f c -> b f n c', n=n))
        return x.reshape(b, f, -1)
