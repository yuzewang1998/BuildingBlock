"""
Adapted from: https://github.com/openai/openai/blob/55363aa496049423c37124b440e9e30366db3ed6/orc/orc/diffusion/vit.py
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn

# from sklearn.cluster import KMeans


from .hilbert import encode as hilbert_encode
from .checkpoint import checkpoint
from .pretrained_clip import FrozenImageCLIP, ImageCLIP, ImageType


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, input_dim=6, hidden_dims=[512], output_dim=512):
        super(LearnablePositionalEncoding, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(
            device=device, dtype=dtype, heads=heads, n_ctx=n_ctx
        )
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x


class WindowedMultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
        window_size: int,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = WindowedQKVMultiheadAttention(
            device=device,
            dtype=dtype,
            heads=heads,
            n_ctx=n_ctx,
            window_size=window_size,
        )
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class LinearAttentionCross(nn.Module):
    def __init__(self, dim, context_dim=None, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        if context_dim is None:
            context_dim = dim
        self.to_q = nn.Conv1d(dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Conv1d(context_dim, hidden_dim * 2, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x, c=None, context=None):  # =None):
        b, c, n = x.shape
        q = self.to_q(x)
        # if context is None:
        #     context = x
        kv = self.to_kv(context).chunk(2, dim=1)
        q = rearrange(q, "b (h c) n -> b h c n", h=self.heads)
        k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), kv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c n -> b (h c) n", h=self.heads)
        return self.to_out(out)


class MLP(nn.Module):
    def __init__(
        self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float
    ):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(
        self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class WindowedQKVMultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        heads: int,
        n_ctx: int,
        window_size: int,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx
        self.window_size = window_size

    def forward(self, qkv):
        # return self.forward_string(qkv)
        # assert self.forward_string(qkv).equal(self.forward_parallel(qkv))
        return self.forward_parallel(qkv)

    def forward_string(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3  # 3 for q,k,v
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        output = torch.zeros(
            (bs, n_ctx, width // 3), device=self.device, dtype=self.dtype
        )
        for i in range(0, n_ctx, self.window_size):
            q_window = q[:, i : i + self.window_size]
            k_window = k[:, i : i + self.window_size]
            v_windows = v[:, i : i + self.window_size]

            weight = torch.einsum("bthc,bshc->bhts", q_window * scale, k_window * scale)
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            # import ipdb;ipdb.set_trace()
            output[:, i : i + self.window_size] = torch.einsum(
                "bhts,bshc->bthc", weight, v_windows
            ).reshape(bs, self.window_size, -1)

        return output.reshape(bs, n_ctx, -1)

    def forward_parallel(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(attn_ch)

        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        num_windows = n_ctx // self.window_size
        q = q.view(bs, num_windows, self.window_size, self.heads, attn_ch)
        k = k.view(bs, num_windows, self.window_size, self.heads, attn_ch)
        v = v.view(bs, num_windows, self.window_size, self.heads, attn_ch)

        weight = torch.einsum("bnthc,bnshc->bnhts", q * scale, k * scale)
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        output = torch.einsum("bnhts,bnshc->bnthc", weight, v).reshape(bs, n_ctx, -1)

        return output


class WindowCPE(nn.Module):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        window_size=8,
    ):
        super().__init__()
        self.window_size = window_size
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.attnCpe = WindowedMultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            window_size=window_size,
        )

    def forward(self, x):
        return self.attnCpe(self.ln_1(x))


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        with_window: bool = False,
        window_size=8,
        cross_condition=False,
        cross_condition_dim=512,
        gate_shifting=True,
        pos_embeding_way="no_pos_embed",
    ):
        super().__init__()
        self.with_window = with_window
        self.window_size = window_size
        self.cross_condition = cross_condition
        self.gate_shifting = gate_shifting
        self.pos_embeding_way = pos_embeding_way

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )

        if self.cross_condition:
            self.cross_attn = CrossAttention(
                device=device,
                dtype=dtype,
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                init_scale=init_scale,
                cross_condition_dim=cross_condition_dim,
            )

        self.ln_1 = nn.LayerNorm(
            width, elementwise_affine=False, device=device, dtype=dtype
        )
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(
            width, elementwise_affine=False, device=device, dtype=dtype
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(width, 8 * width, bias=True)
        )
        # if pos_embeding_way=='windowCpe':
        if "windowCpe" in self.pos_embeding_way:
            self.window_cpe = WindowCPE(
                device=device,
                dtype=dtype,
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                init_scale=init_scale,
                window_size=window_size,
            )
        # if self.with_window:
        #     self.attn_window=WindowedMultiheadAttention(
        #         device=device,
        #         dtype=dtype,
        #         n_ctx=n_ctx,
        #         width=width,
        #         heads=heads,
        #         init_scale=init_scale,
        #         window_size=self.window_size,
        #     )
        #     self.ln_1_window = nn.LayerNorm(width,elementwise_affine=False, device=device, dtype=dtype)
        #     self.mlp_window=MLP(device=device,dtype=dtype,width=width,init_scale=init_scale)
        #     self.ln_2_window=nn.LayerNorm(width,elementwise_affine=False,device=device,dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor = None,
        cross_condition: torch.Tensor = None,
    ):
        # if self.with_window:
        #     x=x+self.attn_window(self.ln_1_window(x))
        #     x=x+self.mlp_window(self.ln_2_window(x))
        if "windowCpe" in self.pos_embeding_way:  # self.pos_embeding_way=='windowCpe':
            x = x + self.window_cpe(x)
        (
            shift_msa,
            scale_msa,
            shift_after_msa,
            scale_after_msa,
            shift_mlp,
            scale_mlp,
            shift_after_mlp,
            scale_after_mlp,
        ) = self.adaLN_modulation(c).chunk(8, dim=1)
        if not self.gate_shifting:
            shift_after_mlp = shift_after_mlp * 0
            shift_after_msa = shift_after_msa * 0

        x = x + modulate(
            self.attn(modulate(self.ln_1(x), shift_msa, scale_msa)),
            shift_after_msa,
            scale_after_msa,
        )

        if self.cross_condition:
            x = x + self.cross_attn(x, cross_cond=cross_condition)

        x = x + modulate(
            self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp)),
            shift_after_mlp,
            scale_after_mlp,
        )
        return x


class WindowResidualAttentionBlock(ResidualAttentionBlock):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        with_window: bool = False,
        window_size=8,
        cross_condition=False,
        cross_condition_dim=512,
        gate_shifting=True,
        pos_embeding_way="no_pos_embed",
    ):
        super().__init__(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            with_window=with_window,
            window_size=window_size,
            cross_condition=cross_condition,
            cross_condition_dim=cross_condition_dim,
            gate_shifting=gate_shifting,
            pos_embeding_way=pos_embeding_way,
        )
        self.attn = WindowedMultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            window_size=self.window_size,
        )


class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        cross_condition_dim=512,
    ):
        super().__init__()

        self.attn = torch.nn.MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_cross = nn.LayerNorm(cross_condition_dim, device=device, dtype=dtype)

    def forward(self, x, c: torch.Tensor = None, cross_cond: torch.Tensor = None):
        cross_cond = self.ln_cross(cross_cond)
        x_ln = self.ln_1(x)
        x_ln = torch.permute(x_ln, (1, 0, 2))
        cross_attn_out, _ = self.attn(x_ln, cross_cond, cross_cond)
        cross_attn_out = torch.permute(cross_attn_out, (1, 0, 2))
        return cross_attn_out


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
        with_window: bool = False,
        window_size=8,
        cross_condition=False,
        gate_shifting=True,
        pos_embeding_way="no_pos_embed",
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.window_size = window_size
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList([])
        for _ in range(layers):
            if with_window:
                self.resblocks.append(
                    WindowResidualAttentionBlock(
                        device=device,
                        dtype=dtype,
                        n_ctx=n_ctx,
                        width=width,
                        heads=heads,
                        init_scale=init_scale,
                        with_window=with_window,
                        window_size=window_size,
                        cross_condition=cross_condition,
                        cross_condition_dim=width,
                        gate_shifting=gate_shifting,
                        pos_embeding_way=pos_embeding_way,
                    )
                )
            self.resblocks.append(
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                    with_window=with_window,
                    window_size=window_size,
                    cross_condition=cross_condition,
                    cross_condition_dim=width,
                    gate_shifting=gate_shifting,
                    pos_embeding_way=pos_embeding_way,
                )
            )

    def forward(self, x: torch.Tensor, c: torch.Tensor = None, cross_condition=None):
        for block in self.resblocks:
            x = block(x, c, cross_condition)
        return x


class PointTransformer1D(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
        dim=256,  #
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        seperate_all=False,
        merge_bbox=False,
        objectness_dim=1,
        class_dim=21,
        translation_dim=3,
        size_dim=3,
        angle_dim=1,
        objfeat_dim=0,
        context_dim=256,
        instanclass_dim=0,
        modulate_time_context_instanclass=False,
        text_condition=False,
        text_dim=256,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        with_window: bool = False,
        window_size=8,
        serial_type="z_order",
        pos_embeding_way="no_pos_embed",
        size_half=True,
        gate_shifting=True,
        surface_loc=False,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        self.seperate_all = seperate_all
        self.objectness_dim = objectness_dim
        self.class_dim = class_dim
        self.translation_dim = translation_dim
        self.size_dim = size_dim
        self.angle_dim = angle_dim
        self.bbox_dim = translation_dim + size_dim + angle_dim
        self.objfeat_dim = objfeat_dim
        # self.modulate_time_context_instanclass =  modulate_time_context_instanclass
        self.text_condition = text_condition
        self.text_dim = text_dim
        self.surface_loc = surface_loc
        self.with_window = with_window
        self.window_size = window_size
        self.serial_type = serial_type
        self.pos_embeding_way = pos_embeding_way
        self.size_half = size_half
        # print(pos_embeding_way)
        if "learned" in self.pos_embeding_way:
            self.pos_embed = LearnablePositionalEncoding(6, [width], width)

        self.time_embed = MLP(
            device=device,
            dtype=dtype,
            width=width,
            init_scale=init_scale * math.sqrt(1.0 / width),
        )
        context_dim += instanclass_dim
        self.context_embed = (
            nn.Sequential(nn.Linear(context_dim, width, bias=True))
            if context_dim
            else None
        )

        self.ln_pre = nn.LayerNorm(
            width, elementwise_affine=False, device=device, dtype=dtype
        )
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            with_window=with_window,
            window_size=window_size,
            cross_condition=self.text_condition,
            gate_shifting=gate_shifting,
            pos_embeding_way=pos_embeding_way,
        )
        self.ln_post = nn.LayerNorm(
            width, elementwise_affine=False, device=device, dtype=dtype
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(width, 4 * width, bias=True)
        )

        if self.seperate_all:
            if self.objectness_dim > 0:
                self.objectness_embedf = PointTransformer1D._encoder_mlp(
                    dim, self.objectness_dim
                )

            if self.objfeat_dim > 0:
                self.objfeat_embedf = PointTransformer1D._encoder_mlp(
                    dim, self.objfeat_dim
                )

            self.class_embedf = PointTransformer1D._encoder_mlp(dim, self.class_dim)
            self.bbox_embedf = PointTransformer1D._encoder_mlp(
                dim, self.translation_dim + self.size_dim + self.angle_dim
            )

            input_channels = dim
            print(
                "separate PointTransformer1D encoder of objectness/class/translation/size/angle"
            )

        else:
            input_channels = channels
            print("PointTransformer1D encoder of all object properties")

        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, input_channels, device=device, dtype=dtype)

        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        if self.seperate_all:
            if self.objectness_dim > 0:
                self.objectness_hidden2output = PointTransformer1D._decoder_mlp(
                    dim, self.objectness_dim
                )

            if self.objfeat_dim > 0:
                self.objfeat_hidden2output = PointTransformer1D._decoder_mlp(
                    dim, self.objfeat_dim
                )

            self.class_hidden2output = PointTransformer1D._decoder_mlp(
                dim, self.class_dim
            )

            self.bbox_hidden2output = PointTransformer1D._decoder_mlp(
                dim, self.translation_dim + self.size_dim + self.angle_dim
            )
            print(
                "separate PointTransformer1D decoder of objectness/class/translation/size/angle"
            )

        else:
            self.final_conv = nn.Conv1d(dim, self.out_dim, 1)
            print("PointTransformer1D decoder of all object properties")

        # Zero-out adaLN modulation layers:
        for block in self.backbone.resblocks:
            if not isinstance(block, (CrossAttention, LinearAttentionCross)):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def z_order_sort_torch(self, points):
        """Compute Morton code (Z-order) based on 2D or 3D coordinates using GPU in PyTorch."""

        scaled_points = (points * 1024).to(torch.int64)  # Scale and cast to uint16

        # Initialize the z-order array
        num_points = scaled_points.shape[:-1]
        z_order = torch.zeros(num_points, dtype=torch.int64, device=points.device)

        # Bit interleaving
        for i in range(10, -1, -1):
            bit_x = (scaled_points[..., 0] >> i) & 1
            bit_y = (
                (scaled_points[..., 1] >> i) & 1 if scaled_points.shape[1] > 1 else 0
            )
            bit_z = (
                (scaled_points[..., 2] >> i) & 1 if scaled_points.shape[1] > 2 else 0
            )

            z_order |= bit_x << (3 * i + 2)
            z_order |= bit_y << (3 * i + 1)
            z_order |= bit_z << (3 * i)
        return z_order

    def hilbert_sort_torch(self, points):

        scaled_points = (points * 1024).to(torch.int64)
        hilberts_order = hilbert_encode(scaled_points, 3, 16)

        return hilberts_order

    def generate_order_values(self, points, shuffle_dim=False):
        # select main axis randomly
        if shuffle_dim:
            permute_order = (
                torch.randperm(3)
                .repeat(points.shape[0], points.shape[1], 1)
                .to(points.device)
            )
            point_tmp = points.gather(-1, permute_order)

        select = np.random.randint(2)
        if select == 0:
            order_values = self.z_order_sort_torch(point_tmp)
        elif select == 1:
            order_values = self.hilbert_sort_torch(point_tmp)
        return order_values

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, context=None, context_cross=None
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        # (B, N, C) --> (B, C, N)
        batch_size, num_points, point_dim = x.size()
        x_trans = x[:, :, : self.translation_dim].clamp(0, 1)

        x_sizes = x[:, :, self.translation_dim : self.translation_dim + self.size_dim]
        loc = x_trans
        if self.surface_loc:
            if self.size_half:
                if self.size_half:
                    xyzxyz = torch.cat((x_trans - x_sizes, x_trans + x_sizes), dim=-1)
                else:
                    xyzxyz = torch.cat(
                        (x_trans - 0.5 * x_sizes, x_trans + 0.5 * x_sizes), dim=-1
                    )

                x_left, x_right, y_front, y_back, z_down, z_up = torch.split(
                    xyzxyz, 1, dim=-1
                )  # b,l,6
                x_center, y_center, z_center = (
                    (x_left + x_right) / 2,
                    (y_front + y_back) / 2,
                    (z_down + z_up) / 2,
                )

                center1 = torch.cat((x_left, y_front, z_center), dim=-1).unsqueeze(2)
                center2 = torch.cat((x_left, y_back, z_center), dim=-1).unsqueeze(2)
                center3 = torch.cat((x_left, y_center, z_down), dim=-1).unsqueeze(2)
                center4 = torch.cat((x_left, y_center, z_up), dim=-1).unsqueeze(2)
                center5 = torch.cat((x_right, y_center, z_down), dim=-1).unsqueeze(2)
                center6 = torch.cat((x_right, y_center, z_up), dim=-1).unsqueeze(2)
                centers = torch.cat(
                    (center1, center2, center3, center4, center5, center6), dim=2
                )  # b,l,6,3
                b, l, _, _ = centers.shape
                random_indices = torch.randint(0, 6, (b, l))
                loc = centers[
                    torch.arange(b).unsqueeze(1),
                    torch.arange(l).unsqueeze(0),
                    random_indices,
                ]

        # B,N
        sorted_indices = None
        if (
            self.with_window or "windowCpe" in self.pos_embeding_way
        ):  # self.pos_embeding_way=='windowCpe':
            if self.serial_type == "z_order":
                order_values = self.z_order_sort_torch(loc)
            elif self.serial_type == "hilbert":
                order_values = self.hilbert_sort_torch(loc)
            elif self.serial_type == "mix":
                order_values = self.generate_order_values(loc)
            elif self.serial_type == "mix_shuffledim":
                order_values = self.generate_order_values(loc, shuffle_dim=True)
            _, sorted_indices = torch.sort(order_values, dim=-1, stable=True)
            x_sorted_batch = torch.gather(
                x, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, point_dim)
            )
            x_sorted = x_sorted_batch
            x = x_sorted

        pos = 0
        # if self.pos_embeding_way=='learned':
        if "learned" in self.pos_embeding_way:
            x_trans = x[:, :, : self.translation_dim].clamp(0, 1)
            x_sizes = x[
                :, :, self.translation_dim : self.translation_dim + self.size_dim
            ]
            if self.size_half:
                xyzxyz = torch.cat((x_trans - x_sizes, x_trans + x_sizes), dim=-1)
            else:
                xyzxyz = torch.cat(
                    (x_trans - 0.5 * x_sizes, x_trans + 0.5 * x_sizes), dim=-1
                )
            pos = torch.permute(self.pos_embed(xyzxyz), (0, 2, 1)).contiguous()

        x = torch.permute(x, (0, 2, 1)).contiguous()

        if self.seperate_all:
            x_class = self.class_embedf(
                x[:, self.bbox_dim : self.bbox_dim + self.class_dim, :]
            )
            if self.objectness_dim > 0:
                x_object = self.objectness_embedf(
                    x[
                        :,
                        self.bbox_dim
                        + self.class_dim : self.bbox_dim
                        + self.class_dim
                        + self.objectness_dim,
                        :,
                    ]
                )
            else:
                x_object = 0

            if self.objfeat_dim > 0:
                x_objfeat = self.objfeat_embedf(
                    x[
                        :,
                        self.bbox_dim
                        + self.class_dim
                        + self.objectness_dim : self.bbox_dim
                        + self.class_dim
                        + self.objectness_dim
                        + self.objfeat_dim,
                        :,
                    ]
                )
            else:
                x_objfeat = 0

            x_bbox = self.bbox_embedf(x[:, 0 : self.bbox_dim, :])
            x = x_class + x_bbox + x_object + x_objfeat

        # inject pos emb
        x = x + pos
        # import ipdb;ipdb.set_trace()
        if context_cross is not None:
            # [B, N, C] --> [B, C, N]
            # context_cross = torch.permute(context_cross, (0, 2, 1)).contiguous()
            context_cross = torch.permute(context_cross, (1, 0, 2)).contiguous()

        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        # import ipdb;ipdb.set_trace()

        context_emb = (
            self.context_embed(context).permute(0, 2, 1)
            if self.context_embed is not None
            else 0
        )

        x = x + context_emb

        x = self._forward_with_cond(x, [(t_embed, self.time_token_cond)], context_cross)

        if self.seperate_all:
            out_bbox = self.bbox_hidden2output(x)
            out_class = self.class_hidden2output(x)
            out = torch.cat([out_bbox, out_class], dim=1).contiguous()
            if self.objectness_dim > 0:
                out_object = self.objectness_hidden2output(x)
                out = torch.cat([out, out_object], dim=1).contiguous()

            if self.objfeat_dim > 0:
                out_objfeat = self.objfeat_hidden2output(x)
                out = torch.cat([out, out_objfeat], dim=1).contiguous()
        else:
            out = self.final_conv(x)

        # (B, N, C) <-- (B, C, N)
        out = torch.permute(out, (0, 2, 1)).contiguous()

        # for i in range(batch_size):
        #     out[i] = out[i][torch.argsort(original_indices[i])]
        if sorted_indices is not None:
            out = torch.gather(
                out,
                1,
                index=sorted_indices.argsort(dim=-1)
                .unsqueeze(-1)
                .expand(-1, -1, point_dim),
            )

        return out

    def _forward_with_cond(
        self,
        x: torch.Tensor,
        cond_as_token: List[Tuple[torch.Tensor, bool]],
        context_cross,
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        # h=x.permute(0, 2, 1)
        # for emb, as_token in cond_as_token:
        #     if not as_token:
        #         h = h + emb[:, None]
        # extra_tokens = [
        #     (emb[:, None] if len(emb.shape) == 2 else emb)
        #     for emb, as_token in cond_as_token
        #     if as_token
        # ]
        # if len(extra_tokens):
        #     h = torch.cat(extra_tokens + [h], dim=1)
        c = torch.zeros_like(cond_as_token[0][0])
        for emb, _ in cond_as_token:
            if emb is not None:
                c = c + emb
        # h = self.ln_pre(h)
        # h = self.backbone(h,c)
        # h = self.ln_post(h)
        # import ipdb;ipdb.set_trace()
        shift_pre, scale_pre, shift_post, scale_post = self.adaLN_modulation(c).chunk(
            4, dim=1
        )
        h = modulate(self.ln_pre(h), shift_pre, scale_pre)
        h = self.backbone(h, c, context_cross)
        h = modulate(self.ln_post(h), shift_post, scale_post)

        h = self.output_proj(h)
        return h.permute(0, 2, 1)

    @staticmethod
    def _encoder_mlp(hidden_size, input_size):
        mlp_layers = [
            nn.Conv1d(input_size, hidden_size, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size * 2, hidden_size, 1),
        ]
        return nn.Sequential(*mlp_layers)

    @staticmethod
    def _decoder_mlp(hidden_size, output_size):
        mlp_layers = [
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size * 2, hidden_size, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size, output_size, 1),
        ]
        return nn.Sequential(*mlp_layers)


class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.time_embed = MLP(
            device=device,
            dtype=dtype,
            width=width,
            init_scale=init_scale * math.sqrt(1.0 / width),
        )
        self.ln_pre = nn.LayerNorm(
            width, elementwise_affine=False, device=device, dtype=dtype
        )
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(
            width, elementwise_affine=False, device=device, dtype=dtype
        )
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(width, 4 * width, bias=True)
        )
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        # for emb, as_token in cond_as_token:
        #     if not as_token:
        #         h = h + emb[:, None]
        # extra_tokens = [
        #     (emb[:, None] if len(emb.shape) == 2 else emb)
        #     for emb, as_token in cond_as_token
        #     if as_token
        # ]
        c = torch.zeros_like(cond_as_token[0][0])
        for emb, _ in cond_as_token:
            c = c + emb
        shift_pre, scale_pre, shift_post, scale_post = self.adaLN_modulation(c).chunk(
            4, dim=1
        )

        # h = self.ln_pre(h)
        h = modulate(self.ln_pre(h), shift_pre, scale_pre)
        h = self.backbone(h, c)
        # h = self.ln_post(h)
        h = modulate(self.ln_post(h), shift_post, scale_post)
        # if len(extra_tokens):
        #     h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)


class CLIPImagePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 1024,
        token_cond: bool = False,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            device=device, dtype=dtype, n_ctx=n_ctx + int(token_cond), **kwargs
        )
        self.n_ctx = n_ctx
        self.token_cond = token_cond
        self.clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(
            device, cache_dir=cache_dir
        )
        self.clip_embed = nn.Linear(
            self.clip.feature_dim, self.backbone.width, device=device, dtype=dtype
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(
        self, batch_size: int, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        with torch.no_grad():
            return dict(embeddings=self.clip(batch_size, **model_kwargs))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        images: Optional[Iterable[Optional[ImageType]]] = None,
        texts: Optional[Iterable[Optional[str]]] = None,
        embeddings: Optional[Iterable[Optional[torch.Tensor]]] = None,
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :param texts: a batch of texts to condition on.
        :param embeddings: a batch of CLIP embeddings to condition on.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        clip_out = self.clip(
            batch_size=len(x), images=images, texts=texts, embeddings=embeddings
        )
        assert len(clip_out.shape) == 2 and clip_out.shape[0] == x.shape[0]

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None].to(clip_out)

        # Rescale the features to have unit variance
        clip_out = math.sqrt(clip_out.shape[1]) * clip_out

        clip_embed = self.clip_embed(clip_out)

        cond = [(clip_embed, self.token_cond), (t_embed, self.time_token_cond)]
        return self._forward_with_cond(x, cond)


class CLIPImageGridPointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 1024,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(
            device,
            cache_dir=cache_dir,
        )
        super().__init__(
            device=device, dtype=dtype, n_ctx=n_ctx + clip.grid_size**2, **kwargs
        )
        self.n_ctx = n_ctx
        self.clip = clip
        self.clip_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.clip.grid_feature_dim,),
                device=device,
                dtype=dtype,
            ),
            nn.Linear(
                self.clip.grid_feature_dim,
                self.backbone.width,
                device=device,
                dtype=dtype,
            ),
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(
        self, batch_size: int, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        _ = batch_size
        with torch.no_grad():
            return dict(embeddings=self.clip.embed_images_grid(model_kwargs["images"]))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        images: Optional[Iterable[ImageType]] = None,
        embeddings: Optional[Iterable[torch.Tensor]] = None,
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [N x C' x T] tensor.
        """
        assert (
            images is not None or embeddings is not None
        ), "must specify images or embeddings"
        assert (
            images is None or embeddings is None
        ), "cannot specify both images and embeddings"
        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))

        if images is not None:
            clip_out = self.clip.embed_images_grid(images)
        else:
            clip_out = embeddings

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None, None].to(clip_out)

        clip_out = clip_out.permute(0, 2, 1)  # NCL -> NLC
        clip_embed = self.clip_embed(clip_out)

        cond = [(t_embed, self.time_token_cond), (clip_embed, True)]
        return self._forward_with_cond(x, cond)


class UpsamplePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cond_input_channels: Optional[int] = None,
        cond_ctx: int = 1024,
        n_ctx: int = 4096 - 1024,
        channel_scales: Optional[Sequence[float]] = None,
        channel_biases: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + cond_ctx, **kwargs)
        self.n_ctx = n_ctx
        self.cond_input_channels = cond_input_channels or self.input_channels
        self.cond_point_proj = nn.Linear(
            self.cond_input_channels, self.backbone.width, device=device, dtype=dtype
        )

        self.register_buffer(
            "channel_scales",
            (
                torch.tensor(channel_scales, dtype=dtype, device=device)
                if channel_scales is not None
                else None
            ),
        )
        self.register_buffer(
            "channel_biases",
            (
                torch.tensor(channel_biases, dtype=dtype, device=device)
                if channel_biases is not None
                else None
            ),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, low_res: torch.Tensor):
        """
        :param x: an [N x C1 x T] tensor.
        :param t: an [N] tensor.
        :param low_res: an [N x C2 x T'] tensor of conditioning points.
        :return: an [N x C3 x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        low_res_embed = self._embed_low_res(low_res)
        cond = [(t_embed, self.time_token_cond), (low_res_embed, True)]
        return self._forward_with_cond(x, cond)

    def _embed_low_res(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_scales is not None:
            x = x * self.channel_scales[None, :, None]
        if self.channel_biases is not None:
            x = x + self.channel_biases[None, :, None]
        return self.cond_point_proj(x.permute(0, 2, 1))


class CLIPImageGridUpsamplePointDiffusionTransformer(UpsamplePointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 4096 - 1024,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(
            device,
            cache_dir=cache_dir,
        )
        super().__init__(
            device=device, dtype=dtype, n_ctx=n_ctx + clip.grid_size**2, **kwargs
        )
        self.n_ctx = n_ctx

        self.clip = clip
        self.clip_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.clip.grid_feature_dim,),
                device=device,
                dtype=dtype,
            ),
            nn.Linear(
                self.clip.grid_feature_dim,
                self.backbone.width,
                device=device,
                dtype=dtype,
            ),
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(
        self, batch_size: int, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        if "images" not in model_kwargs:
            zero_emb = torch.zeros(
                [batch_size, self.clip.grid_feature_dim, self.clip.grid_size**2],
                device=next(self.parameters()).device,
            )
            return dict(embeddings=zero_emb, low_res=model_kwargs["low_res"])
        with torch.no_grad():
            return dict(
                embeddings=self.clip.embed_images_grid(model_kwargs["images"]),
                low_res=model_kwargs["low_res"],
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        low_res: torch.Tensor,
        images: Optional[Iterable[ImageType]] = None,
        embeddings: Optional[Iterable[torch.Tensor]] = None,
    ):
        """
        :param x: an [N x C1 x T] tensor.
        :param t: an [N] tensor.
        :param low_res: an [N x C2 x T'] tensor of conditioning points.
        :param images: a batch of images to condition on.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [N x C3 x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        low_res_embed = self._embed_low_res(low_res)

        if images is not None:
            clip_out = self.clip.embed_images_grid(images)
        elif embeddings is not None:
            clip_out = embeddings
        else:
            # Support unconditional generation.
            clip_out = torch.zeros(
                [len(x), self.clip.grid_feature_dim, self.clip.grid_size**2],
                dtype=x.dtype,
                device=x.device,
            )

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None, None].to(clip_out)

        clip_out = clip_out.permute(0, 2, 1)  # NCL -> NLC
        clip_embed = self.clip_embed(clip_out)

        cond = [
            (t_embed, self.time_token_cond),
            (clip_embed, True),
            (low_res_embed, True),
        ]
        return self._forward_with_cond(x, cond)
