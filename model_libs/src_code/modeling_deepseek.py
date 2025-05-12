# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeepSeek model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_deepseek import DeepseekV2Config
import torch.distributed as dist
import numpy as np

from torch.nn.parameter import Parameter
from .grouped_gemm_util import ops as gg_ops

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from functools import partial

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekV2Config"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(DeepseekV2RMSNorm)


class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->DeepseekV2
class DeepseekV2LinearScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->DeepseekV2
class DeepseekV2DynamicNTKScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding):

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class VanillaMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None, no_act=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if no_act:
            self.act_fn = lambda a: a
        else:
            self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)))
        return down_proj
    

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states, topk):
        input_dtype = hidden_states.dtype
        
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.reshape(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=topk, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=topk, dim=-1, sorted=False
            )

        ### norm gate to sum 1
        if topk > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = topk
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        
        topk_weight = topk_weight

        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss



class SlowMoE(nn.Module):
    """
    My Initial Implementation of MoE models. Slow. Now Used mainly for Measure MACs.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size

        self.ep_size = 1
        self.ep_rank = 0
        self.experts_per_rank = config.n_routed_experts

        self.experts = nn.ModuleList(
            [
                VanillaMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for i in range(self.experts_per_rank)
            ]
        )
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = VanillaMLP(
                config=config, intermediate_size=intermediate_size
            )
        self.group_permute = True

    def forward(self, hidden_states):
        shared_exp_output = None
        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            shared_exp_output = self.shared_experts(hidden_states)
    
        output_shape = hidden_states.shape

        topk_idx, topk_weight, aux_loss = self.gate(hidden_states, topk=self.num_experts_per_tok)
        topk_idx = topk_idx.to(torch.int32)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        hidden_states = hidden_states.repeat_interleave(
            self.num_experts_per_tok, dim=0
        )
        y = torch.empty_like(hidden_states)
        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.to(hidden_states.dtype).view(*output_shape)

        if shared_exp_output is not None:
            y = y + shared_exp_output

        return y


class GroupedMoE(nn.Module):
    """
    MoE with Grouped Matrix Multiplication
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size

        self.ep_size = 1
        self.ep_rank = 0
        self.experts_per_rank = config.n_routed_experts

        self.linear_1 = nn.Linear(self.hidden_size, config.moe_intermediate_size * self.experts_per_rank, bias=False)
        self.linear_2 = nn.Linear(config.moe_intermediate_size * self.experts_per_rank, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = VanillaMLP(
                config=config, intermediate_size=intermediate_size
            )
        self.group_permute = True


    def forward(self, hidden_states):
        shared_exp_output = None
        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            shared_exp_output = self.shared_experts(hidden_states)
    
        output_shape = hidden_states.shape

        topk_idx, topk_weight, aux_loss = self.gate(hidden_states, topk=self.num_experts_per_tok)
        topk_idx = topk_idx.to(torch.int32)

        flatten_indices = topk_idx.view(-1)
        if self.group_permute:
            sorted_indices = None
        else:
            sorted_indices = torch.argsort(flatten_indices, stable=True)

        num_local_tokens_per_expert = torch.bincount(flatten_indices, minlength=self.experts_per_rank)
        tokens_per_expert = num_local_tokens_per_expert.to(
            torch.device("cpu")
        )
        ####################################

        w1 = self.linear_1.weight.view(self.experts_per_rank, self.hidden_size, -1)
        w2 = self.linear_2.weight.view(self.experts_per_rank, -1, self.hidden_size)

        topk = topk_weight.size(1)
        num_unpermuted_tokens = topk_weight.numel()

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        assert topk_weight.shape[0] == hidden_states.shape[0] and num_unpermuted_tokens // hidden_states.shape[0] == topk
        if self.group_permute:
            permuted_hidden_states, row_id_map = gg_ops.permute(hidden_states, topk_idx)
        else:
            permuted_hidden_states = hidden_states.index_select(0, sorted_indices // topk)

        ################### calculation
        fc1_output = gg_ops.gmm(
            permuted_hidden_states, w1, tokens_per_expert, trans_b=False
        )
        intermediate_parallel = self.act_fn(fc1_output)
        permuted_tokens = gg_ops.gmm(
            intermediate_parallel, w2, tokens_per_expert, trans_b=False
        )
        ################### unpermute
        if self.group_permute:
            unpermuted_tokens = gg_ops.unpermute(permuted_tokens, row_id_map, topk_weight)
        else:
            unpermuted_tokens = torch.zeros(
                [num_unpermuted_tokens, permuted_tokens.shape[-1]],
                dtype=permuted_tokens.dtype,
                device=permuted_tokens.device,
            )
            unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
            unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
            unpermuted_tokens = unpermuted_tokens * topk_weight.unsqueeze(-1)
            unpermuted_tokens = unpermuted_tokens.sum(dim=1)

        unpermuted_tokens = unpermuted_tokens.view(output_shape)

        ####################################
        if self.training:
            y = AddAuxiliaryLoss.apply(unpermuted_tokens, aux_loss)
        else:
            y = unpermuted_tokens
        
        if shared_exp_output is not None:
            y = y + shared_exp_output

        return y

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class VanillaAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper

    Modified from Deepseek Attention without MLA 
    """

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        # self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * (self.q_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.q_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        ##### get q
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        ##### get kv
        kv = self.kv_proj(hidden_states)
        kv = (
            kv.view(bsz, q_len, self.num_heads, self.q_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        key, value_states = torch.split(
            kv, [self.q_head_dim, self.v_head_dim], dim=-1
        )

        ##### apply rotray
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        ### core attention
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class SlowPreMixMoE(nn.Module):
    """
    Initial and Simple Implementation, based on deepseek moe layer.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        if hasattr(config, "ep_size") and config.ep_size > 1:
            raise Exception("Obviously, deepseek hasn't implemented training when ep_size > 1")
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList(
                [
                    (
                        VanillaMLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        if i >= self.ep_rank * self.experts_per_rank
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0

            self.experts = nn.ModuleList(
                [
                    VanillaMLP(
                        config, intermediate_size=config.moe_intermediate_size
                    )
                    for i in range(self.experts_per_rank)
                ]
            )
            if config.no_act:
                self.act_fn = lambda a: a
            else:
                self.act_fn = ACT2FN[config.hidden_act]

        self.gate = MoEGate(config)
        if config.ffn_separate_router and config.share_att_ffn_moe and config.premix_att_moe:
            self.gate_ffn = MoEGate(config)

        if config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            # For one_head_attention moe, this means each token only has one type of expert input representation, so the shared expert parameters can be concatenated
            if not self.config.multi_share_expert:
                self.shared_experts = VanillaMLP(
                        config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts, no_act=config.no_act
                    )
            # For multi-head attention moe, each shared expert has its own input
            else:
                self.shared_experts = nn.ModuleList(
                    [
                        (
                            VanillaMLP(
                                config, intermediate_size=config.moe_intermediate_size, no_act=config.no_act
                            )
                        )
                        for _ in range(config.n_shared_experts)
                    ]
                )
        
            if config.disable_share_expert_for_ffn:
                self.ffn_shared_experts = None
            elif config.seperate_share_expert_for_ffn:
                assert config.share_att_ffn_moe
                self.ffn_shared_experts = VanillaMLP(
                    config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts, no_act=config.no_act
                )
            else:
                self.ffn_shared_experts = self.shared_experts
            
        self.group_permute = True

    def gating(self, hidden_states, topk, is_share_ffn=False):
        if is_share_ffn:
            topk_idx, topk_weight, aux_loss = self.gate_ffn(hidden_states, topk=topk)
        else:
            topk_idx, topk_weight, aux_loss = self.gate(hidden_states, topk=topk)

        return topk_idx.to(torch.int32), topk_weight, aux_loss

    def moe_calculation(self, hidden_states, tokens_per_expert, sorted_indices, topk_weight, output_shape, topk_idx):
        topk_idx = topk_idx.to(torch.int32)

        if hidden_states.shape[2] != topk_idx.shape[-1]:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

            hidden_states = hidden_states.repeat_interleave(
                topk_idx.shape[-1], dim=0
            )
        else:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        flat_topk_idx = topk_idx.view(-1)
        
        y = torch.empty_like(hidden_states)
        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.to(hidden_states.dtype).view(*output_shape)

        return y

    def share_expert_func(self, input_for_share_expert, attention_or_ffn):
        if attention_or_ffn in ["attention", "OneHeadAttentionMoE"]:
            if not self.config.multi_share_expert:
                y = self.shared_experts(input_for_share_expert)
            else:
                y = None
                for i in range(self.config.n_shared_experts):
                    this_input = input_for_share_expert[:, :, i, :]
                    this_output = self.shared_experts[i](this_input)
                    if y is None:
                        y = this_output
                    else:
                        y = y + this_output
        elif attention_or_ffn == "ffn":
            y = self.ffn_shared_experts(input_for_share_expert)
        else:
            raise Exception("!!!!")

        return y

    def forward_func(self, input_for_share_expert, moe_input, tokens_per_expert, sorted_indices, topk_weight, output_shape, aux_loss, topk_idx, attention_or_ffn):
        if attention_or_ffn in ["attention", "OneHeadAttentionMoE"]:
            disable_share = self.config.disable_share_expert_for_att
        elif attention_or_ffn == "ffn":
            disable_share = self.config.disable_share_expert_for_ffn
        else:
            raise Exception("Impossible!")

        if moe_input is None:
            # experts in attention moe are all shared (fixed) experts，equals to vanilla attention
            assert self.config.n_shared_experts is not None and self.config.n_shared_experts > 0 and not disable_share

            y = self.share_expert_func(input_for_share_expert, attention_or_ffn)
        else:
            unpermuted_tokens = self.moe_calculation(moe_input, tokens_per_expert, sorted_indices, topk_weight, output_shape, topk_idx)

            if self.training:
                y = AddAuxiliaryLoss.apply(unpermuted_tokens, aux_loss)
            else:
                y = unpermuted_tokens

            if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0 and not disable_share:
                if attention_or_ffn == "OneHeadAttentionMoE":
                    bsz, q_len, h = input_for_share_expert.size()
                    y = y.reshape(bsz, -1, q_len, h).sum(dim=1)
                    
                y = y + self.share_expert_func(input_for_share_expert, attention_or_ffn)

        return y

    def forward(self, input_for_share_expert, moe_input, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx, attention_or_ffn):        
        if self.training:
            return self.forward_func(input_for_share_expert, moe_input, tokens_per_expert, sorted_indices, topk_weight, output_shape, aux_loss, topk_idx, attention_or_ffn)
        
        with torch.no_grad():
            return self.forward_func(input_for_share_expert, moe_input, tokens_per_expert, sorted_indices, topk_weight, output_shape, aux_loss, topk_idx, attention_or_ffn)


class SlowPreMixMoEAttention(nn.Module):
    """Initial and Simple Implementation. Please read PreMixMoEAttention."""

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.init_layer_idx = layer_idx
        if layer_idx is None and not self.config.share_layer:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_experts_per_tok

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.v_head_dim = self.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        # We use this arg for attention moe layers
        self.n_routed_experts = self.config.n_routed_experts 

        self.is_causal = True

        # each expert have a q_matrix
        if self.num_heads > 0:
            if self.q_lora_rank is None:
                self.q_proj = nn.ModuleList(
                    [
                        nn.Linear(
                            self.hidden_size, self.q_head_dim, bias=False
                        )
                        for i in range(self.n_routed_experts)
                    ]
                )
            else:
                if self.config.res_query_lora:
                    self.q_res_proj = nn.Linear(
                        self.hidden_size, self.q_head_dim, bias=False
                    )

                self.q_a_proj = nn.ModuleList(
                    [
                        nn.Linear(
                            self.hidden_size, self.q_lora_rank, bias=False
                        )
                        for i in range(self.n_routed_experts)
                    ]
                )
                self.q_b_proj = nn.ModuleList(
                    [
                        nn.Linear(
                            self.q_lora_rank, self.q_head_dim, bias=False
                        )
                        for i in range(self.n_routed_experts)
                    ]
                )
        # used after attention (token mixing)
        self.vo_experts = SlowPreMixMoE(config)

        # one key for one token
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.q_head_dim,
            bias=False,
        )

        # If there are shared (fixed) experts，this q is for shared experts 
        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            if not self.config.multi_share_expert:
                self.shared_q_proj = nn.Linear(
                    self.hidden_size, self.q_head_dim, bias=False
                )
                self.query_num_for_share_expert = 1
            else:
                self.shared_q_proj = nn.Linear(
                    self.hidden_size, self.q_head_dim * self.config.n_shared_experts, bias=False
                )
                self.query_num_for_share_expert = self.config.n_shared_experts
                assert self.query_num_for_share_expert > 1
                
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.group_permute = True

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.q_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def premix(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
        ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()


        output_shape = hidden_states.shape

        # get routing reults; [sq * bsz, topk]
        topk_idx, topk_weight, aux_loss = self.vo_experts.gating(hidden_states, topk=self.num_heads)
        # todo support fast share experts

        topk = topk_idx.shape[-1]
        flatten_indices = topk_idx.view(-1)
        
        ######################### Get Query ####################################
        topk_idx = topk_idx.to(torch.int32)

        temp_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        temp_hidden_states = temp_hidden_states.repeat_interleave(
            self.num_heads, dim=0
        )
        y = torch.empty((*temp_hidden_states.shape[:-1], self.q_head_dim), dtype=temp_hidden_states.dtype, device=temp_hidden_states.device)
        for i in range(self.n_routed_experts):
            if self.q_lora_rank is None:
                y[flat_topk_idx == i] = self.q_proj[i](temp_hidden_states[flat_topk_idx == i])
            else:
                y[flat_topk_idx == i] = self.q_b_proj[i](self.q_a_proj[i](temp_hidden_states[flat_topk_idx == i]))
        q = y.view(bsz, q_len, topk, self.q_head_dim)

        if self.config.res_query_lora and self.q_lora_rank is not None:
            res_q = self.q_res_proj(hidden_states) # (bsz, q_len, q_dim)
            q = q + res_q.unsqueeze(-2)

        tokens_per_expert = None
        sorted_indices = None

        ## add query for shared expert
        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            query_for_shared_exp = self.shared_q_proj(hidden_states).reshape(bsz, q_len, self.query_num_for_share_expert, self.q_head_dim)
            if q is None:
                q = query_for_shared_exp
            else:
                q = torch.cat((query_for_shared_exp, q), dim=2)

            head_num_for_check = self.num_heads + self.query_num_for_share_expert
        else:
            head_num_for_check = self.num_heads

        q = q.transpose(1, 2)

        ######################### Get Key ####################################

        k = self.k_proj(hidden_states)
        k = k.view(bsz, q_len, 1, self.q_head_dim).transpose(1, 2)

        ######################### Apply Rope ####################################
        kv_seq_len = q_len
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(k, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        value_states = hidden_states.unsqueeze(2).transpose(1, 2) # (bsz, 1, kv_seq_len, dim)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        ######################### Get Mixed Hiden States ####################################

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, head_num_for_check, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, head_num_for_check, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        # attn_output = torch.matmul(attn_weights[:, :, :, :256], value_states[:, :, :256, :])
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, head_num_for_check, q_len, self.hidden_size):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, head_num_for_check, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        ######################### VO Expert ####################################

        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            if self.num_heads == 0:
                input_for_share_expert = attn_output
                attn_output = None
            else:
                input_for_share_expert, attn_output = torch.split(
                    attn_output, [self.query_num_for_share_expert, self.num_heads], dim=2
                )
                if self.query_num_for_share_expert == 1:
                    input_for_share_expert = input_for_share_expert.squeeze(2)
                input_for_share_expert = input_for_share_expert.contiguous()
                attn_output = attn_output.contiguous()
        else:
            input_for_share_expert = None
        
        return input_for_share_expert, attn_output, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx, attn_weights, past_key_value

    def prepare_ffn_moe(self, hidden_states, this_k):
        input_for_share_expert = hidden_states
        expert_inputs = hidden_states

        ################ prepare for moe with hidden_states
        output_shape = hidden_states.shape

        topk_idx, topk_weight, aux_loss = self.vo_experts.gating(hidden_states, topk=this_k, is_share_ffn=self.config.ffn_separate_router)

        flatten_indices = topk_idx.view(-1)
        num_local_tokens_per_expert = torch.bincount(flatten_indices, minlength=self.vo_experts.config.n_routed_experts)
        tokens_per_expert = num_local_tokens_per_expert.to(
            torch.device("cpu")
        )

        sorted_indices = None

        return input_for_share_expert, expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        this_k: int = None,
        token_mix: bool = True,
        layer_idx: int = None,
        att_layer_norm = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if layer_idx is None:
            self.layer_idx = self.init_layer_idx
        else:
            self.layer_idx = layer_idx

        """
        This forward has two paths:
        1. Attention-MoE: It first goes through attention to obtain num_experts_per_tok representations for each token, and then sends them to expert computation.
        2. FFN-MoE: It first prepares the MoE parameters, and then directly sends them to expert computation.
        """
        ########### attention ffn moe
        if token_mix:
            input_for_share_expert, expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx, attn_weights, past_key_value = self.premix(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)

            attention_or_ffn = "attention"

            if att_layer_norm is not None:
                assert self.config.norm_after_mix

                if expert_inputs is not None:
                    expert_inputs = att_layer_norm(expert_inputs)
                if input_for_share_expert is not None:
                    input_for_share_expert = att_layer_norm(input_for_share_expert)
                
        ########### prepare ffn moe, only arg:hidden_states & this_k is used
        else:
            input_for_share_expert, expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx = self.prepare_ffn_moe(hidden_states, this_k)

            output_attentions = False

            attention_or_ffn = "ffn"

        expert_outputs = self.vo_experts(input_for_share_expert, expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx, attention_or_ffn)

        if not output_attentions:
            attn_weights = None

        return expert_outputs, attn_weights, past_key_value


class PreMixMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        if hasattr(config, "ep_size") and config.ep_size > 1:
            raise Exception("Obviously, deepseek hasn't implemented training when ep_size > 1")
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList(
                [
                    (
                        VanillaMLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        if i >= self.ep_rank * self.experts_per_rank
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0

            self.linear_1 = nn.Linear(self.hidden_size, config.moe_intermediate_size * self.experts_per_rank, bias=False)
            self.linear_2 = nn.Linear(config.moe_intermediate_size * self.experts_per_rank, self.hidden_size, bias=False)
            if config.postmix_att_moe:
                self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
                
                self.q_lora_rank = config.q_lora_rank
                if self.q_lora_rank is None:
                    self.k_proj = nn.Linear(
                        self.hidden_size, self.experts_per_rank * self.q_head_dim, bias=False
                    )
                else:
                    if self.config.res_query_lora:
                        self.k_res_proj = nn.Linear(
                            self.hidden_size, self.q_head_dim, bias=False
                        )

                    self.k_a_proj = nn.Linear(
                        self.hidden_size, self.experts_per_rank * config.q_lora_rank, bias=False
                    )
                    self.k_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
                    self.k_b_proj = nn.Linear(
                        self.experts_per_rank * config.q_lora_rank, self.q_head_dim, bias=False
                    )
                
            if config.no_act:
                self.act_fn = lambda a: a
            else:
                self.act_fn = ACT2FN[config.hidden_act]

        self.gate = MoEGate(config)
        if config.ffn_separate_router and config.share_att_ffn_moe and config.premix_att_moe:
            self.gate_ffn = MoEGate(config)

        if config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            # For one_head_attention moe, this means each token only has one type of expert input representation, so the shared expert parameters can be concatenated
            if not self.config.multi_share_expert:
                self.shared_experts = VanillaMLP(
                        config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts, no_act=config.no_act
                    )
            # For multi-head attention moe, each shared expert has its own input
            else:
                self.shared_experts = nn.ModuleList(
                    [
                        (
                            VanillaMLP(
                                config, intermediate_size=config.moe_intermediate_size, no_act=config.no_act
                            )
                        )
                        for _ in range(config.n_shared_experts)
                    ]
                )
        
            if config.disable_share_expert_for_ffn:
                self.ffn_shared_experts = None
            elif config.seperate_share_expert_for_ffn:
                assert config.share_att_ffn_moe
                self.ffn_shared_experts = VanillaMLP(
                    config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts, no_act=config.no_act
                )
            else:
                self.ffn_shared_experts = self.shared_experts
            
        self.group_permute = True
        self.attention_name = ""

    def gating(self, hidden_states, topk, is_share_ffn=False):
        if is_share_ffn:
            topk_idx, topk_weight, aux_loss = self.gate_ffn(hidden_states, topk=topk)
        else:
            topk_idx, topk_weight, aux_loss = self.gate(hidden_states, topk=topk)

        return topk_idx.to(torch.int32), topk_weight, aux_loss

    def moe_calculation(self, hidden_states, tokens_per_expert, sorted_indices, topk_weight, output_shape, topk_idx):
        # For some reason, tokens_per_expert gets moved to cuda once it's passed into forward
        tokens_per_expert = tokens_per_expert.to(
            torch.device("cpu"), non_blocking=True
        )

        ################### weight
        w1 = self.linear_1.weight.view(self.experts_per_rank, self.hidden_size, -1)
        w2 = self.linear_2.weight.view(self.experts_per_rank, -1, self.hidden_size)
    
        ################### permute: organizaed by expert
        # [bsz * seq, topk]
        topk = topk_weight.size(1)
        num_unpermuted_tokens = topk_weight.numel()
        
        input_dim = hidden_states.dim()
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        assert topk_weight.shape[0] == hidden_states.shape[0] or num_unpermuted_tokens == hidden_states.shape[0]
        ### Either the input hidden states are [bsz, seq, dim], or [bsz, seq, topk, dim]. The latter (output from attention) only needs reorganizing, while the former also needs to be copied

        if input_dim == 3:
            assert num_unpermuted_tokens // hidden_states.shape[0] == topk
            if self.group_permute:
                permuted_hidden_states, row_id_map = gg_ops.permute(hidden_states, topk_idx)
            else:
                permuted_hidden_states = hidden_states.index_select(0, sorted_indices // topk)
        elif input_dim == 4:
            if self.group_permute:
                permuted_hidden_states, row_id_map = gg_ops.permute(hidden_states, topk_idx.reshape(-1, 1))
            else:
                permuted_hidden_states = hidden_states.index_select(0, sorted_indices)
        else:
            raise Exception("Impossible!!")

        # torch.cuda.current_stream().synchronize()

        ################### calculation
        fc1_output = gg_ops.gmm(
            permuted_hidden_states, w1, tokens_per_expert, trans_b=False
        )
        intermediate_parallel = self.act_fn(fc1_output)
        permuted_tokens = gg_ops.gmm(
            intermediate_parallel, w2, tokens_per_expert, trans_b=False
        )

        if self.attention_name == "post-mixing":
            if self.q_lora_rank is None:
                wk = self.k_proj.weight.view(self.experts_per_rank, self.hidden_size, -1)
                fck_output = gg_ops.gmm(
                    permuted_hidden_states, wk, tokens_per_expert, trans_b=False
                )
            else:
                if self.config.res_query_lora:
                    res_k = self.k_res_proj(hidden_states).reshape(output_shape[:-1] + (self.q_head_dim,)) # (bsz, q_len, q_dim)
                
                w1 = self.k_a_proj.weight.view(self.experts_per_rank, self.hidden_size, -1)
                w2 = self.k_b_proj.weight.view(self.experts_per_rank, -1, self.q_head_dim)
                k_fc1_output = gg_ops.gmm(
                    permuted_hidden_states, w1, tokens_per_expert, trans_b=False
                )
                intermediate_parallel = self.k_a_layernorm(k_fc1_output)
                fck_output = gg_ops.gmm(
                    intermediate_parallel, w2, tokens_per_expert, trans_b=False
                )
                
            permuted_tokens = torch.cat([permuted_tokens, fck_output], dim=-1)

        ################### unpermute
        k_output = None
        
        if self.group_permute:
            if input_dim == 3:
                unpermuted_tokens = gg_ops.unpermute(permuted_tokens, row_id_map, topk_weight)
            else:
                unpermuted_tokens = gg_ops.unpermute(permuted_tokens, row_id_map.reshape(-1, topk).transpose(0, 1).reshape(-1), topk_weight)
        else:
            unpermuted_tokens = torch.zeros(
                [num_unpermuted_tokens, permuted_tokens.shape[-1]],
                dtype=permuted_tokens.dtype,
                device=permuted_tokens.device,
            )
            unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
            unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
            if self.attention_name == 'post-mixing':
                unpermuted_tokens, k_output = torch.split(
                    unpermuted_tokens, [self.hidden_size, self.q_head_dim], dim=-1
                )
                unpermuted_tokens = unpermuted_tokens * topk_weight.unsqueeze(-1)
                output_shape = output_shape[:-1] + (topk, self.hidden_size)
                k_output = k_output.view(output_shape[:-1] + (self.q_head_dim,))
                if self.config.res_query_lora and self.q_lora_rank is not None:
                    k_output = k_output + res_k.unsqueeze(-2)
            else:
                unpermuted_tokens = unpermuted_tokens * topk_weight.unsqueeze(-1)
                unpermuted_tokens = unpermuted_tokens.sum(dim=1)

        unpermuted_tokens = unpermuted_tokens.view(output_shape)

        return unpermuted_tokens, k_output

    def share_expert_func(self, input_for_share_expert, attention_or_ffn):
        if attention_or_ffn in ["attention", "OneHeadAttentionMoE", 'post-mixing']:
            if not self.config.multi_share_expert:
                y = self.shared_experts(input_for_share_expert)
            else:
                y = None
                for i in range(self.config.n_shared_experts):
                    this_input = input_for_share_expert[:, :, i, :]
                    this_output = self.shared_experts[i](this_input)
                    if y is None:
                        y = this_output
                    else:
                        y = y + this_output
        elif attention_or_ffn == "ffn":
            y = self.ffn_shared_experts(input_for_share_expert)
        else:
            raise Exception("!!!!")

        return y

    def forward_func(self, input_for_share_expert, moe_input, tokens_per_expert, sorted_indices, topk_weight, output_shape, aux_loss, topk_idx, attention_or_ffn):
        self.attention_name = attention_or_ffn
        if attention_or_ffn == 'post-mixing':
            self.group_permute = False

        if attention_or_ffn in ["attention", "OneHeadAttentionMoE", "post-mixing"]:
            disable_share = self.config.disable_share_expert_for_att
        elif attention_or_ffn == "ffn":
            disable_share = self.config.disable_share_expert_for_ffn
        else:
            raise Exception("Impossible!")

        if moe_input is None:
            # experts in attention moe are all shared (fixed) experts，equals to vanilla attention
            assert self.config.n_shared_experts is not None and self.config.n_shared_experts > 0 and not disable_share

            y = self.share_expert_func(input_for_share_expert, attention_or_ffn)
        else:
            unpermuted_tokens, k_output = self.moe_calculation(moe_input, tokens_per_expert, sorted_indices, topk_weight, output_shape, topk_idx)

            if self.training:
                y = AddAuxiliaryLoss.apply(unpermuted_tokens, aux_loss)
            else:
                y = unpermuted_tokens

            if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0 and not disable_share:
                if attention_or_ffn == "OneHeadAttentionMoE":
                    bsz, q_len, h = input_for_share_expert.size()
                    y = y.reshape(bsz, -1, q_len, h).sum(dim=1)
                
                if self.attention_name == 'post-mixing':
                    y = torch.cat([y, self.share_expert_func(input_for_share_expert, attention_or_ffn).unsqueeze(-2)], dim=-2) # [bsz, seq, topk + 1, dim]
                else:
                    y = y + self.share_expert_func(input_for_share_expert, attention_or_ffn)

        if self.attention_name == 'post-mixing':
            return y, k_output
        
        return y

    def forward(self, input_for_share_expert, moe_input, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx, attention_or_ffn):        
        if self.training:
            return self.forward_func(input_for_share_expert, moe_input, tokens_per_expert, sorted_indices, topk_weight, output_shape, aux_loss, topk_idx, attention_or_ffn)
        
        with torch.no_grad():
            return self.forward_func(input_for_share_expert, moe_input, tokens_per_expert, sorted_indices, topk_weight, output_shape, aux_loss, topk_idx, attention_or_ffn)


class PostMixMoEAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.init_layer_idx = layer_idx
        if layer_idx is None and not self.config.share_layer:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_experts_per_tok

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.v_head_dim = self.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        # We use this arg for attention moe layers
        self.n_routed_experts = self.config.n_routed_experts 

        self.is_causal = True

        # used after attention (token mixing)
        self.vo_experts = PreMixMoE(config)

        # one q for one token
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.q_head_dim,
            bias=False,
        )
        
        self.k_proj_for_shared_expert = nn.Linear(
            self.hidden_size,
            self.q_head_dim,
            bias=False,
        )

        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.group_permute = False

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.q_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def prepare_ffn_moe(self, hidden_states, this_k):
        input_for_share_expert = hidden_states
        expert_inputs = hidden_states

        ################ prepare for moe with hidden_states
        output_shape = hidden_states.shape

        topk_idx, topk_weight, aux_loss = self.vo_experts.gating(hidden_states, topk=this_k, is_share_ffn=False)

        flatten_indices = topk_idx.view(-1)
        num_local_tokens_per_expert = torch.bincount(flatten_indices, minlength=self.vo_experts.config.n_routed_experts)
        tokens_per_expert = num_local_tokens_per_expert.to(
            torch.device("cpu")
        )

        if self.group_permute:
            sorted_indices = None
        else:
            sorted_indices = torch.argsort(flatten_indices, stable=True)

        return input_for_share_expert, expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        this_k: int = None,
        token_mix: bool = True,
        layer_idx: int = None,
        att_layer_norm = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if layer_idx is None:
            self.layer_idx = self.init_layer_idx
        else:
            self.layer_idx = layer_idx
            
        assert self.num_heads > 0
        bsz, q_len, _ = hidden_states.size()
        
        ######################### independent k expert processing #########################
        input_for_share_expert, expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx = self.prepare_ffn_moe(hidden_states, self.num_heads)
        attention_or_ffn = "post-mixing"
        # [bsz, seq, k + 1, dim], +1 due to shared expert
        expert_outputs, k_output = self.vo_experts(input_for_share_expert, expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx, attention_or_ffn)
        ##################################################
        
        ###################################################### post mixing
        head_num_for_check = 1
        
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, 1, self.q_head_dim).transpose(1, 2) # [bsz, 1, q_len, dim]
        
        shared_key = self.k_proj_for_shared_expert(hidden_states).view(bsz, q_len, 1, self.q_head_dim)
        k_output = torch.cat([k_output, shared_key], dim=-2)
        
        k = k_output.transpose(1, 2) # [bsz, k+1, q_len, dim]
        
        ######################### Apply Rope ####################################
        kv_seq_len = q_len
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(k, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        ######################### Update Cache ####################################
        value_states = expert_outputs.transpose(1, 2)
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        ######################### Get Mixed Hiden States ####################################
        value_states = expert_outputs.reshape(bsz, -1, expert_outputs.shape[-1]).unsqueeze(1) # (bsz, 1, seq * k+1, dim)
        key_states = key_states.transpose(1, 2).reshape(bsz, -1, key_states.shape[-1]).unsqueeze(1)  # (bsz, 1, seq * k+1, dim)
        
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, head_num_for_check, q_len, kv_seq_len * (1 + topk_idx.shape[-1])):
            raise ValueError(
                f"Attention weights should be of size {(bsz, head_num_for_check, q_len, kv_seq_len * (1 + topk_idx.shape[-1]))}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            
            attention_mask = attention_mask.repeat_interleave(
                (1 + topk_idx.shape[-1]), dim=-1
            )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states.to(query_states))

        if attn_output.size() != (bsz, head_num_for_check, q_len, self.hidden_size):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, head_num_for_check, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.squeeze(1).contiguous()

        ######################### VO Expert ####################################

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class PreMixMoEAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.init_layer_idx = layer_idx
        if layer_idx is None and not self.config.share_layer:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_experts_per_tok

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        # self.qk_rope_head_dim = config.qk_rope_head_dim
        # self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = self.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.n_routed_experts = self.config.n_routed_experts 

        self.is_causal = True

        # each expert have a q_matrix
        if self.num_heads > 0:
            if self.q_lora_rank is None:
                self.q_proj = nn.Linear(
                    self.hidden_size, self.n_routed_experts * self.q_head_dim, bias=False
                )
            else:
                if self.config.res_query_lora:
                    self.q_res_proj = nn.Linear(
                        self.hidden_size, self.q_head_dim, bias=False
                    )

                self.q_a_proj = nn.Linear(
                    self.hidden_size, self.n_routed_experts * config.q_lora_rank, bias=False
                )
                self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
                self.q_b_proj = nn.Linear(
                    self.n_routed_experts * config.q_lora_rank, self.q_head_dim, bias=False
                )

        # used after attention (token mixing)
        self.vo_experts = PreMixMoE(config)

        # one key for one token
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.q_head_dim,
            bias=False,
        )

        # If there are shared (fixed) experts，this q is for shared experts 
        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            if not self.config.multi_share_expert:
                self.shared_q_proj = nn.Linear(
                    self.hidden_size, self.q_head_dim, bias=False
                )
                self.query_num_for_share_expert = 1
            else:
                self.shared_q_proj = nn.Linear(
                    self.hidden_size, self.q_head_dim * self.config.n_shared_experts, bias=False
                )
                self.query_num_for_share_expert = self.config.n_shared_experts
                assert self.query_num_for_share_expert > 1
                
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.group_permute = True

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.q_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def premix(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
        ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        # assert position_ids.dim() == 2 and position_ids.shape[0] == bsz and position_ids.shape[1] == q_len, "position id is what?"
        ######################### Get Query ####################################
        if self.num_heads > 0:
            ######################### Get Routing Decision ####################################
            output_shape = hidden_states.shape

            # get routing reults; [sq * bsz, topk]
            topk_idx, topk_weight, aux_loss = self.vo_experts.gating(hidden_states, topk=self.num_heads)
            # todo support fast share experts

            topk = topk_idx.shape[-1]
            flatten_indices = topk_idx.view(-1)

            num_local_tokens_per_expert = torch.bincount(flatten_indices, minlength=self.n_routed_experts)
            tokens_per_expert = num_local_tokens_per_expert.to(
                torch.device("cpu")
            )

            if self.group_permute:
                sorted_indices = None
            else:
                sorted_indices = torch.argsort(flatten_indices, stable=True)
            
            ######################### Permute ####################################
            if self.q_lora_rank is None:
                flatten_hidden_states = hidden_states.view(-1, self.hidden_size)

                if self.group_permute:
                    permuted_hidden_states, row_id_map = gg_ops.permute(flatten_hidden_states, topk_idx)
                else:
                    permuted_hidden_states = flatten_hidden_states.index_select(0, sorted_indices // topk)

                w1 = self.q_proj.weight.view(self.n_routed_experts, self.hidden_size, -1)
                
                permuted_q = gg_ops.gmm(
                    permuted_hidden_states, w1, tokens_per_expert, trans_b=False
                )
                
            else:
                if self.config.res_query_lora:
                    res_q = self.q_res_proj(hidden_states) # (bsz, q_len, q_dim)

                flatten_hidden_states = hidden_states.view(-1, self.hidden_size)

                if self.group_permute:
                    permuted_hidden_states, row_id_map = gg_ops.permute(flatten_hidden_states, topk_idx)
                else:
                    permuted_hidden_states = flatten_hidden_states.index_select(0, sorted_indices // topk)

                w1 = self.q_a_proj.weight.view(self.n_routed_experts, self.hidden_size, -1)
                w2 = self.q_b_proj.weight.view(self.n_routed_experts, -1, self.q_head_dim)
                
                fc1_output = gg_ops.gmm(
                    permuted_hidden_states, w1, tokens_per_expert, trans_b=False
                )
                intermediate_parallel = self.q_a_layernorm(fc1_output)
                permuted_q = gg_ops.gmm(
                    intermediate_parallel, w2, tokens_per_expert, trans_b=False
                )

            ######################### UnPermute ####################################
            if self.group_permute:
                unpermuted_tokens = gg_ops.unpermute(permuted_q, row_id_map)
                unpermuted_tokens = unpermuted_tokens.reshape(topk, -1, permuted_q.size(-1))
                unpermuted_tokens = unpermuted_tokens.transpose(0, 1)
            else:
                num_unpermuted_tokens = permuted_q.size(0)
                unpermuted_tokens = torch.zeros(
                    [num_unpermuted_tokens, permuted_q.shape[-1]],
                    dtype=permuted_q.dtype,
                    device=permuted_q.device,
                )
                unpermuted_tokens.index_copy_(0, sorted_indices, permuted_q)
                unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_q.size(-1))

            q = unpermuted_tokens.view(bsz, q_len, topk, self.q_head_dim)

            if self.config.res_query_lora and self.q_lora_rank is not None:
                q = q + res_q.unsqueeze(-2)

        else:
            # If I don't use MoE, I must maintain shared expert
            assert self.config.n_shared_experts is not None and self.config.n_shared_experts > 0
            assert self.num_heads == 0
            q = None
            tokens_per_expert = None
            sorted_indices = None
            topk_weight = None
            aux_loss = None
            output_shape = None
            topk_idx = None

        ## add query for shared expert
        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            query_for_shared_exp = self.shared_q_proj(hidden_states).reshape(bsz, q_len, self.query_num_for_share_expert, self.q_head_dim)
            if q is None:
                q = query_for_shared_exp
            else:
                q = torch.cat((query_for_shared_exp, q), dim=2)

            head_num_for_check = self.num_heads + self.query_num_for_share_expert
        else:
            head_num_for_check = self.num_heads

        q = q.transpose(1, 2)

        ######################### Get Key ####################################

        k = self.k_proj(hidden_states)
        k = k.view(bsz, q_len, 1, self.q_head_dim).transpose(1, 2)

        ######################### Apply Rope ####################################
        kv_seq_len = q_len
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(k, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        value_states = hidden_states.unsqueeze(2).transpose(1, 2) # (bsz, 1, kv_seq_len, dim)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        ######################### Get Mixed Hiden States ####################################

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, head_num_for_check, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, head_num_for_check, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, head_num_for_check, q_len, self.hidden_size):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, head_num_for_check, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        ######################### VO Expert ####################################

        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            if self.num_heads == 0:
                input_for_share_expert = attn_output
                attn_output = None
            else:
                input_for_share_expert, attn_output = torch.split(
                    attn_output, [self.query_num_for_share_expert, self.num_heads], dim=2
                )
                if self.query_num_for_share_expert == 1:
                    input_for_share_expert = input_for_share_expert.squeeze(2)
                input_for_share_expert = input_for_share_expert.contiguous()
                attn_output = attn_output.contiguous()
        else:
            input_for_share_expert = None
        
        return input_for_share_expert, attn_output, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx, attn_weights, past_key_value

    def prepare_ffn_moe(self, hidden_states, this_k):
        input_for_share_expert = hidden_states
        expert_inputs = hidden_states

        ################ prepare for moe with hidden_states
        output_shape = hidden_states.shape

        topk_idx, topk_weight, aux_loss = self.vo_experts.gating(hidden_states, topk=this_k, is_share_ffn=self.config.ffn_separate_router)

        flatten_indices = topk_idx.view(-1)
        num_local_tokens_per_expert = torch.bincount(flatten_indices, minlength=self.vo_experts.config.n_routed_experts)
        tokens_per_expert = num_local_tokens_per_expert.to(
            torch.device("cpu")
        )

        sorted_indices = None

        return input_for_share_expert, expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        this_k: int = None,
        token_mix: bool = True,
        layer_idx: int = None,
        att_layer_norm = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if layer_idx is None:
            self.layer_idx = self.init_layer_idx
        else:
            self.layer_idx = layer_idx

        """
        This forward has two paths:
        1. Attention-MoE: It first goes through attention to obtain num_experts_per_tok representations for each token, and then sends them to expert computation.
        2. FFN-MoE: It first prepares the MoE parameters, and then directly sends them to expert computation.
        """
        ########### attention ffn moe
        if token_mix:
            input_for_share_expert, expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx, attn_weights, past_key_value = self.premix(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)

            attention_or_ffn = "attention"

            if att_layer_norm is not None:
                assert self.config.norm_after_mix

                if expert_inputs is not None:
                    expert_inputs = att_layer_norm(expert_inputs)
                if input_for_share_expert is not None:
                    input_for_share_expert = att_layer_norm(input_for_share_expert)
                
        ########### prepare ffn moe, only arg:hidden_states & this_k is used
        else:
            input_for_share_expert, expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx = self.prepare_ffn_moe(hidden_states, this_k)

            output_attentions = False

            attention_or_ffn = "ffn"

        expert_outputs = self.vo_experts(input_for_share_expert, expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx, attention_or_ffn)

        if not output_attentions:
            attn_weights = None

        return expert_outputs, attn_weights, past_key_value


def shared_att_moe_for_ffn(moe_att_module: PreMixMoEAttention, this_topk: int, skip_flag: bool, inputs):
    """
    During ablation studies, I experimented to determine which layer - attention or ffn - should use more experts. 
    The final result was that ffn was skipped, and all computational resources were given to attention. In this case, skip_flag needs to be set to True.
    """
    if skip_flag:
        return inputs
    
    expert_outputs, _, _ = moe_att_module(inputs, token_mix=False, this_k=this_topk)
    return expert_outputs

class UMoEDecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config

        ############## get attention
        if config.one_head_attention_moe:
            assert config.premix_att_moe
            
            self.self_attn = OneHeadAttentionMoE(config=config,layer_idx=layer_idx)
        elif config.mix_then_routing_unify:
            assert config.premix_att_moe

            self.self_attn = MixThenRoutingMoE(config=config,layer_idx=layer_idx)
        elif config.premix_att_moe:
            assert not config.baseline_moa

            self.self_attn = PreMixMoEAttention(config=config,layer_idx=layer_idx)
        elif config.baseline_moa:
            assert not config.share_att_ffn_moe

            self.self_attn = MoAMoEAttention(config=config,layer_idx=layer_idx)
        elif config.baseline_switchhead:
            assert not config.share_att_ffn_moe
            
            self.self_attn = SwitchHead(config=config,layer_idx=layer_idx)
        else:
            self.self_attn = VanillaAttention(config=config,layer_idx=layer_idx)

        ############## get ffn
        if config.premix_att_moe and config.share_att_ffn_moe:
            assert config.ffn_num_experts_per_tok > 0 or (config.ffn_num_experts_per_tok == 0 and config.skip_shared_ffn)
            self.mlp = partial(shared_att_moe_for_ffn, self.self_attn, config.ffn_num_experts_per_tok, config.skip_shared_ffn) 
        elif config.mix_then_routing_unify:
            self.mlp = lambda x: 0
        else:
            self.mlp = (
                GroupedMoE(config)
                if (
                    config.n_routed_experts is not None
                    and layer_idx >= config.first_k_dense_replace
                    and layer_idx % config.moe_layer_freq == 0
                    and not config.no_ffn_moe
                )
                else VanillaMLP(config)
            )
        
        ########## get layernorm. Previous work suggest that if we share parameters across layer, it's better to maintain distinct normalizations
        self.input_layernorm = nn.ModuleList(
                [
                    DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                    for _ in range(self.config.share_layer_repeat_num)
                ]
        ) if (self.config.share_layer and False) else DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_attention_layernorm = nn.ModuleList(
                [
                    DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if not config.mix_then_routing_unify else lambda x: 0
                    for _ in range(self.config.share_layer_repeat_num)
                ]
        )  if (self.config.share_layer and False) else (DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if not config.mix_then_routing_unify else lambda x: 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        layer_idx: Optional[int] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if self.config.share_layer:
            assert layer_idx is not None
            this_layer_offset = layer_idx % self.config.share_layer_repeat_num
            # this_input_layernorm = self.input_layernorm[this_layer_offset]
            # this_post_attention_layernorm = self.post_attention_layernorm[this_layer_offset]
            this_input_layernorm = self.input_layernorm
            this_post_attention_layernorm = self.post_attention_layernorm
        else:
            this_input_layernorm = self.input_layernorm
            this_post_attention_layernorm = self.post_attention_layernorm
            
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        if not self.config.norm_after_mix:
            hidden_states = this_input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            layer_idx=layer_idx,
            layer_norm=this_input_layernorm if self.config.norm_after_mix else None,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = this_post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


DeepseekV2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DeepseekV2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare DeepseekV2 Model outputting raw hidden-states without any specific head on top.",
    DeepseekV2_START_DOCSTRING,
)
class DeepseekV2PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UMoEDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


DeepseekV2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare DeepseekV2 Model outputting raw hidden-states without any specific head on top.",
    DeepseekV2_START_DOCSTRING,
)
class DeepseekV2Model(DeepseekV2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`UMoEDecoderLayer`]

    Args:
        config: DeepseekV2Config
    """

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.config = config

        if not config.share_layer:
            self.layers = nn.ModuleList(
                [
                    UMoEDecoderLayer(config, layer_idx)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
        else:
            assert config.num_hidden_layers % config.share_layer_repeat_num == 0

            self.layers = nn.ModuleList(
                [UMoEDecoderLayer(config, layer_idx=None) for _ in range(config.num_hidden_layers // config.share_layer_repeat_num)]
            )

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        assert not self._use_flash_attention_2, f"Are you sure use flash attention?"
        self.norm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx in range(self.config.num_hidden_layers):            
            if self.config.share_layer:
                decoder_layer = self.layers[layer_idx // self.config.share_layer_repeat_num]
            else:
                decoder_layer = self.layers[layer_idx]
                layer_idx = None

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    layer_idx,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    layer_idx=layer_idx,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DeepseekV2ForCausalLM(DeepseekV2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        calculate_loss_without_label = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV2ForCausalLM

        >>> model = DeepseekV2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        elif calculate_loss_without_label:
            loss_fct = CrossEntropyLoss(reduction="none")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention_mask_batch = attention_mask[..., 1:].contiguous()
            
            loss = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1) / shift_attention_mask_batch.sum(1)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


@add_start_docstrings(
    """
    The DeepseekV2 Model transformer with a sequence classification head on top (linear layer).

    [`DeepseekV2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    DeepseekV2_START_DOCSTRING,
)
class DeepseekV2ForSequenceClassification(DeepseekV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = DeepseekV2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, transformers.,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


#################### Following is my implementation of MoA
#################### Following is my implementation of MoA
#################### Following is my implementation of MoA
#################### Following is my implementation of MoA
#################### Following is my implementation of MoA
#################### Following is my implementation of MoA


class SlowMoAMoEAttention(nn.Module):
    """MoE attention from 'Mixture of Attention Heads: Selecting Attention Heads Per Token"""

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.init_layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_experts_per_tok

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # v_head_dim
        self.v_head_dim = config.v_head_dim

        # q(k)_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.n_routed_experts = self.config.n_routed_experts 

        self.is_causal = True

        # each expert have a q_matrix
        self.q_proj = nn.ModuleList(
            [
                nn.Linear(
                    self.hidden_size, self.q_head_dim, bias=False
                )
                for i in range(self.n_routed_experts)
            ]
        )
        

        self.v_proj = nn.Linear(
            self.hidden_size, self.v_head_dim, bias=False
        )

        # used after attention (token mixing)
        self.o_proj = nn.ModuleList(
            [
                nn.Linear(
                    self.v_head_dim, self.hidden_size, bias=False
                )
                for i in range(self.n_routed_experts)
            ]
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.q_head_dim,
            bias=False,
        )

        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.group_permute = True

        self.gate = MoEGate(config)

        if config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            assert self.config.n_shared_experts == 1, "Only 1 shared experts is supported for MoA!!"
            self.shared_experts = nn.Linear(
                self.v_head_dim, self.hidden_size, bias=False
            )

            self.shared_q_proj = nn.Linear(
                self.hidden_size, self.q_head_dim, bias=False
            )


    def gating(self, hidden_states, topk):
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states, topk=topk)

        return topk_idx.to(torch.int32), topk_weight, aux_loss

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.q_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        this_k: int = None,
        token_mix: bool = True,
        layer_idx: int = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if layer_idx is None:
            self.layer_idx = self.init_layer_idx
        else:
            raise Exception("unexpected!")
            self.layer_idx = layer_idx

        bsz, q_len, _ = hidden_states.size()

        # assert position_ids.dim() == 2 and position_ids.shape[0] == bsz and position_ids.shape[1] == q_len, "position id is what?"
        ######################### Permute for Query ####################################
        output_shape = hidden_states.shape

        # get routing reults; [sq * bsz, topk]
        topk_idx, topk_weight, aux_loss = self.gating(hidden_states, topk=self.num_heads)

        topk = topk_idx.shape[-1]
        
        ####################################################################################################
        flat_topk_idx = topk_idx.view(-1)
        temp_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        temp_hidden_states = temp_hidden_states.repeat_interleave(
            self.num_heads, dim=0
        )
        y = torch.empty(temp_hidden_states.shape[:-1] + (self.q_head_dim,), dtype=hidden_states.dtype, device=hidden_states.device)
        for i, expert in enumerate(self.q_proj):
            y[flat_topk_idx == i] = expert(temp_hidden_states[flat_topk_idx == i])
        q = y.to(hidden_states.dtype).reshape(bsz, q_len, topk, self.q_head_dim)
        
        ####################################################################################################
        
        head_num_for_check = topk

        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            shared_q = self.shared_q_proj(hidden_states).unsqueeze(-2)
            q = torch.cat((shared_q, q), dim=2)
            head_num_for_check += 1

        q = q.transpose(1, 2)

        ######################### Get Key ####################################

        k = self.k_proj(hidden_states)
        k = k.view(bsz, q_len, 1, self.q_head_dim).transpose(1, 2)

        ######################### Apply Rope ####################################
        kv_seq_len = q_len
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(k, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        ######################### get value ####################################
        value_states = self.v_proj(hidden_states).unsqueeze(2).transpose(1, 2) # (bsz, 1, kv_seq_len, v_dim)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        ######################### Get Mixed Hiden States ####################################

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, head_num_for_check, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, head_num_for_check, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, head_num_for_check, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, head_num_for_check, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:

            input_for_share_expert, attn_output = torch.split(
                attn_output, [1, self.num_heads], dim=2
            )
            input_for_share_expert = input_for_share_expert.squeeze(2)
            input_for_share_expert = input_for_share_expert.contiguous()
            attn_output = attn_output.contiguous()
        else:
            input_for_share_expert = None
        
        ######################### O Expert ####################################
        attn_output = attn_output.reshape(-1, attn_output.shape[-1])
        y = torch.empty(attn_output.shape[:-1] + (self.hidden_size,), dtype=hidden_states.dtype, device=hidden_states.device)
        for i, expert in enumerate(self.o_proj):
            y[flat_topk_idx == i] = expert(attn_output[flat_topk_idx == i])
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        unpermuted_tokens = y.to(hidden_states.dtype).view(output_shape)

        ### apply aux loss
        if self.training:
            expert_outputs = AddAuxiliaryLoss.apply(unpermuted_tokens, aux_loss)
        else:
            expert_outputs = unpermuted_tokens
        
        if not output_attentions:
            attn_weights = None

        if input_for_share_expert is not None:
            expert_outputs = expert_outputs + self.shared_experts(input_for_share_expert)

        return expert_outputs, attn_weights, past_key_value



class MoAMoEAttention(nn.Module):
    """MoE attention from 'Mixture of Attention Heads: Selecting Attention Heads Per Token"""

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.init_layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_experts_per_tok

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # v_head_dim
        self.v_head_dim = config.v_head_dim

        # q(k)_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.n_routed_experts = self.config.n_routed_experts 

        self.is_causal = True

        # each expert have a q_matrix
        self.q_proj = nn.Linear(
            self.hidden_size, self.n_routed_experts * self.q_head_dim, bias=False
        )

        self.v_proj = nn.Linear(
            self.hidden_size, self.v_head_dim, bias=False
        )

        # used after attention (token mixing)
        self.o_proj = nn.Linear(
            self.hidden_size, self.n_routed_experts * self.v_head_dim, bias=False
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.q_head_dim,
            bias=False,
        )

        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.group_permute = True

        self.gate = MoEGate(config)

        if config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            assert self.config.n_shared_experts == 1, "Only 1 shared experts is supported for MoA!!"
            self.shared_experts = nn.Linear(
                self.v_head_dim, self.hidden_size, bias=False
            )

            self.shared_q_proj = nn.Linear(
                self.hidden_size, self.q_head_dim, bias=False
            )


    def gating(self, hidden_states, topk):
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states, topk=topk)

        return topk_idx.to(torch.int32), topk_weight, aux_loss

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.q_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        this_k: int = None,
        token_mix: bool = True,
        layer_idx: int = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if layer_idx is None:
            self.layer_idx = self.init_layer_idx
        else:
            raise Exception("unexpected!")
            self.layer_idx = layer_idx

        bsz, q_len, _ = hidden_states.size()

        # assert position_ids.dim() == 2 and position_ids.shape[0] == bsz and position_ids.shape[1] == q_len, "position id is what?"
        ######################### Permute for Query ####################################
        output_shape = hidden_states.shape

        # get routing reults; [sq * bsz, topk]
        topk_idx, topk_weight, aux_loss = self.gating(hidden_states, topk=self.num_heads)

        topk = topk_idx.shape[-1]
        flatten_indices = topk_idx.view(-1)

        num_local_tokens_per_expert = torch.bincount(flatten_indices, minlength=self.n_routed_experts)
        tokens_per_expert = num_local_tokens_per_expert.to(
            torch.device("cpu")
        )

        flatten_hidden_states = hidden_states.view(-1, output_shape[-1])

        if self.group_permute:
            sorted_indices = None
            permuted_hidden_states, row_id_map = gg_ops.permute(flatten_hidden_states, topk_idx)
        else:
            sorted_indices = torch.argsort(flatten_indices, stable=True)
            permuted_hidden_states = flatten_hidden_states.index_select(0, sorted_indices // topk)

        ######################### Get Query ####################################
        w1 = self.q_proj.weight.view(self.n_routed_experts, self.hidden_size, -1)
        
        permuted_q = gg_ops.gmm(
            permuted_hidden_states, w1, tokens_per_expert, trans_b=False
        )
            
        ######################### UnPermute ####################################
        if self.group_permute:
            unpermuted_tokens = gg_ops.unpermute(permuted_q, row_id_map)
            unpermuted_tokens = unpermuted_tokens.reshape(topk, -1, permuted_q.size(-1))
            unpermuted_tokens = unpermuted_tokens.transpose(0, 1)
        else:
            num_unpermuted_tokens = permuted_q.size(0)
            unpermuted_tokens = torch.zeros(
                [num_unpermuted_tokens, permuted_q.shape[-1]],
                dtype=permuted_q.dtype,
                device=permuted_q.device,
            )
            unpermuted_tokens.index_copy_(0, sorted_indices, permuted_q)
            unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_q.size(-1))

        q = unpermuted_tokens.view(bsz, q_len, topk, self.q_head_dim)
        head_num_for_check = topk

        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            shared_q = self.shared_q_proj(hidden_states).unsqueeze(-2)
            q = torch.cat((shared_q, q), dim=2)
            head_num_for_check += 1

        q = q.transpose(1, 2)

        ######################### Get Key ####################################

        k = self.k_proj(hidden_states)
        k = k.view(bsz, q_len, 1, self.q_head_dim).transpose(1, 2)

        ######################### Apply Rope ####################################
        kv_seq_len = q_len
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(k, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        ######################### get value ####################################
        value_states = self.v_proj(hidden_states).unsqueeze(2).transpose(1, 2) # (bsz, 1, kv_seq_len, v_dim)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        ######################### Get Mixed Hiden States ####################################

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, head_num_for_check, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, head_num_for_check, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, head_num_for_check, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, head_num_for_check, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:

            input_for_share_expert, attn_output = torch.split(
                attn_output, [1, self.num_heads], dim=2
            )
            input_for_share_expert = input_for_share_expert.squeeze(2)
            input_for_share_expert = input_for_share_expert.contiguous()
            attn_output = attn_output.contiguous()
        else:
            input_for_share_expert = None
        
        ######################### O Expert ####################################

        w1 = self.o_proj.weight.view(self.n_routed_experts, self.v_head_dim, -1)
    
        ################### permute: organizaed by expert
        # [bsz * seq, topk]
        num_unpermuted_tokens = topk_weight.numel()
        
        hidden_states = attn_output.reshape(-1, attn_output.shape[-1])

        assert num_unpermuted_tokens == hidden_states.shape[0]

        if self.group_permute:
            permuted_hidden_states, row_id_map = gg_ops.permute(hidden_states, topk_idx.reshape(-1, 1))
        else:
            permuted_hidden_states = hidden_states.index_select(0, sorted_indices)

        ################### calculation
        permuted_tokens = gg_ops.gmm(
            permuted_hidden_states, w1, tokens_per_expert, trans_b=False
        )
        ################### unpermute
        if self.group_permute:
            unpermuted_tokens = gg_ops.unpermute(permuted_tokens, row_id_map.reshape(-1, topk).transpose(0, 1).reshape(-1), topk_weight)
        else:
            unpermuted_tokens = torch.zeros(
                [num_unpermuted_tokens, permuted_tokens.shape[-1]],
                dtype=permuted_tokens.dtype,
                device=permuted_tokens.device,
            )
            unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
            unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
            unpermuted_tokens = unpermuted_tokens * topk_weight.unsqueeze(-1)
            unpermuted_tokens = unpermuted_tokens.sum(dim=1)

        unpermuted_tokens = unpermuted_tokens.view(output_shape)

        ### apply aux loss
        if self.training:
            expert_outputs = AddAuxiliaryLoss.apply(unpermuted_tokens, aux_loss)
        else:
            expert_outputs = unpermuted_tokens
        
        if not output_attentions:
            attn_weights = None

        if input_for_share_expert is not None:
            expert_outputs = expert_outputs + self.shared_experts(input_for_share_expert)

        return expert_outputs, attn_weights, past_key_value


class GroupedMatrixMoE(nn.Module):

    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        self.ep_size = 1
        self.ep_rank = 0
        self.experts_per_rank = config.n_routed_experts
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(input_dim, output_dim * self.experts_per_rank, bias=False)

        self.gate = MoEGate(config)
        self.group_permute = True


    def forward(self, hidden_states, gating_input):
        assert gating_input.shape[:-1] == hidden_states.shape[:-1]
        output_shape = hidden_states.shape

        topk_idx, topk_weight, aux_loss = self.gate(gating_input, topk=self.num_experts_per_tok)
        topk_idx = topk_idx.to(torch.int32)

        flatten_indices = topk_idx.view(-1)
        if self.group_permute:
            sorted_indices = None
        else:
            sorted_indices = torch.argsort(flatten_indices, stable=True)

        num_local_tokens_per_expert = torch.bincount(flatten_indices, minlength=self.experts_per_rank)
        tokens_per_expert = num_local_tokens_per_expert.to(
            torch.device("cpu")
        )
        ####################################

        w1 = self.linear.weight.view(self.experts_per_rank, self.input_dim, -1)

        topk = topk_weight.size(1)
        num_unpermuted_tokens = topk_weight.numel()

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        assert topk_weight.shape[0] == hidden_states.shape[0] and num_unpermuted_tokens // hidden_states.shape[0] == topk
        if self.group_permute:
            permuted_hidden_states, row_id_map = gg_ops.permute(hidden_states, topk_idx)
        else:
            permuted_hidden_states = hidden_states.index_select(0, sorted_indices // topk)

        ################### calculation
        permuted_tokens = gg_ops.gmm(
            permuted_hidden_states, w1, tokens_per_expert, trans_b=False
        )
        ################### unpermute
        if self.group_permute:
            unpermuted_tokens = gg_ops.unpermute(permuted_tokens, row_id_map, topk_weight)
        else:
            unpermuted_tokens = torch.zeros(
                [num_unpermuted_tokens, permuted_tokens.shape[-1]],
                dtype=permuted_tokens.dtype,
                device=permuted_tokens.device,
            )
            unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
            unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
            unpermuted_tokens = unpermuted_tokens * topk_weight.unsqueeze(-1)
            unpermuted_tokens = unpermuted_tokens.sum(dim=1)

        output_shape = output_shape[:-1] + (self.output_dim,)
        unpermuted_tokens = unpermuted_tokens.view(output_shape)

        ####################################
        if self.training:
            y = AddAuxiliaryLoss.apply(unpermuted_tokens, aux_loss)
        else:
            y = unpermuted_tokens
        
        return y



class SlowGroupedMatrixMoE(nn.Module):

    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        self.ep_size = 1
        self.ep_rank = 0
        self.experts_per_rank = config.n_routed_experts
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.experts = nn.ModuleList(
            [
                nn.Linear(
                    input_dim, output_dim, bias=False
                )
                for i in range(self.experts_per_rank)
            ]
        )
        
        self.gate = MoEGate(config)
        self.group_permute = True


    def forward(self, hidden_states, gating_input):
        assert gating_input.shape[:-1] == hidden_states.shape[:-1]
        output_shape = hidden_states.shape

        topk_idx, topk_weight, aux_loss = self.gate(gating_input, topk=self.num_experts_per_tok)
        topk_idx = topk_idx.to(torch.int32)

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        hidden_states = hidden_states.repeat_interleave(
            self.num_experts_per_tok, dim=0
        )
        y = torch.empty(hidden_states.shape[:-1] + (self.output_dim,), dtype=hidden_states.dtype, device=hidden_states.device)
        for i, expert in enumerate(self.experts):
            y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        output_shape = output_shape[:-1] + (self.output_dim,)
        y = y.to(hidden_states.dtype).reshape(*output_shape)

        return y


class SwitchHead(nn.Module):

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.init_layer_idx = layer_idx
        if layer_idx is None and not self.config.share_layer:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
            
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.n_routed_experts = self.config.n_routed_experts 

        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size, self.q_head_dim * self.num_heads, bias=False
        )

        # used after attention (token mixing)
        self.v_experts = nn.ModuleList([GroupedMatrixMoE(config, input_dim=self.hidden_size, output_dim=self.v_head_dim) for _ in range(self.num_heads)])
        self.o_experts = nn.ModuleList([GroupedMatrixMoE(config, input_dim=self.v_head_dim, output_dim=self.hidden_size) for _ in range(self.num_heads)])

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.q_head_dim * self.num_heads,
            bias=False,
        )

        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.group_permute = True

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.q_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        layer_idx: int = None,
        **kwargs,
        ):
        if layer_idx is None:
            self.layer_idx = self.init_layer_idx
        else:
            raise Exception("unexpected!")
            self.layer_idx = layer_idx
            
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        ######################### Get Query ####################################
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        head_num_for_check = self.num_heads

        ######################### Get Key ####################################

        k = self.k_proj(hidden_states)
        k = k.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        ######################### Apply Rope ####################################
        kv_seq_len = q_len
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(k, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        ################## get value by moe for each heads
        value_states = torch.cat([this_v_experts(hidden_states, gating_input=hidden_states).unsqueeze(1) for this_v_experts in self.v_experts], dim=1) # (bsz, head_num, kv_seq_len, v_head_dim)
        ##################
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        ######################### Get Mixed Hiden States ####################################

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, head_num_for_check, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, head_num_for_check, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, head_num_for_check, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, head_num_for_check, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        final_output = torch.stack([self.o_experts[idx](attn_output[:, idx, :, :], gating_input=hidden_states) for idx in range(head_num_for_check)]).sum(dim=0)
        
        return final_output, attn_weights, past_key_value
    
    
class OneHeadAttentionMoE(nn.Module):

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.init_layer_idx = layer_idx
        if layer_idx is None and not self.config.share_layer:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.v_head_dim = self.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.n_routed_experts = self.config.n_routed_experts 

        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size, self.q_head_dim * self.num_heads, bias=False
        )

        # used after attention (token mixing)
        self.vo_experts = PreMixMoE(config)

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.q_head_dim * self.num_heads,
            bias=False,
        )

        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.group_permute = True

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.q_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def multi_head_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
        ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        ######################### Get Query ####################################
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        head_num_for_check = self.num_heads

        ######################### Get Key ####################################

        k = self.k_proj(hidden_states)
        k = k.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        ######################### Apply Rope ####################################
        kv_seq_len = q_len
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(k, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        value_states = hidden_states.unsqueeze(2).transpose(1, 2) # (bsz, 1, kv_seq_len, dim)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        ######################### Get Mixed Hiden States ####################################

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, head_num_for_check, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, head_num_for_check, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, head_num_for_check, q_len, self.hidden_size):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, head_num_for_check, q_len, self.hidden_size)}, but is"
                f" {attn_output.size()}"
            )

        return attn_output, attn_weights, past_key_value

    def prepare_ffn_moe(self, hidden_states, this_k, ffn_input=True):
        expert_inputs = hidden_states

        ################ prepare for moe with hidden_states
        output_shape = hidden_states.shape

        topk_idx, topk_weight, aux_loss = self.vo_experts.gating(hidden_states, topk=this_k, is_share_ffn=self.config.ffn_separate_router and ffn_input)

        flatten_indices = topk_idx.view(-1)
        num_local_tokens_per_expert = torch.bincount(flatten_indices, minlength=self.vo_experts.config.n_routed_experts)
        tokens_per_expert = num_local_tokens_per_expert.to(
            torch.device("cpu")
        )

        sorted_indices = None

        return expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        this_k: int = None,
        token_mix: bool = True,
        layer_idx: int = None,
        att_layer_norm = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if layer_idx is None:
            self.layer_idx = self.init_layer_idx
        else:
            self.layer_idx = layer_idx

        """
        This forward pass has two paths:
        0. FFN-MoE: It first prepares the MoE parameters, then directly sends them to expert computation.
        1. Attention-MoE: It first goes through attention to obtain one representation for each token, effectively replacing the original representation, then prepares the MoE parameters, and finally sends them to expert computation.
        """
        ########### attention ffn moe
        if token_mix:
            token_for_moe, attn_weights, past_key_value = self.multi_head_attention(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
            this_k = self.config.num_experts_per_tok
            ffn_input = False

            if att_layer_norm is not None:
                assert self.config.norm_after_mix
                token_for_moe = att_layer_norm(token_for_moe)

            bsz, head_num, q_len, h = token_for_moe.size()
            
            if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:
                if head_num == 1:
                    input_for_share_expert = token_for_moe.squeeze(1)
                else:
                    input_for_share_expert, token_for_moe = torch.split(
                        token_for_moe, [1, head_num-1], dim=1
                    )
                    input_for_share_expert = input_for_share_expert.squeeze(1)
            else:
                input_for_share_expert = None
            
            token_for_moe = token_for_moe.reshape(bsz, -1, h)

            attention_or_ffn = "OneHeadAttentionMoE"
        else:
            assert this_k is not None

            token_for_moe = hidden_states
            output_attentions = False
            ffn_input = True

            input_for_share_expert = hidden_states

            attention_or_ffn = "ffn"

        expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx = self.prepare_ffn_moe(token_for_moe, this_k, ffn_input=ffn_input)

        expert_outputs = self.vo_experts(input_for_share_expert, expert_inputs, tokens_per_expert, sorted_indices, topk_weight, aux_loss, output_shape, topk_idx, attention_or_ffn)

        if token_mix:
            expert_outputs = expert_outputs.reshape(bsz, -1, q_len, h).sum(dim=1)
            
        if not output_attentions:
            attn_weights = None

        return expert_outputs, attn_weights, past_key_value



class MixThenRoutingMoE(nn.Module):

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.init_layer_idx = layer_idx
        if layer_idx is None and not self.config.share_layer:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
            
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.v_head_dim = self.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.n_routed_experts = self.config.n_routed_experts 

        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size, self.q_head_dim * self.num_heads, bias=False
        )

        # used after attention (token mixing)
        self.vo_experts = GroupedMoE(config)

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.q_head_dim * self.num_heads,
            bias=False,
        )

        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.group_permute = True

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.q_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.q_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def multi_head_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
        ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        ######################### Get Query ####################################
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        head_num_for_check = self.num_heads

        ######################### Get Key ####################################

        k = self.k_proj(hidden_states)
        k = k.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        ######################### Apply Rope ####################################
        kv_seq_len = q_len
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(k, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        value_states = hidden_states.unsqueeze(2).transpose(1, 2) # (bsz, 1, kv_seq_len, dim)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        ######################### Get Mixed Hiden States ####################################

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, head_num_for_check, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, head_num_for_check, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, head_num_for_check, q_len, self.hidden_size):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, head_num_for_check, q_len, self.hidden_size)}, but is"
                f" {attn_output.size()}"
            )

        return attn_output, attn_weights, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        this_k: int = None,
        token_mix: bool = True,
        layer_idx: int = None,
        att_layer_norm = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if layer_idx is None:
            self.layer_idx = self.init_layer_idx
        else:
            self.layer_idx = layer_idx

        """
        This forward pass has two paths:
        0. FFN-MoE: It first prepares the MoE parameters, then directly sends them to expert computation.
        1. Attention-MoE: It first goes through attention to obtain one representation for each token, effectively replacing the original representation, then prepares the MoE parameters, and finally sends them to expert computation.
        """
        ########### attention ffn moe
        mixed_token, attn_weights, past_key_value = self.multi_head_attention(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
        
        bsz, head_num, q_len, h = mixed_token.size()

        moe_input = torch.cat((mixed_token, hidden_states.unsqueeze(1)), dim=1).reshape(bsz, (head_num + 1) * q_len, h)
        if att_layer_norm is not None:
            assert self.config.norm_after_mix
            moe_input = att_layer_norm(moe_input)
        expert_outputs = self.vo_experts(moe_input)
        expert_outputs = expert_outputs.reshape(bsz, head_num + 1, q_len, h).sum(dim=1)

        if not output_attentions:
            attn_weights = None

        return expert_outputs, attn_weights, past_key_value