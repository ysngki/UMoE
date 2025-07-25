import torch
from torch import nn
from transformers.activations import ACT2FN

from .basic_layer_components import (
    VanillaMLP,
)
from .moe_router import MoEGate
from .basic_utils import raise_moe_lib_exception
from .grouped_gemm_util import ops as gg_ops

###################### MoE Experts libs
try:
    import transformer_engine as te
    has_te = True
except ImportError:
    has_te = False

try:
    from unsloth_moe import grouped_gemm
    has_unsloth = True
except ImportError:
    has_unsloth = False
######################


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

        self.act_fn = ACT2FN[config.hidden_act]

        if self.config.use_megatron_cutlass_group_gemm or self.config.use_unsloth_moe:
            if self.config.use_unsloth_moe:
                assert has_unsloth, "You should install https://github.com/unslothai/unsloth/tree/main/unsloth/kernels/moe"

            if self.config.swiglue_mlp:
                self.linear_1 = nn.Linear(self.hidden_size, config.moe_intermediate_size * self.experts_per_rank * 2, bias=False)
            else:
                self.linear_1 = nn.Linear(self.hidden_size, config.moe_intermediate_size * self.experts_per_rank, bias=False)

            self.linear_2 = nn.Linear(config.moe_intermediate_size * self.experts_per_rank, self.hidden_size, bias=False)
        elif self.config.use_te_group_linear:
            if not has_te:
                raise Exception("Transformer engine should be installed!")

            self.linear_1 = te.pytorch.GroupedLinear(num_gemms=self.experts_per_rank, in_features=self.hidden_size, out_features=config.moe_intermediate_size, bias=False)

            if self.config.swiglue_mlp:
                self.linear_g = te.pytorch.GroupedLinear(num_gemms=self.experts_per_rank, in_features=self.hidden_size, out_features=config.moe_intermediate_size, bias=False)

            self.linear_2 = te.pytorch.GroupedLinear(num_gemms=self.experts_per_rank, in_features=config.moe_intermediate_size, out_features=self.hidden_size, bias=False)
        else:
            raise_moe_lib_exception()

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = VanillaMLP(
                config=config, intermediate_size=intermediate_size
            )
        self.group_permute = True
        self.pytorch_native_permute = False


    def forward(self, hidden_states):
        output_shape = hidden_states.shape

        ## shared expert
        shared_exp_output = None
        if self.config.n_shared_experts is not None and self.config.n_shared_experts > 0:
            shared_exp_output = self.shared_experts(hidden_states)

        ## routing
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states, topk=self.num_experts_per_tok)
        topk_idx = topk_idx.to(torch.int32)

        topk = topk_weight.size(1)
        num_unpermuted_tokens = topk_weight.numel()
        ################### permute: organizaed by expert
        flatten_indices = topk_idx.view(-1)
        if self.group_permute:
            sorted_indices = None
        elif self.pytorch_native_permute:
            sorted_indices = torch.argsort(flatten_indices, stable=True)

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        assert topk_weight.shape[0] == hidden_states.shape[0] and num_unpermuted_tokens // hidden_states.shape[0] == topk
        
        if self.group_permute:
            # row_id_map is:
            #  [p_0^0, p_1^0, ... p_N^0, p_0^1, p_1^1, ..., P_N^1, ...., P_0^{K-1}, ...., P_N^{K-1}]
            # where ```p_i^j```` is the indice of j-th copy of token-i in the permuted_hidden_states 
            permuted_hidden_states, row_id_map = gg_ops.permute(hidden_states, topk_idx)
        elif self.pytorch_native_permute:
            permuted_hidden_states = hidden_states.index_select(0, sorted_indices // topk)

        ######################## calculation
        num_local_tokens_per_expert = torch.bincount(flatten_indices, minlength=self.experts_per_rank)
        # tokens_per_expert = num_local_tokens_per_expert.to(
        #     torch.device("cpu")
        # )
        tokens_per_expert = num_local_tokens_per_expert

        if self.config.use_megatron_cutlass_group_gemm:
            tokens_per_expert = tokens_per_expert.to(
                torch.device("cpu")
            )

            w1 = self.linear_1.weight.view(self.experts_per_rank, self.hidden_size, -1)
            w2 = self.linear_2.weight.view(self.experts_per_rank, -1, self.hidden_size)

            fc1_output = gg_ops.gmm(
                permuted_hidden_states, w1, tokens_per_expert, trans_b=False
            )
            if self.config.swiglue_mlp:
                gate_h, up_h = fc1_output.chunk(2, dim=-1)
                intermediate_parallel = F.silu(gate_h) * up_h
            else:
                intermediate_parallel = self.act_fn(fc1_output)
            permuted_tokens = gg_ops.gmm(
                intermediate_parallel, w2, tokens_per_expert, trans_b=False
            )
        elif self.config.use_unsloth_moe:
            w1 = self.linear_1.weight.view(self.experts_per_rank, -1, self.hidden_size)
            w2 = self.linear_2.weight.view(self.experts_per_rank, self.hidden_size, -1)

            fc1_output = grouped_gemm(
                X=permuted_hidden_states,
                W=w1,
                m_sizes=tokens_per_expert,
                topk=topk,
                autotune=True,
            )
            if self.config.swiglue_mlp:
                gate_h, up_h = fc1_output.chunk(2, dim=-1)
                intermediate_parallel = F.silu(gate_h) * up_h
            else:
                intermediate_parallel = self.act_fn(fc1_output)
            permuted_tokens = grouped_gemm(
                X=intermediate_parallel,
                W=w2,
                m_sizes=tokens_per_expert,
                topk=topk,
                autotune=True,
            )
        elif self.config.use_te_group_linear:
            m_sizes = tokens_per_expert.to(
                torch.device("cpu")
            ).tolist()

            fc1_output = self.linear_1(permuted_hidden_states, m_sizes)
            if self.config.swiglue_mlp:
                g_output = self.linear_g(permuted_hidden_states, m_sizes)
                permuted_tokens = F.silu(fc1_output) * g_output
                del fc1_output, g_output
            else:
                permuted_tokens = self.act_fn(fc1_output)
                del fc1_output

            permuted_tokens = self.linear_2(permuted_tokens, m_sizes)

        ################### unpermute
        if self.group_permute:
            unpermuted_tokens = gg_ops.unpermute(permuted_tokens, row_id_map, topk_weight)
        elif self.pytorch_native_permute:
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