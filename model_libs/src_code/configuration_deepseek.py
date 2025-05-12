from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}
class DeepseekV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV2Model`]. It is used to instantiate an DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DeepSeek-V2.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DeepseekV2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1407):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        n_shared_experts (`int`, *optional*, defaults to None):
            Number of shared experts, None means dense model.
        n_routed_experts (`int`, *optional*, defaults to None):
            Number of routed experts, None means dense model.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor or routed experts.
        topk_method (`str`, *optional*, defaults to `gready`):
            Topk method used in routed gate.
        n_group (`int`, *optional*, defaults to None):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to None):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        num_experts_per_tok (`int`, *optional*, defaults to None):
            Number of selected experts, None means dense model.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                            \--k dense layers--/
        norm_topk_prob (`bool`, *optional*, defaults to False):
            Whether to normalize the weights of the routed experts.
        scoring_func (`str`, *optional*, defaults to 'softmax'):
            Method of computing expert weights.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Auxiliary loss weight coefficient.
        seq_aux = (`bool`, *optional*, defaults to True):
            Whether to compute the auxiliary loss for each individual sample.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        premix_att_moe:
            Whether to enable premix version of attention MoE.
        share_att_ffn_moe:
            If premix version of MoE is enabled, whether to share this MoE directly with the FFN layer.
        ffn_num_experts_per_tok:
            If shared with FFN layer, how many experts to select?
        ffn_separate_router:
            If attention and FFN share MoE expert parameters, should they use different routers? Default is no.
        disable_share_expert_for_att:
            Whether to disable shared expert computation for attention, default includes it. Current shared expert logic is: if n_shared_experts is set, it will be created. But whether to use it for computation is determined by this parameter.
        disable_share_expert_for_ffn:
            Default is False, meaning FFN won't use shared expert for computation. If set to false, FFN will use shared expert, and if multi_share_expert is true, then seperate_share_expert_for_ffn must also be true.
        seperate_share_expert_for_ffn:
            Default is true. When disable_share_expert_for_ffn is false (enabling shared expert for FFN), whether to create a separate ffn_share_experts specifically for FFN use. This should only be considered when FFN and attention share parameters.
        multi_share_expert:
            The simplest shared expert is just an FFN, increasing the number of shared experts only increases the intermediate dimension size. However, if we want to use shared experts to implement multi-head attention, meaning having multiple shared experts with different inputs, this should be set to true.
        res_query_lora:
            If set to True, when using multi-head attention MoE, we use ResNet to improve LoRA effect, allowing smaller rank while achieving better results with fewer parameters.
        mix_then_routing_unify:
            Default False. If True:
            1) FFN and attention will be unified into one layer, layer count needs to be doubled accordingly
            2) Each time, we first do multi-head attention, then concatenate the multi-head output (embedding dim) with the token itself
            Finally, these tokens are combined and sent to a vanilla MoE, computed independently as tokens, then directly added together
        share_layer:
            Whether to reference Universal Transformer and share decoder layers
        share_layer_repeat_num:
            If sharing layers, how many layers to share? If set to 2, layer 0 and layer 1 will use the same attention
        norm_after_mix:
            Should layer norm happen before or after attention? If after, there's an issue where q,k computation uses pre-norm values, while v,o uses post-norm values
        skip_shared_ffn:
            Default is false. During ablation, when experimenting which layer (attention or FFN) should use more experts, FFN was skipped and computation resources were allocated to attention. In this case, skip_flag needs to be set to True.
        no_act:
            Default is false. If true, then PreMixMoE and VanillaMLP created within PreMixMoE will have no activation function (it's an identity function)
    ```python
    >>> from transformers import DeepseekV2Model, DeepseekV2Config

    >>> # Initializing a Deepseek-V2 style configuration
    >>> configuration = DeepseekV2Config()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "PremixMoe_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=11008,
        moe_intermediate_size = 1407,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts = None,
        n_routed_experts = None,
        ep_size = 1,
        routed_scaling_factor = 1.0,
        kv_lora_rank = 512,
        q_lora_rank = 1536,
        qk_rope_head_dim = 64,
        v_head_dim = 128,
        qk_nope_head_dim = 128,
        topk_method = 'gready',
        n_group = None,
        topk_group = None,
        num_experts_per_tok = None,
        moe_layer_freq = 1,
        first_k_dense_replace = 0,
        norm_topk_prob = False,
        scoring_func = 'softmax',
        aux_loss_alpha = 0.001,
        seq_aux = True,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        premix_att_moe=False,
        no_ffn_moe=False,
        share_att_ffn_moe=False,
        ffn_num_experts_per_tok=0,
        share_layer=False,
        baseline_moa=False,
        baseline_switchhead=False,
        ffn_separate_router=False,
        one_head_attention_moe=False,
        disable_share_expert_for_att=False,
        disable_share_expert_for_ffn=False,
        seperate_share_expert_for_ffn=True,
        multi_share_expert=False,
        res_query_lora=False,
        mix_then_routing_unify=False,
        share_layer_repeat_num=0,
        norm_after_mix=False,
        skip_shared_ffn=False,
        no_act=False,
        postmix_att_moe=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.premix_att_moe = premix_att_moe
        self.share_att_ffn_moe = share_att_ffn_moe
        self.no_ffn_moe = no_ffn_moe
        self.ffn_num_experts_per_tok = ffn_num_experts_per_tok
        self.share_layer = share_layer
        self.baseline_moa = baseline_moa
        self.baseline_switchhead = baseline_switchhead
        self.ffn_separate_router = ffn_separate_router
        self.one_head_attention_moe = one_head_attention_moe
        self.disable_share_expert_for_ffn = disable_share_expert_for_ffn
        self.multi_share_expert = multi_share_expert
        self.seperate_share_expert_for_ffn = seperate_share_expert_for_ffn
        self.disable_share_expert_for_att = disable_share_expert_for_att
        self.res_query_lora = res_query_lora
        self.mix_then_routing_unify = mix_then_routing_unify
        self.share_layer_repeat_num = share_layer_repeat_num
        self.norm_after_mix = norm_after_mix
        self.skip_shared_ffn = skip_shared_ffn
        self.no_act = no_act
        self.postmix_att_moe = postmix_att_moe

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )