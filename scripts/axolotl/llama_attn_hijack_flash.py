"""Flash attention monkey patch for llama model"""

# copied from https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py

import logging
import warnings
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from flash_attn.bert_padding import pad_input, unpad_input
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as OriginalLlamaDecoderLayer,
)
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    apply_rotary_pos_emb,
    repeat_kv,
)
from xformers.ops import SwiGLU

from utils import get_cu_seqlens_from_pos_ids, set_module_name

try:
    from flash_attn.flash_attn_interface import (  # pylint: disable=ungrouped-imports
        flash_attn_kvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
    )
except ImportError:
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_kvpacked_func as flash_attn_varlen_kvpacked_func,
    )
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_qkvpacked_func as flash_attn_varlen_qkvpacked_func,
    )

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

try:
    from flash_attn.layers.rotary import apply_rotary_emb_func

    flash_rope_installed = True
    print(">>>> Flash RoPE installed")
except ImportError:
    flash_rope_installed = False
    raise ImportError(
        "Please install RoPE kernels: `pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary`"
    )


LOG = logging.getLogger("axolotl")


def replace_llama_mlp_with_swiglu(model):
    for name, module in model.named_modules():
        if isinstance(module, LlamaMLP):
            mlp = FusedMLP(
                module.config, module.gate_proj, module.up_proj, module.down_proj
            )
            set_module_name(model, name, mlp)


def replace_llama_qkv_with_fused(model):
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            qkv = FusedAttention(
                module.config,
                module.q_proj,
                module.k_proj,
                module.v_proj,
                module.o_proj,
            )
            set_module_name(model, name, qkv)


def replace_llama_attn_with_flash_attn(
    packed: Optional[bool] = False,
    cross_entropy: Optional[bool] = False,
    rms_norm: Optional[bool] = False,
):
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (  # pylint: disable=protected-access
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = flashattn_forward
    if packed:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = LlamaDecoderLayer
        transformers.models.llama.modeling_llama.LlamaModel.forward = (
            llama_model_forward
        )
    transformers.models.llama.modeling_llama.LlamaAttention._init_rope = _init_rope

    # skip only if explicitly disabled
    if cross_entropy:
        try:
            from flash_attn.losses.cross_entropy import CrossEntropyLoss

            LOG.info("patching with flash_attn.losses.cross_entropy")
            transformers.models.llama.modeling_llama.CrossEntropyLoss = partial(
                CrossEntropyLoss, inplace_backward=True
            )
        except ImportError:
            LOG.info(
                "optimized flash-attention CrossEntropyLoss not found (run `pip install 'git+https://github.com/Dao-AILab/flash-attention.git#egg=xentropy_cuda_lib&subdirectory=csrc/xentropy'`)"
            )

    # skip only if explicitly disabled
    if rms_norm:
        try:
            from flash_attn.ops.rms_norm import RMSNorm

            class LlamaRMSNorm(RMSNorm):
                """Patched LLamaRMSNorm"""

                def __init__(self, hidden_size, eps=1e-6):
                    super().__init__(hidden_size, eps=eps)

            LOG.info("patching with flash_attn.ops.rms_norm")
            transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
        except ImportError:
            LOG.info(
                "optimized flash-attention RMSNorm not found (run `pip install 'git+https://github.com/Dao-AILab/flash-attention.git#egg=dropout_layer_norm&subdirectory=csrc/layer_norm'`)"
            )


def _init_rope(self):
    if self.config.rope_scaling is None:
        scaling_factor = 1
    else:
        scaling_type = self.config.rope_scaling["type"]
        scaling_factor = self.config.rope_scaling["factor"]
        assert scaling_type == "linear"
    self.rotary_emb = FlashRotaryEmbedding(
        self.head_dim,
        base=10000,
        interleaved=False,
        scaling_factor=scaling_factor,
    )


class FusedAttention(LlamaAttention):
    """
    Fused QKV Attention layer for incrementally improved training efficiency
    """

    def __init__(
        self,
        config,
        q: torch.nn.Linear,  # pylint: disable=invalid-name
        k: torch.nn.Linear,  # pylint: disable=invalid-name
        v: torch.nn.Linear,  # pylint: disable=invalid-name
        o: torch.nn.Linear,  # pylint: disable=invalid-name
    ):
        super().__init__(config)
        self.config = config
        self.init_device = next(iter(q.state_dict().values())).device

        # define equivalent fused qkv projection
        self.out_features: List[int] = [q.out_features, k.out_features, v.out_features]
        self.qkv_proj = torch.nn.Linear(
            q.in_features, sum(self.out_features), device=self.init_device, bias=False
        )
        self.o_proj = o

        # overwrite initialized weights with pretrained weights
        self.qkv_proj.weight.data = torch.cat(
            (q.weight.data, k.weight.data, v.weight.data), dim=0
        )

    def _post_training(self, model, name):
        q_proj, k_proj, v_proj = torch.split(
            self.qkv_proj.weight.data, self.out_features, dim=0
        )

        new_attn = LlamaAttention(self.config)
        new_attn.q_proj.weight.data = q_proj
        new_attn.k_proj.weight.data = k_proj
        new_attn.v_proj.weight.data = v_proj
        new_attn.o_proj.weight.data = self.o_proj.weight.data

        set_module_name(model, name, new_attn)


class FusedMLP(torch.nn.Module):
    """
    Fused MLP layer for incrementally improved training efficiency
    """

    def __init__(
        self,
        config,
        gate_proj: torch.nn.Linear,
        up_proj: torch.nn.Linear,
        down_proj: torch.nn.Linear,
    ):
        super().__init__()
        self.config = config
        self.swiglu = SwiGLU(
            in_features=config.hidden_size,
            hidden_features=config.intermediate_size,
            bias=False,
            _pack_weights=True,
        )
        # overwrite initialized weights with pretrained weights
        self.swiglu.w12.weight.data = torch.cat(
            (gate_proj.weight.data, up_proj.weight.data), dim=0
        )
        self.swiglu.w3.weight.data = down_proj.weight.data

    def _post_training(self, model, name):
        w1, w2 = torch.split(  # pylint: disable=invalid-name
            self.swiglu.w12.weight.data, self.config.intermediate_size, dim=0
        )

        # Assign the split weights back to the original layers
        new_mlp = LlamaMLP(self.config)
        new_mlp.gate_proj.weight.data = w1
        new_mlp.up_proj.weight.data = w2
        new_mlp.down_proj.weight.data = self.swiglu.w3.weight.data

        set_module_name(model, name, new_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        return self.swiglu(x)


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self,
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
):  # pylint: disable=unused-argument
    # [bsz, seq_len]
    return attention_mask


def flashattn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,  # pylint: disable=unused-argument
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    # pylint: disable=duplicate-code
    bsz, q_len, _ = hidden_states.size()

    if not hasattr(self, "pretraining_tp"):
        self.pretraining_tp = 1

    if self.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        if isinstance(self, FusedAttention):
            query_states, key_states, value_states = self.qkv_proj(hidden_states).split(
                self.out_features, dim=-1
            )
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    #
    # flash-attn v2 start
    #

    if self.training:
        # during training q,k,v always have same seqlen
        assert key_states.shape == query_states.shape
        is_causal = True
    else:
        # turn off FA causal mask after first inference autoregressive iteration
        # only on first autoregressive step q,k,v have same seqlen
        is_causal = key_states.shape == query_states.shape

    dropout_rate = 0.0 if not self.training else getattr(self, "attention_dropout", 0.0)

    if cu_seqlens is not None and max_seqlen is not None and cu_seqlens.dim() == 1:
        # special handling using sample packing
        qkv = torch.stack(
            [query_states, key_states, value_states], dim=2
        )  # [bsz, nh, 3, q_len, hd]
        qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
        qkv = rearrange(qkv, "b s ... -> (b s) ...")

        output = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p=dropout_rate,
            softmax_scale=None,
            causal=True,
        )
        output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
    elif query_states.shape == key_states.shape:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        qkv_unpad, cu_seqlens_q, max_seqlen_q, _, output_pad_fn = generate_qkv(
            query_states,
            key_states,
            value_states,
            qkvpacked=True,
            # We have disabled _prepare_decoder_attention_mask in LlamaModel
            # the attention_mask should be the same as the key_padding_mask
            key_padding_mask=attention_mask,
            query_padding_mask=attention_mask[:, -query_states.size(1) :]
            if attention_mask is not None
            else None,
        )
        output_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad,
            cu_seqlens_q,
            max_seqlen_q,
            dropout_p=dropout_rate,
            softmax_scale=None,
            causal=is_causal,
        )
        output = output_pad_fn(output_unpad)
    else:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if attention_mask is None or attention_mask.all().item():
            output = flash_attn_kvpacked_func(
                query_states,
                torch.stack([key_states, value_states], 2),
                dropout_p=dropout_rate,
                causal=is_causal,
            )
        else:
            (  # pylint: disable=unbalanced-tuple-unpacking
                q_unpad,
                kv_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                _,
                _,
                output_pad_fn,
            ) = generate_qkv(
                query_states,
                key_states,
                value_states,
                kvpacked=True,
                key_padding_mask=attention_mask,
                query_padding_mask=attention_mask[:, -query_states.size(1) :]
                if attention_mask is not None
                else None,
            )
            if q_unpad.dtype != kv_unpad.dtype:
                kv_unpad = kv_unpad.to(q_unpad.dtype)
            output_unpad = flash_attn_varlen_kvpacked_func(
                q_unpad,
                kv_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=dropout_rate,
                softmax_scale=None,
                causal=is_causal,
            )
            output = output_pad_fn(output_unpad)

    attn_output = output
    if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    attn_output = rearrange(attn_output, "b s h d -> b s (h d)")

    #
    # flash-attn v2 end
    #

    if self.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.pretraining_tp, dim=1
        )
        attn_output = sum(
            F.linear(attn_output[i], o_proj_slices[i])
            for i in range(self.pretraining_tp)
        )
    else:
        attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


# based on https://github.com/Dao-AILab/flash-attention/blob/364a5b/tests/test_flash_attn.py#L38
def generate_qkv(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    kvpacked=False,
    qkvpacked=False,
):  # pylint: disable=invalid-name,unnecessary-lambda-assignment
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
            q, query_padding_mask
        )

        output_pad_fn = lambda output_unpad: pad_input(  # noqa: E731
            output_unpad, indices_q, batch_size, seqlen_q
        )

    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=q_unpad.device,
        )
        max_seqlen_q = seqlen_q

        output_pad_fn = lambda output_unpad: rearrange(  # noqa: E731
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, _, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * seqlen_k,
            step=seqlen_k,
            dtype=torch.int32,
            device=k_unpad.device,
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        return (qkv_unpad, cu_seqlens_q, max_seqlen_q, qkv, output_pad_fn)

    if kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        return (
            q_unpad,
            kv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            kv,
            output_pad_fn,
        )

    return (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
    )


def llama_model_forward(
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
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    if input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    cu_seqlens = None
    max_seqlen = None
    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()
        cu_seqlens, max_seqlen = get_cu_seqlens_from_pos_ids(position_ids)
        cu_seqlens = cu_seqlens.squeeze()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

    attention_mask = (
        self._prepare_decoder_attention_mask(  # pylint: disable=protected-access
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            transformers.logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(
                        *inputs,
                    )

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                None,
                padding_mask,
                cu_seqlens,
                max_seqlen,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
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


class LlamaDecoderLayer(OriginalLlamaDecoderLayer):
    """
    patched version of LlamaDecoderLayer to pass through the precalculated cu_seqlens
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cu_seqlens (`torch.Tensor`, *optional*) cumulative sequence len when packing
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class FlashRotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        scaling_factor=1.0,
        pos_idx_in_fp32=True,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        scaling_factor: RotaryEmbedding extended with linear scaling.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
            / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training

        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seqlen_offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(
            q.shape[1] + seqlen_offset, device=q.device, dtype=q.dtype
        )
        if self.scale is None:
            return apply_rotary_emb_func(
                q,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self.interleaved,
                True,  # inplace=True
            ), apply_rotary_emb_func(
                k,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self.interleaved,
                True,  # inplace=True
            )
        else:
            assert False
