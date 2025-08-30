#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GFSA (Global Frequency-Sensitive Attention) for HuggingFace AST.

- Swaps the AST attention with a GFSA variant.
- Loads base weights via `from_pretrained(...)`.
- Freezes the backbone; trains only GFSA per-head betas and the classifier.
- Includes an ablation class with modes: 'standard', 'no_high_order', 'learnable_beta'.

Tested with transformers >= 4.38 (ASTModel).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import (
    ASTLayer,
    ASTEncoder,
    ASTAttention,
    ASTSelfAttention,
    ASTSelfOutput,
)


@dataclass
class GFSA_config:
    """Config for the standard GFSA wrapper."""
    # num_heads is taken from model config to avoid mismatches
    order_h: int = 4                # order used in the HF component
    renormalize: bool = False       # whether to renormalize to probabilities after adding HF term


@dataclass
class GFSA_config_ablation:
    """Config for the ablation wrapper."""
    config: str = "standard"        # one of {'standard', 'no_high_order', 'learnable_beta'}
    order_h: int = 4
    renormalize: bool = False



class ASTModel_GFSA(ASTModel):
    """ASTModel with encoder replaced by GFSA encoder."""
    def __init__(self, config, gfsa_config: GFSA_config):
        super().__init__(config)
        self.gfsa_config = gfsa_config
        self.encoder = ASTEncoder_GFSA(config, gfsa_config)


class ASTEncoder_GFSA(ASTEncoder):
    """Encoder that builds GFSA layers."""
    def __init__(self, config, gfsa_config: GFSA_config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [ASTLayer_GFSA(config, gfsa_config) for _ in range(config.num_hidden_layers)]
        )


class ASTLayer_GFSA(ASTLayer):
    """Layer that swaps in GFSA attention."""
    def __init__(self, config, gfsa_config: GFSA_config):
        super().__init__(config)
        self.attention = ASTAttention_GFSA(config, gfsa_config)


class ASTAttention_GFSA(ASTAttention):
    """AST Attention that uses GFSA's self-attention."""
    def __init__(self, config, gfsa_config: GFSA_config):
        super().__init__(config)
        self.attention = ASTSelfAttention_GFSA(config, gfsa_config)


class ASTSelfAttention_GFSA(ASTSelfAttention):
    """GFSA Self-Attention: adds a high-order 'frequency-sensitive' term to attention probs."""
    def __init__(self, config, gfsa_config: GFSA_config):
        super().__init__(config)
        # initialize per-head learnable beta (lambda) with zeros
        num_heads = config.num_attention_heads
        self.lamb = nn.Parameter(torch.zeros(num_heads))
        self._eye_cache: Optional[torch.Tensor] = None
        self._order_h = gfsa_config.order_h
        self._renorm = gfsa_config.renormalize

    # small cache to avoid re-allocating identity matrices each step
    def _eye(self, L: int, device: torch.device, dtype: torch.dtype):
        if (
            self._eye_cache is None
            or self._eye_cache.size(0) < L
            or self._eye_cache.device != device
            or self._eye_cache.dtype != dtype
        ):
            self._eye_cache = torch.eye(L, dtype=dtype, device=device)
        return self._eye_cache[:L, :L]

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        # Standard AST projections
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Base attention logits
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Softmax -> probs
        attention_probs = F.softmax(attention_scores, dim=-1)

        # ---- GFSA high-order term ----
        # high_order = (h-1) * (A - I) @ A + A
        h = self._order_h
        L = attention_probs.size(-1)
        I = self._eye(L, attention_probs.device, attention_probs.dtype)[None, None, ...]
        high_order = (h - 1) * (attention_probs - I) @ attention_probs
        high_order += attention_probs
        beta = self.lamb[None, :, None, None]  # (1, heads, 1, 1)

        attention_probs = attention_probs + beta * high_order

        # optional renorm to keep it stochastic/stable
        if self._renorm:
            attention_probs = attention_probs.clamp_min(0)
            attention_probs = attention_probs / (attention_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Dropout and head mask (same as base)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


# -----------------------------
# Ablation variant
# -----------------------------

class ASTModel_GFSA_ablation(ASTModel):
    def __init__(self, config, gfsa_config: GFSA_config_ablation):
        super().__init__(config)
        self.gfsa_config = gfsa_config
        self.encoder = ASTEncoder_GFSA_ablation(config, gfsa_config)


class ASTEncoder_GFSA_ablation(ASTEncoder):
    def __init__(self, config, gfsa_config: GFSA_config_ablation):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [ASTLayer_GFSA_ablation(config, gfsa_config) for _ in range(config.num_hidden_layers)]
        )


class ASTLayer_GFSA_ablation(ASTLayer):
    def __init__(self, config, gfsa_config: GFSA_config_ablation):
        super().__init__(config)
        self.attention = ASTAttention_GFSA_ablation(config, gfsa_config)


class ASTAttention_GFSA_ablation(ASTAttention):
    def __init__(self, config, gfsa_config: GFSA_config_ablation):
        super().__init__(config)
        self.attention = ASTSelfAttention_GFSA_ablation(config, gfsa_config)
        self.gfsa_config = gfsa_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.lamb = nn.Parameter(torch.zeros(num_heads), requires_grad=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # with torch.cuda.amp.autocast(enabled=False):
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        del qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        h=4
        identity = torch.eye(attn.shape[-1],attn.shape[-1]).to(attn.device)
        identity = identity[None, None, ...]
        high_order = (h-1) * (attn - identity)@attn
        high_order += attn
        beta = self.lamb[None, :, None, None]
        attn = attn + beta * high_order 
        
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class ASTSelfAttention_GFSA_ablation(ASTSelfAttention):
    """Ablation: 'standard' (fixed beta=0 init, but learnable=buffer if desired), 
                  'no_high_order' (vanilla attention),
                  'learnable_beta' (trainable per-head beta)."""
    def __init__(self, config, gfsa_config: GFSA_config_ablation):
        super().__init__(config)
        self.configuration = gfsa_config.config
        self._order_h = gfsa_config.order_h
        self._renorm = gfsa_config.renormalize
        self._eye_cache: Optional[torch.Tensor] = None

        num_heads = config.num_attention_heads
        if self.configuration == "learnable_beta":
            self.lamb = nn.Parameter(torch.zeros(num_heads))
        elif self.configuration == "standard":
            # fixed zeros (no HF effect unless finetuned later by enabling grad)
            self.register_buffer("lamb", torch.zeros(num_heads), persistent=False)
        # 'no_high_order' => no lamb needed

    def _eye(self, L: int, device: torch.device, dtype: torch.dtype):
        if (
            self._eye_cache is None
            or self._eye_cache.size(0) < L
            or self._eye_cache.device != device
            or self._eye_cache.dtype != dtype
        ):
            self._eye_cache = torch.eye(L, dtype=dtype, device=device)
        return self._eye_cache[:L, :L]

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)

        if self.configuration in ("standard", "learnable_beta"):
            h = self._order_h
            L = attention_probs.size(-1)
            I = self._eye(L, attention_probs.device, attention_probs.dtype)[None, None, ...]
            high_order = (h - 1) * (attention_probs - I) @ attention_probs
            high_order = high_order + attention_probs
            beta = self.lamb[None, :, None, None]
            attention_probs = attention_probs + beta * high_order

            if self._renorm:
                attention_probs = attention_probs.clamp_min(0)
                attention_probs = attention_probs / (attention_probs.sum(dim=-1, keepdim=True) + 1e-9)
        # else: 'no_high_order' -> vanilla attention_probs

        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer).transpose(1,2).view(B,-1,C).contiguous()
        # context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


# -----------------------------
# High-level task wrappers
# -----------------------------

class AST_GFSA(nn.Module):
    """
    High-level task model:
    - Loads base AST weights into GFSA-modified architecture.
    - Freezes the backbone; trains GFSA betas + classifier head.
    - final_output: 'CLS' or 'ALL'
    """
    def __init__(
        self,
        max_length: int,
        num_classes: int,
        final_output: str,
        model_ckpt: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        order_h: int = 4,
        renormalize: bool = False,
    ):
        super().__init__()
        self.gfsa_config = GFSA_config(order_h=order_h, renormalize=renormalize)
        self.model = ASTModel_GFSA.from_pretrained(
            model_ckpt, self.gfsa_config, max_length=max_length, ignore_mismatched_sizes=True
        )
        self.model_config = self.model.config
        assert final_output in {"CLS", "ALL"}
        self.final_output = final_output

        self.embeddings = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)

        # Freeze backbone
        self.embeddings.requires_grad_(False)
        self.encoder.requires_grad_(False)
        self._unfreeze_gfsa()

    def _unfreeze_gfsa(self):
        for i in range(self.model_config.num_hidden_layers):
            self.encoder.layer[i].attention.attention.lamb.requires_grad_(True)

    def train(self, mode: bool = True):
        # keep backbone in eval; train only head + beta params
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            self.layernorm.train()
            self.classification_head.train()
        else:
            for m in self.children():
                m.train(mode)

    def forward(self, x: torch.Tensor):
        # Expect (B, 1, T, F) or (B, T, F). Make it (B, T, F).
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        x = self.embeddings(x)
        hidden_states = self.encoder(x)[0]
        hidden_states = self.layernorm(hidden_states)

        if self.final_output == "CLS":
            return self.classification_head(hidden_states[:, 0])
        return self.classification_head(hidden_states.mean(dim=1))


class AST_GFSA_ablation(nn.Module):
    """
    Ablation wrapper.
    config: 'standard' | 'no_high_order' | 'learnable_beta'
    """
    def __init__(
        self,
        max_length: int,
        num_classes: int,
        final_output: str,
        gfsa_mode: str = "standard",
        model_ckpt: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        order_h: int = 4,
        renormalize: bool = False,
    ):
        super().__init__()
        self.gfsa_config = GFSA_config_ablation(config=gfsa_mode, order_h=order_h, renormalize=renormalize)
        self.model = ASTModel_GFSA_ablation.from_pretrained(
            model_ckpt, self.gfsa_config, max_length=max_length, ignore_mismatched_sizes=True
        )
        self.model_config = self.model.config
        assert final_output in {"CLS", "ALL"}
        self.final_output = final_output

        self.embeddings = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)

        # Freeze backbone
        self.embeddings.requires_grad_(False)
        self.encoder.requires_grad_(False)
        self._unfreeze_if_needed()

    def _unfreeze_if_needed(self):
        if self.gfsa_config.config == "learnable_beta" or self.gfsa_config.config == "standard":
            # 'standard' keeps beta as buffer (non-trainable) unless you decide to enable grad later
            for i in range(self.model_config.num_hidden_layers):
                attn = self.encoder.layer[i].attention.attention
                if hasattr(attn, "lamb") and isinstance(attn.lamb, nn.Parameter):
                    attn.lamb.requires_grad_(True)

    def train(self, mode: bool = True):
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            self.layernorm.train()
            self.classification_head.train()
        else:
            for m in self.children():
                m.train(mode)

    def forward(self, x: torch.Tensor):
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        x = self.embeddings(x)
        hidden_states = self.encoder(x)[0]
        hidden_states = self.layernorm(hidden_states)

        if self.final_output == "CLS":
            return self.classification_head(hidden_states[:, 0])
        return self.classification_head(hidden_states.mean(dim=1))


if __name__ == "__main__":
    # Dummy batch: (B, 1, T, F)
    B, T, FREQ = 8, 1024, 128
    x = torch.randn(B, 1, T, FREQ)

    # Standard GFSA
    model = AST_GFSA(
        max_length=T,
        num_classes=4,
        final_output="CLS",
        model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593",
        order_h=4,
        renormalize=False,   # set True if you want row-stochastic attention after GFSA
    )

    # Only train trainable params (GFSA betas + classifier)
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)

    logits = model(x)
    print("Logits:", logits.shape)

    # Ablation example
    ablate = AST_GFSA_ablation(
        max_length=T,
        num_classes=4,
        final_output="CLS",
        gfsa_mode="learnable_beta",     # 'standard' | 'no_high_order' | 'learnable_beta'
        order_h=4,
        renormalize=False,
    )
    logits2 = ablate(x)
    print("Ablation logits:", logits2.shape)