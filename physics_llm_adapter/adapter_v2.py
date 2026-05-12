"""
Physics-LLM Adapter V2

Edge-deployable adapter with:
- DistilGPT-2 (82M params) for efficient inference
- Dedicated numerical regression head for precise values
- Hybrid architecture: LLM for categorical, regression head for numerical
- Optimized for embodied robotics and edge deployment
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# =============================================================================
# LoRA (Low-Rank Adaptation) for GPT-2 Conv1D layers
# =============================================================================

class LoRAConv1D(nn.Module):
    """LoRA wrapper for GPT-2's Conv1D layers.

    Conv1D forward: y = x @ weight + bias   (weight shape: [nx, nf])
    LoRA addition:  y += (x @ A @ B) * (alpha / rank)

    A: [nx, rank]  initialized with Kaiming uniform
    B: [rank, nf]  initialized to zeros (zero LoRA output at start)
    """

    def __init__(self, original: Conv1D, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        nx = original.nx  # in_features
        nf = original.nf  # out_features

        self.lora_A = nn.Parameter(torch.empty(nx, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, nf))
        self.lora_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base_out + lora_out

    def merge_weights(self):
        """Merge LoRA weights into original Conv1D (irreversible)."""
        self.original.weight.data += (self.lora_A @ self.lora_B).detach() * self.scaling

sys.path.append(str(Path(__file__).parent.parent / "physics_former"))

try:
    from mcq_head import MCQScoringHead, encode_texts
except ImportError:
    from physics_llm_adapter.mcq_head import MCQScoringHead, encode_texts

try:
    from adapter_heads import (
        OutputType, DescriptiveSubtype, CLEVRERQuestionCategory,
        CLEVRER_DESCRIPTIVE_VOCAB, CLEVRER_DESCRIPTIVE_VOCAB_SIZE,
        CLEVRER_ANSWER_TO_IDX, CLEVRER_IDX_TO_ANSWER,
        SUBTYPE_VOCABS, AGENT_CENTRIC_QUESTIONS,
        classify_descriptive_subtype, classify_clevrer_question,
        DescriptiveSubHead, DescriptiveHead, NumericalHead,
        ObjectMaskingHead,
    )
except ImportError:
    from physics_llm_adapter.adapter_heads import (
        OutputType, DescriptiveSubtype, CLEVRERQuestionCategory,
        CLEVRER_DESCRIPTIVE_VOCAB, CLEVRER_DESCRIPTIVE_VOCAB_SIZE,
        CLEVRER_ANSWER_TO_IDX, CLEVRER_IDX_TO_ANSWER,
        SUBTYPE_VOCABS, AGENT_CENTRIC_QUESTIONS,
        classify_descriptive_subtype, classify_clevrer_question,
        DescriptiveSubHead, DescriptiveHead, NumericalHead,
        ObjectMaskingHead,
    )


class PhysicsLLMAdapterV2(nn.Module):
    """
    Edge-deployable Physics-LLM Adapter with:
    - DistilGPT-2 (82M params) for efficient inference
    - Dedicated numerical regression head
    - Hybrid output: categorical via LLM, numerical via regression
    - Total model size: ~100M params (suitable for Jetson, RPi, etc.)
    """
    
    # Supported LLM backbones. ``dim`` is the model's hidden size and must match
    # what the prefix MLP projects to so the prefix tokens align with token
    # embeddings the LLM produces. ``family`` distinguishes architectures that
    # need different access patterns:
    #   - ``gpt2``    : has ``llm.transformer`` (GPT2Model) and uses
    #                   ``transformers.Conv1D`` attention layers; we keep the
    #                   custom ``LoRAConv1D`` integration for these.
    #   - ``llama``   : LLaMA-family decoder (Qwen2.5, LLaMA, Mistral, ...) with
    #                   ``llm.model`` as the base + standard ``nn.Linear``
    #                   attention layers; LoRA goes through ``peft`` instead.
    LLM_CONFIGS = {
        "distilgpt2":                    {"dim": 768,  "params": "82M",  "family": "gpt2"},
        "gpt2":                          {"dim": 768,  "params": "124M", "family": "gpt2"},
        "gpt2-medium":                   {"dim": 1024, "params": "355M", "family": "gpt2"},
        "gpt2-large":                    {"dim": 1280, "params": "774M", "family": "gpt2"},
        # Qwen2.5 instruction-tuned family. Hidden sizes verified against the
        # HF model cards (Qwen2.5 config.hidden_size). Used for Phase 7+.
        "Qwen/Qwen2.5-0.5B-Instruct":    {"dim": 896,  "params": "0.5B", "family": "llama"},
        "Qwen/Qwen2.5-1.5B-Instruct":    {"dim": 1536, "params": "1.5B", "family": "llama"},
        "Qwen/Qwen2.5-3B-Instruct":      {"dim": 2048, "params": "3B",   "family": "llama"},
    }
    
    def __init__(
        self,
        physics_model,
        physics_dim: int = 768,
        llm_name: str = "distilgpt2",
        num_prefix_tokens: int = 64,
        freeze_physics: bool = True,
        freeze_llm: bool = True,
        input_noise: float = 0.01,
        use_object_masking: bool = False,
        object_masking_weight: float = 0.1,
        label_smoothing: float = 0.0,
        build_mcq_head: bool = True,
        build_descriptive_head: bool = True,
        adapter_mlp_style: str = "deep",
        tokens_per_object: int = 0,
        # ----- Phase 7 (Qwen + LoRA + prefix-grounding) -----
        # When use_lora=True we apply LoRA at __init__ time. The LoRA backend
        # is auto-selected by LLM family: GPT-2 keeps the in-house
        # ``LoRAConv1D`` system; LLaMA-style (Qwen) goes through ``peft``.
        # ``freeze_llm`` stays True in this regime: only LoRA + adapter +
        # heads update during training.
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # ``prefix_dropout``: probability per training step of replacing the
        # prefix tokens with zeros. Forces the LM-side LoRA to learn that the
        # prefix carries the discriminative signal (without it the LoRA may
        # ignore the prefix entirely since the same prefix structure is
        # always present). Read by V3.compute_loss; harmless for V2.
        prefix_dropout: float = 0.0,
    ):
        super().__init__()
        
        # Configure LLM
        if llm_name not in self.LLM_CONFIGS:
            raise ValueError(f"Unknown LLM: {llm_name}. Options: {list(self.LLM_CONFIGS.keys())}")
        
        self.llm_name = llm_name
        self.llm_dim = self.LLM_CONFIGS[llm_name]["dim"]
        self.llm_params = self.LLM_CONFIGS[llm_name]["params"]
        self.llm_family = self.LLM_CONFIGS[llm_name].get("family", "gpt2")
        # Phase 7 knobs (stored so the V3 training loop can reach them).
        self.use_lora = bool(use_lora)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = lora_target_modules
        self.prefix_dropout = float(prefix_dropout)
        
        self.physics_dim = physics_dim
        self.input_noise = input_noise  # Noise level for regularization
        self.use_object_masking = use_object_masking
        self.object_masking_weight = object_masking_weight
        self.label_smoothing = label_smoothing

        # Per-object prefix mode: when tokens_per_object > 0 the adapter
        # emits K prefix tokens *per object* (tokens_per_object=K) instead
        # of a single mean-pooled prefix. This preserves per-object
        # identity through the adapter, recovering the shape/material
        # signal that the probe found the old mean-pool path was washing
        # out (encoder_pool shapes_present AUC 0.83 -> prefix_pool 0.65,
        # materials 0.74 -> 0.69 on CLEVRER). Total prefix size becomes
        # ``physics_model.max_objects * K`` (e.g. 20 * 4 = 80).
        if tokens_per_object < 0:
            raise ValueError(
                f"tokens_per_object must be >= 0, got {tokens_per_object}"
            )
        self.tokens_per_object = int(tokens_per_object)
        if self.tokens_per_object > 0:
            max_objects = int(physics_model.max_objects)
            self.num_prefix_tokens = max_objects * self.tokens_per_object
        else:
            self.num_prefix_tokens = num_prefix_tokens

        self.physics_model = physics_model
        if freeze_physics:
            self.physics_model.eval()
            for param in self.physics_model.parameters():
                param.requires_grad = False

        # Adapter MLP topology — checkpoint compatible.
        #   "deep"    : physics -> 4*llm_dim -> 4*llm_dim -> llm_dim*num_prefix  (current)
        #   "shallow" : physics -> physics   -> llm_dim*num_prefix               (legacy)
        # ``shallow`` matches pre-refactor checkpoints that only have adapter.[0,1,4,5]
        # state-dict entries (2-linear bridge, final LayerNorm on prefix output).
        # When tokens_per_object > 0 we build the per-object MLP instead of
        # the scene-level ``adapter``; the two are mutually exclusive so
        # checkpoints cleanly discriminate by state-dict keys.
        self.adapter_mlp_style = adapter_mlp_style
        if self.tokens_per_object > 0:
            adapter_hidden = self.llm_dim * 4  # 3072 for distilgpt2
            self.adapter_per_object = nn.Sequential(
                nn.Linear(physics_dim, adapter_hidden),
                nn.LayerNorm(adapter_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(adapter_hidden, adapter_hidden),
                nn.LayerNorm(adapter_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                # Emit K prefix tokens per object.
                nn.Linear(adapter_hidden, self.llm_dim * self.tokens_per_object),
            )
            # Learned bias disambiguating tokens within the same object
            # (without it, each of the K slots for one object is a pure
            # linear function of the same per-object embedding, so they
            # collapse toward each other). Shape [K, llm_dim], added
            # broadcasted in ``create_prefix_tokens_per_object``.
            self.per_object_slot_bias = nn.Parameter(
                torch.zeros(self.tokens_per_object, self.llm_dim)
            )
        elif adapter_mlp_style == "deep":
            adapter_hidden = self.llm_dim * 4  # 3072 for distilgpt2
            self.adapter = nn.Sequential(
                nn.Linear(physics_dim, adapter_hidden),
                nn.LayerNorm(adapter_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(adapter_hidden, adapter_hidden),
                nn.LayerNorm(adapter_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(adapter_hidden, self.llm_dim * self.num_prefix_tokens),
            )
        elif adapter_mlp_style == "shallow":
            # Legacy topology: indices 0 (Linear), 1 (LN), 4 (Linear), 5 (LN)
            self.adapter = nn.Sequential(
                nn.Linear(physics_dim, physics_dim),
                nn.LayerNorm(physics_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(physics_dim, self.llm_dim * self.num_prefix_tokens),
                nn.LayerNorm(self.llm_dim * self.num_prefix_tokens),
            )
        else:
            raise ValueError(
                f"Unknown adapter_mlp_style={adapter_mlp_style!r}. "
                f"Expected 'deep' or 'shallow'."
            )
        self._init_adapter_weights()
        
        self.numerical_head = NumericalHead(
            input_dim=physics_dim,
            hidden_dim=768,
            num_outputs=6
        )
        
        # Descriptive head is optional — older checkpoints may not include it.
        # Routing is by attribute presence (hasattr(self, 'descriptive_head')).
        self.has_descriptive_head = build_descriptive_head
        if build_descriptive_head:
            self.descriptive_head = DescriptiveHead(
                input_dim=physics_dim,
                hidden_dim=512
            )
        
        print(f"Loading LLM: {self.llm_name} ({self.llm_params} params)...")
        self.llm = AutoModelForCausalLM.from_pretrained(self.llm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        # Older GPT-2 tokenizers ship without a ``pad_token``; Qwen2.5 ships
        # one already. Only set if missing so we don't clobber Qwen's
        # native pad token (which has a different ID from EOS, useful for
        # right-padding during generation).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if freeze_llm:
            self.llm.eval()
            for param in self.llm.parameters():
                param.requires_grad = False

        # Phase 7: optional LoRA application. We dispatch by LLM family --
        # GPT-2 uses the in-house ``apply_lora()``/``LoRAConv1D`` system
        # (Conv1D-aware), LLaMA-style models use the ``peft`` library
        # (standard ``nn.Linear`` LoRA). Both paths leave the base LLM
        # weights frozen so only LoRA, the adapter MLP, and any heads train.
        if self.use_lora:
            if self.llm_family == "gpt2":
                # GPT-2 path: defer to ``apply_lora`` which knows about
                # ``c_attn`` / ``c_proj`` Conv1D modules.
                target = self.lora_target_modules or ('c_attn', 'c_proj')
                self.apply_lora(
                    rank=self.lora_rank, alpha=self.lora_alpha,
                    dropout=self.lora_dropout, target_modules=tuple(target),
                )
            else:
                # LLaMA / Qwen path: peft LoraConfig over standard q/k/v/o.
                # Importing peft lazily so the GPT-2-only flow doesn't take
                # the dependency unless someone asks for it.
                from peft import LoraConfig, get_peft_model, TaskType
                target = self.lora_target_modules or [
                    'q_proj', 'k_proj', 'v_proj', 'o_proj']
                lora_cfg = LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    target_modules=list(target),
                    bias='none',
                    task_type=TaskType.CAUSAL_LM,
                )
                self.llm = get_peft_model(self.llm, lora_cfg)
                # Quick sanity print so the train log shows LoRA is active.
                trainable = sum(
                    p.numel() for p in self.llm.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.llm.parameters())
                pct = 100.0 * trainable / max(total, 1)
                print(f"[LoRA/peft] {self.llm_name}: rank={self.lora_rank}, "
                      f"alpha={self.lora_alpha}, target={list(target)}")
                print(f"[LoRA/peft] trainable params: {trainable:,} / "
                      f"{total:,} ({pct:.3f}%)")
        
        # Object masking head (ALOE-style self-supervised auxiliary loss)
        if use_object_masking:
            self.object_masking_head = ObjectMaskingHead(
                embed_dim=physics_dim,
                num_mask_schemes=6,
                mask_ratio=0.3
            )
        
        # MCQ scoring head — optional. Older checkpoints do not contain
        # `mcq_head.*` weights and can be loaded with build_mcq_head=False
        # (routing falls back to the LLM causal-LM scoring path).
        self.has_mcq_head = build_mcq_head
        if build_mcq_head:
            self.mcq_head = MCQScoringHead(
                physics_dim=physics_dim,
                text_dim=self.llm_dim,
                hidden_dim=512,
                dropout=0.1
            )
        
        print(f"PhysicsLLMAdapterV2 initialized:")
        print(f"  LLM: {self.llm_name} (dim={self.llm_dim}, family={self.llm_family})")
        print(f"  Physics dim: {physics_dim}")
        print(f"  Prefix tokens: {num_prefix_tokens}")
        print(f"  Numerical head outputs: 6")
        print(f"  Descriptive head: {'multi-head (count=6, exist=2, color=8, shape=3, material=2)' if build_descriptive_head else 'disabled'}")
        print(f"  MCQ head: {'direct classifier (physics+text -> score)' if build_mcq_head else 'disabled'}")
        print(f"  Object masking: {'ENABLED' if use_object_masking else 'disabled'} (weight={object_masking_weight})")
        print(f"  Label smoothing: {label_smoothing}")
        print(f"  Physics frozen: {freeze_physics}")
        print(f"  LLM frozen: {freeze_llm}")
        if self.use_lora:
            print(f"  LoRA: ENABLED (rank={self.lora_rank}, alpha={self.lora_alpha}, "
                  f"backend={'in-house Conv1D' if self.llm_family == 'gpt2' else 'peft'})")
        if self.prefix_dropout > 0:
            print(f"  Prefix dropout: p={self.prefix_dropout} "
                  f"(applied during V3 training to ground prefix usage)")

    # ------------------------------------------------------------------
    # Architecture-portable accessors. The GPT-2 family stores the base
    # transformer at ``llm.transformer`` and the input embedding at
    # ``llm.transformer.wte``; LLaMA-family models (Qwen2.5, LLaMA, ...)
    # store them at ``llm.model`` and ``llm.model.embed_tokens``. Wrapping
    # both with HF's standard accessors lets the rest of this module stay
    # agnostic to the LLM choice. PEFT-wrapped models proxy these calls
    # to the underlying base model, so this works post-LoRA too.
    # ------------------------------------------------------------------

    def _get_input_embeddings(self) -> nn.Module:
        """Return the LLM's token-id -> embedding lookup module.

        Equivalent to ``self.llm.transformer.wte`` on GPT-2 and
        ``self.llm.model.embed_tokens`` on Qwen/LLaMA. Standard HF API,
        works for PEFT-wrapped models as well.
        """
        return self.llm.get_input_embeddings()

    def _get_base_model(self) -> nn.Module:
        """Return the LLM's base transformer (no LM head).

        Used by the MCQ-head text encoder, which wants contextual
        embeddings (last_hidden_state). For PEFT-wrapped models we have
        to descend into ``base_model.model`` first because peft inserts
        a ``base_model`` proxy.
        """
        base = self.llm
        # peft.PeftModel wraps as base_model -> LoraModel -> the actual HF model.
        # Reach the real HF model regardless.
        if hasattr(base, 'base_model'):
            inner = getattr(base.base_model, 'model', base.base_model)
            if hasattr(inner, 'transformer') or hasattr(inner, 'model'):
                base = inner
        if hasattr(base, 'transformer'):    # GPT-2 family (GPT2Model)
            return base.transformer
        if hasattr(base, 'model'):          # LLaMA / Qwen family
            return base.model
        raise AttributeError(
            f"Cannot find base transformer on {type(self.llm).__name__}; "
            f"expected `.transformer` or `.model` attribute.")

    def train(self, mode=True):
        """Override to keep frozen modules in eval mode (prevents dropout noise in scoring)."""
        super().train(mode)
        if mode:
            # Physics model is always frozen
            self.physics_model.eval()
            # Keep LLM in eval unless its transformer layers are being trained.
            # Architecture-agnostic check: if no LLM parameter requires grad
            # (the typical frozen-LLM + LoRA-only training setup), keep the
            # base LLM in eval mode so dropout / batchnorm stay deterministic.
            # This is more general than the old ``transformer.h`` check, which
            # only applied to GPT-2.
            llm_params_training = any(
                p.requires_grad for p in self.llm.parameters()
            )
            if not llm_params_training:
                self.llm.eval()
        return self

    def _init_adapter_weights(self):
        # Initialise whichever adapter MLP was built (scene-level ``adapter``
        # or per-object ``adapter_per_object``). Small gain matches the
        # original V2 init; larger gains caused the prefix to overwhelm the
        # frozen LLM's embedding scale early in training.
        mlp = (self.adapter_per_object if self.tokens_per_object > 0
               else self.adapter)
        for module in mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def extract_physics_features(self, physics_states, object_mask):
        """Extract features from physics model with optional noise injection for regularization."""
        # Add noise during training for better generalization
        if self.training and self.input_noise > 0:
            noise = torch.randn_like(physics_states) * self.input_noise
            physics_states = physics_states + noise
        
        with torch.no_grad() if not self.physics_model.training else torch.enable_grad():
            is_sequence = physics_states.dim() == 4
            
            if is_sequence:
                batch_size, seq_len, num_objects, state_dim = physics_states.shape
                velocity = physics_states[:, 1:] - physics_states[:, :-1]
                zero_velocity = torch.zeros(
                    batch_size, 1, num_objects, state_dim,
                    device=physics_states.device, dtype=physics_states.dtype
                )
                velocity_padded = torch.cat([zero_velocity, velocity], dim=1)
                augmented_states = torch.cat([physics_states, velocity_padded], dim=-1)
            else:
                batch_size, num_objects, state_dim = physics_states.shape
                zero_velocity = torch.zeros_like(physics_states)
                augmented_states = torch.cat([physics_states, zero_velocity], dim=-1)
            
            object_embeddings = self.physics_model.encode_physics(augmented_states, object_mask)
            
            if object_embeddings.dim() == 4:
                batch_size, seq_len, num_obj, hidden = object_embeddings.shape
                if object_mask is not None:
                    # Collapse 3D temporal mask to 2D (objects don't appear/disappear mid-sequence)
                    if object_mask.dim() == 3:
                        object_mask = object_mask.max(dim=1)[0]  # [B, N]
                    mask_expanded = object_mask.unsqueeze(1).unsqueeze(-1)
                    mask_expanded = mask_expanded.expand(-1, seq_len, -1, hidden)
                    object_embeddings = object_embeddings * mask_expanded
                    obj_sum = object_embeddings.sum(dim=2)
                    obj_count = object_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                    seq_features = obj_sum / obj_count.unsqueeze(-1)
                else:
                    seq_features = object_embeddings.mean(dim=2)
                features = seq_features.mean(dim=1)
            elif object_embeddings.dim() == 3:
                if object_mask is not None:
                    if object_mask.dim() == 2:
                        mask_expanded = object_mask.unsqueeze(-1)
                    else:
                        mask_expanded = object_mask[:, 0, :].unsqueeze(-1)
                    object_embeddings = object_embeddings * mask_expanded
                    features = object_embeddings.sum(dim=1) / object_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                else:
                    features = object_embeddings.mean(dim=1)
            else:
                features = object_embeddings
            
            return features
    
    def extract_object_embeddings(self, physics_states, object_mask):
        """Extract per-object embeddings BEFORE pooling (for object masking).

        Returns:
            object_embeddings: [B, N, D] per-object embeddings
            flat_mask: [B, N] 2D object mask
        """
        if self.training and self.input_noise > 0:
            noise = torch.randn_like(physics_states) * self.input_noise
            physics_states = physics_states + noise

        with torch.no_grad():
            is_sequence = physics_states.dim() == 4

            if is_sequence:
                batch_size, seq_len, num_objects, state_dim = physics_states.shape
                velocity = physics_states[:, 1:] - physics_states[:, :-1]
                zero_velocity = torch.zeros(
                    batch_size, 1, num_objects, state_dim,
                    device=physics_states.device, dtype=physics_states.dtype
                )
                velocity_padded = torch.cat([zero_velocity, velocity], dim=1)
                augmented_states = torch.cat([physics_states, velocity_padded], dim=-1)
            else:
                batch_size, num_objects, state_dim = physics_states.shape
                zero_velocity = torch.zeros_like(physics_states)
                augmented_states = torch.cat([physics_states, zero_velocity], dim=-1)

            object_embeddings = self.physics_model.encode_physics(augmented_states, object_mask)

            # Collapse to [B, N, D] if temporal
            if object_embeddings.dim() == 4:
                object_embeddings = object_embeddings.mean(dim=1)  # avg over time

            # Flatten mask to [B, N]
            flat_mask = object_mask
            if flat_mask is not None and flat_mask.dim() == 3:
                flat_mask = flat_mask.max(dim=1)[0]

        return object_embeddings, flat_mask

    def compute_object_masking_loss(self, physics_states, object_mask):
        """Compute ALOE-style self-supervised object masking loss.

        Only active when self.use_object_masking is True and model is training.

        Returns:
            loss: scalar tensor (0.0 if disabled or not training)
        """
        if not self.use_object_masking or not self.training:
            return torch.tensor(0.0, device=physics_states.device)

        object_embeddings, flat_mask = self.extract_object_embeddings(
            physics_states, object_mask
        )
        # Gradient flows through masking head only (embeddings are detached inside head)
        loss, _, _ = self.object_masking_head(object_embeddings, flat_mask)
        return loss

    def create_prefix_tokens(self, physics_features):
        """Convert physics features to LLM prefix tokens."""
        batch_size = physics_features.size(0)
        adapted = self.adapter(physics_features)
        prefix_tokens = adapted.view(batch_size, self.num_prefix_tokens, self.llm_dim)
        return prefix_tokens

    def create_prefix_tokens_per_object(self, physics_states, object_mask):
        """Per-object prefix path (used when ``tokens_per_object > 0``).

        Produces a prefix of shape ``[B, max_objects * K, llm_dim]`` by
        applying a shared MLP to each per-object encoder embedding and
        emitting ``K = self.tokens_per_object`` tokens per object. Padded
        slots (mask == 0) are zeroed, and the returned attention-mask has
        ``1`` for real object-slots' tokens and ``0`` for padded slots so
        the LLM's attention ignores them.

        Motivation: the encoder OOD probe
        (``clevrer_benchmark/results/encoder_ood_probing_v2_surface.json``)
        found that per-object shape info (92% linear-probe accuracy on
        ``encoder_per_obj``) collapses to 65% mean-AUC at the mean-pooled
        ``prefix_pool``. Mean-pool followed by a single MLP destroys
        per-object identity. Emitting tokens per object preserves it.

        Returns:
            prefix_tokens: ``[B, N*K, llm_dim]``
            prefix_mask:   ``[B, N*K]`` attention mask (1.0 for real, 0.0
                           for padded-object slots)
        """
        if self.tokens_per_object <= 0:
            raise RuntimeError(
                "create_prefix_tokens_per_object() requires "
                "tokens_per_object > 0 (got {})".format(self.tokens_per_object)
            )
        object_embeddings, flat_mask = self.extract_object_embeddings(
            physics_states, object_mask
        )
        B, N, D = object_embeddings.shape
        K = self.tokens_per_object
        adapted = self.adapter_per_object(object_embeddings)       # [B, N, K*llm_dim]
        per_obj_tokens = adapted.view(B, N, K, self.llm_dim)
        per_obj_tokens = per_obj_tokens + self.per_object_slot_bias
        if flat_mask is not None:
            # Zero out padded-object tokens so their embeddings don't leak
            # information into the LLM even if a downstream consumer
            # forgets the attention mask.
            mask_e = flat_mask.unsqueeze(-1).unsqueeze(-1)          # [B, N, 1, 1]
            per_obj_tokens = per_obj_tokens * mask_e
            prefix_mask = flat_mask.unsqueeze(-1).expand(-1, -1, K).reshape(B, N * K)
        else:
            prefix_mask = torch.ones(
                B, N * K, device=object_embeddings.device,
                dtype=object_embeddings.dtype,
            )
        prefix_tokens = per_obj_tokens.view(B, N * K, self.llm_dim)
        return prefix_tokens, prefix_mask

    def predict_numerical(self, physics_states, object_mask) -> Dict[str, torch.Tensor]:
        """
        Predict numerical physics values directly from physics features.
        
        Returns dict with keys:
            distance, speed, time_to_collision, kinetic_energy, momentum, object_count
        """
        physics_features = self.extract_physics_features(physics_states, object_mask)
        outputs = self.numerical_head(physics_features)
        
        return {
            "distance": outputs[:, 0],
            "speed": outputs[:, 1],
            "time_to_collision": outputs[:, 2],
            "kinetic_energy": outputs[:, 3],
            "momentum": outputs[:, 4],
            "object_count": outputs[:, 5],
        }
    
    def forward(
        self,
        physics_states,
        object_mask,
        question_text,
        max_length=50,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1,
        logits_processor=None,
    ):
        """Forward pass: Physics + Question -> Answer (categorical via LLM).

        Default generation is stochastic (``do_sample=True``, T=0.7, top_p=0.9)
        for backward compatibility. Pass ``do_sample=False`` for greedy decoding
        (deterministic) or ``num_beams>1`` for beam search.
        """
        batch_size = physics_states.size(0)
        device = physics_states.device
        
        physics_features = self.extract_physics_features(physics_states, object_mask)
        prefix_tokens = self.create_prefix_tokens(physics_features)
        
        prompted_questions = [q + " Answer:" for q in question_text]
        question_tokens = self.tokenizer(
            prompted_questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        
        question_embeds = self._get_input_embeddings()(question_tokens.input_ids)
        # Phase 7 dtype fix: when the LLM is loaded in bf16 (Qwen2.5 default)
        # but the adapter MLP returns fp32, ``cat`` promotes to fp32 and the
        # downstream LoRA-wrapped Linear layers raise ``mat1 and mat2 must
        # have the same dtype, but got Float and BFloat16``. Training masked
        # this with ``torch.amp.autocast``; eval doesn't autocast.
        prefix_tokens = prefix_tokens.to(dtype=question_embeds.dtype)
        combined_embeds = torch.cat([prefix_tokens, question_embeds], dim=1)

        prefix_mask = torch.ones(batch_size, self.num_prefix_tokens, device=device)
        combined_mask = torch.cat([prefix_mask, question_tokens.attention_mask], dim=1)
        
        gen_kwargs = dict(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            num_beams=num_beams,
        )
        # HF emits warnings if temperature/top_p are set without sampling; omit them.
        if do_sample:
            gen_kwargs['temperature'] = temperature
            gen_kwargs['top_p'] = top_p
        if logits_processor is not None:
            from transformers import LogitsProcessorList
            if not isinstance(logits_processor, LogitsProcessorList):
                logits_processor = LogitsProcessorList([logits_processor])
            gen_kwargs['logits_processor'] = logits_processor
        outputs = self.llm.generate(**gen_kwargs)
        
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answers = []
        for text in generated_text:
            if "Answer:" in text:
                answer = text.split("Answer:")[-1].strip()
            else:
                answer = text.strip()
            answers.append(answer)
        
        return answers

    def _compute_question_lengths(self, question_text: List[str]) -> List[int]:
        question_lengths: List[int] = []
        for q in question_text:
            q_with_marker = q + " Answer:"
            q_ids = self.tokenizer.encode(q_with_marker, add_special_tokens=False)
            question_lengths.append(len(q_ids))
        return question_lengths

    def score_answer_candidates(
        self,
        physics_states: torch.Tensor,
        object_mask: torch.Tensor,
        question_text: List[str],
        answer_candidates: List[List[str]],
        max_length: int = 128
    ) -> torch.Tensor:
        """Return scores for each candidate answer (higher is better).

        Scoring is based on average negative log-likelihood over answer tokens.
        """
        batch_size = physics_states.size(0)
        device = physics_states.device

        if len(question_text) != batch_size:
            raise ValueError("question_text must have same length as batch")
        if len(answer_candidates) != batch_size:
            raise ValueError("answer_candidates must have same length as batch")

        num_choices = [len(c) for c in answer_candidates]
        if not num_choices or any(n == 0 for n in num_choices):
            raise ValueError("Each sample must have >=1 answer candidate")
        if len(set(num_choices)) != 1:
            raise ValueError("All samples in a batch must have the same number of choices")
        k = num_choices[0]

        physics_features = self.extract_physics_features(physics_states, object_mask)
        prefix_tokens = self.create_prefix_tokens(physics_features)

        expanded_prefix = prefix_tokens.unsqueeze(1).expand(-1, k, -1, -1).reshape(
            batch_size * k, self.num_prefix_tokens, self.llm_dim
        )

        expanded_questions: List[str] = []
        expanded_full_text: List[str] = []
        for q, choices in zip(question_text, answer_candidates):
            for a in choices:
                expanded_questions.append(q)
                expanded_full_text.append(q + " Answer: " + a)

        tokens = self.tokenizer(
            expanded_full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        question_lengths = self._compute_question_lengths(expanded_questions)

        text_embeds = self._get_input_embeddings()(tokens.input_ids)
        # Phase 7 dtype fix (see V2.forward concat above for rationale).
        expanded_prefix = expanded_prefix.to(dtype=text_embeds.dtype)
        combined_embeds = torch.cat([expanded_prefix, text_embeds], dim=1)

        prefix_mask = torch.ones(batch_size * k, self.num_prefix_tokens, device=device)
        combined_mask = torch.cat([prefix_mask, tokens.attention_mask], dim=1)

        labels = tokens.input_ids.clone()
        for i in range(batch_size * k):
            q_len = question_lengths[i]
            labels[i, :q_len] = -100

        prefix_labels = torch.full(
            (batch_size * k, self.num_prefix_tokens),
            -100,
            dtype=torch.long,
            device=device
        )
        labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=labels,
            return_dict=True
        )

        # Compute per-sample loss manually since outputs.loss is averaged
        logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute cross-entropy per token
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            shift_labels.view(-1)
        ).view(batch_size * k, -1)
        
        # Mask out padding and question tokens (where label is -100)
        valid_mask = (shift_labels != -100).float()
        per_sample_loss = (per_token_loss * valid_mask).sum(dim=-1) / (valid_mask.sum(dim=-1) + 1e-8)
        
        # Higher score = lower loss = better answer
        scores = (-per_sample_loss).reshape(batch_size, k)
        return scores

    def select_answer_from_choices(
        self,
        physics_states: torch.Tensor,
        object_mask: torch.Tensor,
        question_text: List[str],
        choices: List[List[str]],
        max_length: int = 128
    ) -> List[str]:
        scores = self.score_answer_candidates(
            physics_states=physics_states,
            object_mask=object_mask,
            question_text=question_text,
            answer_candidates=choices,
            max_length=max_length
        )
        best = torch.argmax(scores, dim=-1).tolist()
        return [c[i] for c, i in zip(choices, best)]

    def compute_multiple_choice_loss(
        self,
        physics_states: torch.Tensor,
        object_mask: torch.Tensor,
        question_text: List[str],
        choices: List[List[str]],
        correct_choice_idx: torch.Tensor,
        max_length: int = 128,
        label_smoothing: float = 0.1
    ) -> Tuple[torch.Tensor, int, int]:
        # Route through adapter so gradient trains it:
        #   physics -> adapter -> prefix_tokens -> pool -> mcq_head -> cross_entropy
        device = physics_states.device
        batch_size = len(question_text)
        k = len(choices[0])

        # Physics -> adapter -> prefix tokens (gradient flows through adapter)
        physics_features = self.extract_physics_features(physics_states, object_mask)
        prefix_tokens = self.create_prefix_tokens(physics_features)  # [B, num_prefix, llm_dim]
        prefix_context = prefix_tokens.mean(dim=1)  # [B, llm_dim]

        # Encode question and choices via frozen LLM (text encoder only, no grad).
        # ``_get_base_model()`` returns ``llm.transformer`` on GPT-2 and
        # ``llm.model`` on Qwen/LLaMA, both of which produce contextual
        # ``last_hidden_state`` outputs that ``encode_texts`` mean-pools.
        base_text_encoder = self._get_base_model()
        question_embed = encode_texts(
            question_text, self.tokenizer, base_text_encoder, device, max_length=max_length
        )
        flat_choices = [c for sample_choices in choices for c in sample_choices]
        choice_embed_flat = encode_texts(
            flat_choices, self.tokenizer, base_text_encoder, device, max_length=max_length
        )
        choice_embeds = choice_embed_flat.view(batch_size, k, -1)

        # Score choices against adapter output (not raw physics)
        scores = self.mcq_head(prefix_context, question_embed, choice_embeds)

        loss = F.cross_entropy(scores, correct_choice_idx.to(device), label_smoothing=label_smoothing)
        preds = torch.argmax(scores, dim=-1)
        correct = (preds == correct_choice_idx.to(device)).sum().item()
        total = len(question_text)
        return loss, correct, total

    def _is_binary_question(self, question_text: str) -> bool:
        """Detect if a question expects a yes/no answer."""
        q_lower = question_text.lower().strip()
        binary_patterns = [
            q_lower.startswith("will "),
            q_lower.startswith("is "),
            q_lower.startswith("are "),
            q_lower.startswith("does "),
            q_lower.startswith("do "),
            q_lower.startswith("can "),
            q_lower.startswith("could "),
            q_lower.startswith("would "),
            q_lower.startswith("has "),
            q_lower.startswith("have "),
            q_lower.startswith("did "),
            "yes or no" in q_lower,
            "true or false" in q_lower,
        ]
        return any(binary_patterns)

    def answer_question(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        question_text: str,
        question_type: str = "",
        target_objects: Optional[List[int]] = None,
        choices: Optional[List[str]] = None,
        max_length: int = 30
    ) -> str:
        if not choices and self._is_binary_question(question_text):
            choices = ["yes", "no"]
        
        if choices:
            out = self.select_answer_from_choices(
                physics_states=states,
                object_mask=masks,
                question_text=[question_text],
                choices=[choices],
                max_length=128
            )
            return out[0]

        out = self.forward(states, masks, [question_text], max_length=max_length)
        return out[0] if out else ""
    
    def answer_clevrer(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        question_text: str,
        question_type: Optional[str] = None,
        choices: Optional[List[str]] = None
    ) -> str:
        """Unified CLEVRER QA with hybrid routing (descriptive->head, others->LLM)."""
        # Auto-detect question type if not provided
        if question_type is None:
            category = classify_clevrer_question(question_text)
            question_type = category.value
        
        # Route to appropriate head
        if question_type == "descriptive":
            # Use multi-head classification
            answers = self.answer_descriptive(states, masks, [question_text])
            return answers[0] if answers else ""
        
        else:
            # Use LLM for reasoning questions (explanatory, predictive, counterfactual)
            if choices:
                # Multiple choice - use likelihood scoring
                out = self.select_answer_from_choices(
                    physics_states=states,
                    object_mask=masks,
                    question_text=[question_text],
                    choices=[choices],
                    max_length=128
                )
                return out[0]
            else:
                # Open-ended - use LLM generation
                return self.answer_question(
                    states=states,
                    masks=masks,
                    question_text=question_text,
                    choices=choices
                )
    
    def answer_clevrer_batch(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        question_texts: List[str],
        question_types: Optional[List[str]] = None,
        choices_list: Optional[List[List[str]]] = None
    ) -> List[str]:
        """Batch CLEVRER QA with hybrid routing. Groups by type for efficiency."""
        batch_size = len(question_texts)
        
        # Auto-detect types if not provided
        if question_types is None:
            question_types = [classify_clevrer_question(q).value for q in question_texts]
        
        if choices_list is None:
            choices_list = [None] * batch_size
        
        # Group by type for efficient processing
        answers = [""] * batch_size
        
        # Process descriptive questions together
        desc_indices = [i for i, t in enumerate(question_types) if t == "descriptive"]
        if desc_indices:
            desc_questions = [question_texts[i] for i in desc_indices]
            # For descriptive, we need matching states - assume same states for batch
            desc_answers = self.answer_descriptive(states, masks, desc_questions)
            for i, idx in enumerate(desc_indices):
                answers[idx] = desc_answers[i]
        
        # Process reasoning questions (explanatory, predictive, counterfactual)
        reasoning_indices = [i for i, t in enumerate(question_types) if t != "descriptive"]
        for idx in reasoning_indices:
            q = question_texts[idx]
            choices = choices_list[idx]
            
            if choices:
                out = self.select_answer_from_choices(
                    physics_states=states,
                    object_mask=masks,
                    question_text=[q],
                    choices=[choices],
                    max_length=128
                )
                answers[idx] = out[0]
            else:
                answers[idx] = self.answer_question(
                    states=states,
                    masks=masks,
                    question_text=q,
                    choices=choices
                )
        
        return answers
    
    def compute_loss(self, physics_states, object_mask, question_text, answer_text):
        """Compute LLM causal-LM loss with optional label smoothing."""
        batch_size = physics_states.size(0)
        device = physics_states.device
        
        physics_features = self.extract_physics_features(physics_states, object_mask)
        prefix_tokens = self.create_prefix_tokens(physics_features)
        
        full_text = [q + " Answer: " + a for q, a in zip(question_text, answer_text)]
        tokens = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        
        question_lengths = self._compute_question_lengths(question_text)
        
        text_embeds = self._get_input_embeddings()(tokens.input_ids)
        # Phase 7 dtype fix (see V2.forward concat above for rationale).
        prefix_tokens = prefix_tokens.to(dtype=text_embeds.dtype)
        combined_embeds = torch.cat([prefix_tokens, text_embeds], dim=1)

        prefix_mask = torch.ones(batch_size, self.num_prefix_tokens, device=device)
        combined_mask = torch.cat([prefix_mask, tokens.attention_mask], dim=1)
        
        labels = tokens.input_ids.clone()
        for i in range(batch_size):
            q_len = question_lengths[i]
            labels[i, :q_len] = -100
        
        prefix_labels = torch.full(
            (batch_size, self.num_prefix_tokens),
            -100,
            dtype=torch.long,
            device=device
        )
        labels = torch.cat([prefix_labels, labels], dim=1)
        
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=labels if self.label_smoothing == 0 else None,
            return_dict=True
        )
        
        # Compute loss — with optional label smoothing
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = shift_labels != -100

        if self.label_smoothing > 0:
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100, label_smoothing=self.label_smoothing
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            loss = outputs.loss

        # Token-level accuracy on answer portion
        if shift_mask.any():
            preds = shift_logits.argmax(dim=-1)
            token_correct = ((preds == shift_labels) & shift_mask).sum().item()
            token_total = shift_mask.sum().item()
        else:
            token_correct = 0
            token_total = 0
        
        return loss, token_correct, token_total
    
    def compute_numerical_loss(
        self,
        physics_states,
        object_mask,
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute MSE loss for numerical predictions."""
        predictions = self.predict_numerical(physics_states, object_mask)
        
        total_loss = 0.0
        count = 0
        
        for key in targets:
            if key in predictions:
                pred = predictions[key]
                target = targets[key].to(pred.device)
                loss = F.mse_loss(pred, target)
                total_loss += loss
                count += 1
        
        return total_loss / max(count, 1)
    
    def answer_descriptive(
        self,
        physics_states: torch.Tensor,
        object_mask: torch.Tensor,
        question_text: List[str]
    ) -> List[str]:
        """Answer descriptive questions via multi-head classification."""
        physics_features = self.extract_physics_features(physics_states, object_mask)
        return self.descriptive_head.predict_batch(physics_features, question_text)
    
    def compute_descriptive_loss(
        self,
        physics_states: torch.Tensor,
        object_mask: torch.Tensor,
        question_text: List[str],
        answer_text: List[str],
        label_smoothing: float = 0.1
    ) -> torch.Tensor:
        """Compute cross-entropy loss for descriptive question classification."""
        physics_features = self.extract_physics_features(physics_states, object_mask)
        return self.descriptive_head.compute_loss(
            physics_features, question_text, answer_text, label_smoothing
        )
    
    def compute_combined_loss(
        self,
        physics_states,
        object_mask,
        question_text,
        answer_text,
        choices=None,
        correct_choice_idx=None,
        numerical_targets: Optional[Dict[str, torch.Tensor]] = None,
        categorical_weight: float = 1.0,
        numerical_weight: float = 1.0,
        descriptive_weight: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss — LLM causal-LM loss for all reasoning samples.
        
        Routing:
        1. Descriptive questions -> descriptive_head classification
        2. All other questions (MCQ or not) -> LLM causal-LM loss on answer text
           This trains the adapter via prefix tuning (the proven 69.3% approach).
        
        Args:
            choices: List[Optional[List[str]]] — unused for training (kept for API compat)
            correct_choice_idx: List[Optional[int]] — unused for training
        """
        device = physics_states.device
        batch_size = len(question_text)
        
        # ── Step 1: Split descriptive vs reasoning ──
        desc_idx = []
        llm_idx = []
        for i in range(batch_size):
            cat = classify_clevrer_question(question_text[i])
            if cat == CLEVRERQuestionCategory.DESCRIPTIVE:
                desc_idx.append(i)
            else:
                llm_idx.append(i)
        
        # ── Step 2: Descriptive head loss ──
        desc_loss = torch.tensor(0.0, device=device)
        if desc_idx:
            desc_loss = self.compute_descriptive_loss(
                physics_states[desc_idx], object_mask[desc_idx],
                [question_text[i] for i in desc_idx],
                [answer_text[i] for i in desc_idx]
            )
        
        # ── Step 3: LLM causal-LM loss for all reasoning questions ──
        # Gradient: loss -> LLM -> prefix_tokens -> adapter (trains adapter via prefix tuning)
        cat_loss = torch.tensor(0.0, device=device)
        token_correct = 0
        token_total = 0
        if llm_idx:
            cat_loss, token_correct, token_total = self.compute_loss(
                physics_states[llm_idx], object_mask[llm_idx],
                [question_text[i] for i in llm_idx],
                [answer_text[i] for i in llm_idx]
            )
        
        # ── Step 4: Numerical loss (skip if all targets are zero) ──
        num_loss = torch.tensor(0.0, device=device)
        if numerical_targets is not None and numerical_weight > 0:
            has_nonzero = any(
                v.abs().sum().item() > 1e-8
                for v in numerical_targets.values()
                if isinstance(v, torch.Tensor)
            )
            if has_nonzero:
                num_loss = self.compute_numerical_loss(physics_states, object_mask, numerical_targets)
        
        total_loss = (categorical_weight * cat_loss +
                      descriptive_weight * desc_loss +
                      numerical_weight * num_loss)

        return total_loss, {
            "categorical": cat_loss,
            "descriptive": desc_loss,
            "numerical": num_loss,
            "mcq_correct": token_correct,
            "mcq_total": token_total
        }

    def compute_contrastive_loss(
        self,
        physics_states: torch.Tensor,
        object_mask: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Compute physics-usage contrastive loss on prefix tokens.

        Forces prefix tokens from REAL physics to be DIFFERENT from prefix tokens
        from ZEROED physics for the same sample. This ensures the adapter actually
        uses physics information rather than ignoring it.

        Args:
            physics_states: [batch, seq, objects, state_dim]
            object_mask: [batch, objects] or [batch, seq, objects]
            temperature: Temperature for softmax (lower = harder separation)

        Returns:
            Contrastive loss scalar (higher when real/zero are too similar)
        """
        batch_size = physics_states.size(0)
        if batch_size < 1:
            return torch.tensor(0.0, device=physics_states.device)

        # Create zeroed physics states
        zero_states = torch.zeros_like(physics_states)

        # Extract features for real and zero physics
        real_features = self.extract_physics_features(physics_states, object_mask)
        zero_features = self.extract_physics_features(zero_states, object_mask)

        # Create prefix tokens
        real_prefix = self.create_prefix_tokens(real_features)  # [batch, num_prefix, llm_dim]
        zero_prefix = self.create_prefix_tokens(zero_features)

        # Flatten to single vector per sample
        real_flat = real_prefix.view(batch_size, -1)  # [batch, num_prefix * llm_dim]
        zero_flat = zero_prefix.view(batch_size, -1)

        # L2 normalize
        real_norm = F.normalize(real_flat, p=2, dim=1)
        zero_norm = F.normalize(zero_flat, p=2, dim=1)

        # Compute cosine similarity between real and zero for each sample
        # We want this to be LOW (different outputs for different inputs)
        cos_sim = (real_norm * zero_norm).sum(dim=1)  # [batch]

        # Always-active loss: maps cos_sim [-1,1] -> loss [0,1]
        # Loss=0 when cos_sim=-1 (fully separated), loss=1 when cos_sim=1 (identical)
        # Unlike hinge loss, this is ALWAYS active — no dead zone at initialization
        loss = (1 + cos_sim).mean() / 2

        return loss

    def compute_combined_loss_with_contrastive(
        self,
        physics_states,
        object_mask,
        question_text,
        answer_text,
        choices=None,
        correct_choice_idx=None,
        numerical_targets: Optional[Dict[str, torch.Tensor]] = None,
        categorical_weight: float = 1.0,
        numerical_weight: float = 1.0,
        descriptive_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.07
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss INCLUDING contrastive loss on prefix tokens.

        The contrastive loss forces the adapter to produce distinguishable
        prefix tokens for different physics scenes, preventing collapse.
        """
        # Get base losses
        base_loss, loss_dict = self.compute_combined_loss(
            physics_states, object_mask, question_text, answer_text,
            choices=choices,
            correct_choice_idx=correct_choice_idx,
            numerical_targets=numerical_targets,
            categorical_weight=categorical_weight,
            numerical_weight=numerical_weight,
            descriptive_weight=descriptive_weight
        )

        # Add contrastive loss
        contrastive_loss = self.compute_contrastive_loss(
            physics_states, object_mask, temperature=contrastive_temperature
        )

        total_loss = base_loss + contrastive_weight * contrastive_loss
        loss_dict["contrastive"] = contrastive_loss

        # Add object masking loss (if enabled)
        if self.use_object_masking and self.training:
            obj_mask_loss = self.compute_object_masking_loss(
                physics_states, object_mask
            )
            total_loss = total_loss + self.object_masking_weight * obj_mask_loss
            loss_dict["object_masking"] = obj_mask_loss

        return total_loss, loss_dict

    # -----------------------------------------------------------------
    # LoRA management
    # -----------------------------------------------------------------

    def apply_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        target_modules: Tuple[str, ...] = ('c_attn', 'c_proj'),
    ):
        """Apply LoRA adapters to specified Conv1D modules in the LLM.

        Args:
            rank: LoRA rank (lower = fewer params, higher = more capacity).
            alpha: LoRA scaling factor.
            dropout: Dropout on LoRA path.
            target_modules: Which Conv1D layer names to wrap (e.g. 'c_attn', 'c_proj').
        """
        if hasattr(self, '_lora_layers') and self._lora_layers:
            print("[LoRA] Already applied — call remove_lora() first to re-apply")
            return

        self._lora_layers: List[Tuple[nn.Module, str, Conv1D]] = []

        for name, module in self.llm.named_modules():
            if any(name.endswith(t) for t in target_modules) and isinstance(module, Conv1D):
                parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
                parent = self.llm.get_submodule(parent_name) if parent_name else self.llm
                lora_module = LoRAConv1D(module, rank=rank, alpha=alpha, dropout=dropout)
                # Align new lora_A/lora_B with the wrapped Conv1D's device/dtype
                # so apply_lora() is safe to call after adapter.to(device). Without
                # this, lora_A/lora_B are created on CPU while self.original lives
                # on CUDA, which raises a device-mismatch error on the first forward.
                ref_weight = module.weight
                lora_module = lora_module.to(device=ref_weight.device, dtype=ref_weight.dtype)
                setattr(parent, child_name, lora_module)
                self._lora_layers.append((parent, child_name, module))

        lora_params = sum(
            p.numel() for n, p in self.llm.named_parameters() if 'lora_' in n
        )
        print(f"[LoRA] Applied rank={rank}, alpha={alpha} to {len(self._lora_layers)} layers")
        print(f"[LoRA] Trainable LoRA params: {lora_params:,}")

    def merge_lora(self):
        """Merge LoRA weights into base Conv1D layers and remove wrappers."""
        if not hasattr(self, '_lora_layers') or not self._lora_layers:
            print("[LoRA] Nothing to merge")
            return
        for parent, child_name, original in self._lora_layers:
            lora_mod = getattr(parent, child_name)
            lora_mod.merge_weights()
            setattr(parent, child_name, lora_mod.original)
        print(f"[LoRA] Merged and removed {len(self._lora_layers)} LoRA layers")
        self._lora_layers = []

    def remove_lora(self):
        """Remove LoRA wrappers WITHOUT merging (discards LoRA weights)."""
        if not hasattr(self, '_lora_layers') or not self._lora_layers:
            return
        for parent, child_name, original in self._lora_layers:
            setattr(parent, child_name, original)
        print(f"[LoRA] Removed {len(self._lora_layers)} LoRA layers (not merged)")
        self._lora_layers = []

    @property
    def has_lora(self) -> bool:
        return hasattr(self, '_lora_layers') and len(self._lora_layers) > 0

    # -----------------------------------------------------------------
    # Training phase management
    # -----------------------------------------------------------------

    def _unfreeze(self, *modules):
        """Enable training for given modules."""
        for m in modules:
            m.train()
            for p in m.parameters():
                p.requires_grad = True

    def set_training_phase(self, phase, lora_rank: int = 8, lora_alpha: float = 16.0):
        """Set training phase for gradual unfreezing.

        Phases:
            'adapter'          Phase 1: adapter + numerical head (LLM frozen)
            'lora'             Phase 2: adapter + LoRA on LLM attention + numerical head
            'llm_head'         (legacy) adapter + LLM lm_head + numerical head
            'llm_full'         (legacy) adapter + full LLM + numerical head
            'numerical_only'   numerical head only
            'descriptive_only' descriptive head only
            'end_to_end'       everything trainable
        """
        # Freeze everything first. Heads may be absent depending on the
        # construction flags, so guard with hasattr to avoid AttributeError.
        all_modules = [self.physics_model, self.adapter, self.numerical_head, self.llm]
        if hasattr(self, 'descriptive_head'):
            all_modules.append(self.descriptive_head)
        if hasattr(self, 'mcq_head'):
            all_modules.append(self.mcq_head)
        if self.use_object_masking:
            all_modules.append(self.object_masking_head)
        for m in all_modules:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        # Adapter-training phases must also unfreeze descriptive_head, because
        # compute_combined_loss routes descriptive samples through it. If those
        # samples appear in a batch while descriptive_head is frozen and the
        # categorical/numerical paths contribute no grad (e.g. batch is
        # all-descriptive with zero numerical targets), total_loss ends up with
        # grad_fn=None and .backward() raises 'element 0 ... does not require grad'.
        adapter_heads = [self.adapter, self.numerical_head]
        if hasattr(self, 'descriptive_head'):
            adapter_heads.append(self.descriptive_head)

        # Handle LoRA phase specially
        if phase == 'lora':
            if not self.has_lora:
                self.apply_lora(rank=lora_rank, alpha=lora_alpha)
            # Unfreeze adapter + numerical + descriptive heads + LoRA params
            omh = [self.object_masking_head] if self.use_object_masking else []
            self._unfreeze(*adapter_heads, *omh)
            # Unfreeze only LoRA parameters in LLM (keep base weights frozen)
            lora_count = 0
            for name, param in self.llm.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                    lora_count += 1
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            desc_tag = " + descriptive" if hasattr(self, 'descriptive_head') else ""
            print(f"[PHASE] Training adapter + LoRA + numerical{desc_tag}"
                  + (" + object masking" if omh else ""))
            print(f"[PHASE] Trainable params: {trainable:,} ({lora_count} LoRA tensors)")
            return

        # Standard phases
        omh = [self.object_masking_head] if self.use_object_masking else []
        desc_tag = " + descriptive" if hasattr(self, 'descriptive_head') else ""
        # For end_to_end, optionally include mcq_head and descriptive_head.
        e2e_modules = [self.physics_model] + adapter_heads + [self.llm]
        if hasattr(self, 'mcq_head'):
            e2e_modules.append(self.mcq_head)
        phase_map = {
            'adapter':         (adapter_heads + omh,
                                f"Training adapter + numerical{desc_tag}"
                                + (" + object masking" if omh else "")),
            'llm_head':        ([self.llm.lm_head] + adapter_heads + omh,
                                f"Training adapter + LLM head + numerical{desc_tag}"
                                + (" + object masking" if omh else "")),
            'llm_full':        ([self.llm] + adapter_heads + omh,
                                f"Training adapter + full LLM + numerical{desc_tag}"
                                + (" + object masking" if omh else "")),
            'numerical_only':  ([self.numerical_head],
                                "Training numerical head only"),
            'descriptive_only':(
                                [self.descriptive_head] if hasattr(self, 'descriptive_head') else [],
                                "Training descriptive head only"),
            'end_to_end':      (e2e_modules + omh,
                                "Training end-to-end"),
        }
        if phase not in phase_map:
            raise ValueError(f"Unknown phase: {phase}. Options: {list(phase_map.keys()) + ['lora']}")
        modules, desc = phase_map[phase]
        print(f"[PHASE] {desc}")
        self._unfreeze(*modules)


def create_adapter_v2(physics_model, **kwargs):
    """Factory function to create V2 adapter with DistilGPT-2 (edge-optimized)."""
    return PhysicsLLMAdapterV2(
        physics_model=physics_model,
        **kwargs
    )
