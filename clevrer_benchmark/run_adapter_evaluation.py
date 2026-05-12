"""
CLEVRER Benchmark Evaluation with Physics-LLM Adapter

Evaluates the trained PhysicsLLMAdapterV2 on CLEVRER benchmark questions.
Uses the adapter's question answering capabilities for physics reasoning.

Usage:
    python run_adapter_evaluation.py --clevrer_dir $CLEVRER_DIR --max_scenes 100
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from clevrer_benchmark.scene_converter import (
    load_clevrer_scene,
    clevrer_scene_to_state_tensor
)
from clevrer_benchmark.question_mapper import (
    CLEVRERQuestionMapper,
    CLEVRERQuestionType
)


@dataclass
class EvaluationResult:
    """Results for a single question evaluation."""
    scene_id: str
    question_id: str
    question_text: str
    clevrer_type: str
    ground_truth: Any
    predicted: str
    correct: bool
    # Only populated when the question is MCQ (``choices`` non-empty on input).
    # Kept separate from the summary dict so the aggregate JSON stays small; a
    # sidecar JSONL consumes these for failure-analysis tooling.
    choices: Optional[List[Dict[str, Any]]] = None

    def to_detail_dict(self) -> Dict[str, Any]:
        correct_choices = []
        wrong_choices = []
        if self.choices:
            correct_choices = [c.get('choice', '') for c in self.choices if c.get('answer') == 'correct']
            wrong_choices = [c.get('choice', '') for c in self.choices if c.get('answer') == 'wrong']
        return {
            'scene_id': self.scene_id,
            'question_id': self.question_id,
            'question_text': self.question_text,
            'clevrer_type': self.clevrer_type,
            'ground_truth': self.ground_truth,
            'predicted': self.predicted,
            'correct': self.correct,
            'choices': self.choices,
            'correct_choices': correct_choices,
            'wrong_choices': wrong_choices,
        }


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    total_questions: int = 0
    correct_count: int = 0
    
    by_clevrer_type: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {'correct': 0, 'total': 0})
    )
    
    individual_results: List[EvaluationResult] = field(default_factory=list)
    
    def add_result(self, result: EvaluationResult):
        self.total_questions += 1
        self.correct_count += int(result.correct)
        self.by_clevrer_type[result.clevrer_type]['total'] += 1
        self.by_clevrer_type[result.clevrer_type]['correct'] += int(result.correct)
        self.individual_results.append(result)
    
    def get_accuracy(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.correct_count / self.total_questions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall': {
                'total_questions': self.total_questions,
                'correct': self.correct_count,
                'accuracy': self.get_accuracy()
            },
            'by_clevrer_type': {
                k: {
                    'total': v['total'],
                    'correct': v['correct'],
                    'accuracy': v['correct'] / max(1, v['total'])
                }
                for k, v in self.by_clevrer_type.items()
            }
        }


def load_adapter_model(
    adapter_checkpoint: str,
    physics_checkpoint: str,
    device: str = 'cuda'
):
    """Load a PhysicsLLMAdapterV2 / V3 model from a self-contained checkpoint.

    The adapter checkpoint embeds the physics encoder weights (under
    ``physics_model.*``) and adapter metadata. Phase 3 / Phase 4 mixed-format
    saves used ``PhysicsLLMAdapterV2`` with a scene-level (``adapter.*``) MLP;
    Phase B (scene-text injection) and Phase C (per-object prefix) saves use
    ``PhysicsLLMAdapterV3`` with either the same ``adapter.*`` MLP or a new
    ``adapter_per_object.*`` MLP.

    Shape inference is done from the state_dict so no external config file
    is needed. The optional ``config`` sidecar written by the Phase B/C
    training notebook provides the scene-text + per-object flags so we can
    reconstruct the live ``PhysicsLLMAdapterV3`` instance with those flags
    toggled on (they affect only training-loss code paths + generation, but
    the class still needs to know about them to route through the right
    prefix-construction method in ``V3.forward``).
    """
    from physics_llm_adapter.adapter_v2 import PhysicsLLMAdapterV2
    from physics_former.training.models.physics_former_full import FullPhysicsFormer

    adapter_ckpt = torch.load(adapter_checkpoint, map_location=device, weights_only=False)
    # Accept both historical conventions: Phase 3 / Phase 4 mixed-format save
    # under 'model_state_dict'; the Phase 4 free-form notebook (and the V3
    # warm-start path in train_adapter_v2.py) uses 'adapter_state_dict'.
    # Fallback treats the whole object as a raw state dict (no wrapper).
    adapter_sd = adapter_ckpt.get('model_state_dict',
                                  adapter_ckpt.get('adapter_state_dict',
                                                   adapter_ckpt))

    # Infer physics model config from adapter checkpoint (it contains embedded physics model)
    hidden_dim = adapter_sd['physics_model.encoder.object_encoder.4.weight'].shape[0]
    ff_dim = adapter_sd['physics_model.transformer_layers.0.ff.0.weight'].shape[0]
    num_layers = len([k for k in adapter_sd.keys()
                      if 'physics_model.transformer_layers' in k and '.attention.q_proj.weight' in k])
    num_heads = adapter_sd['physics_model.transformer_layers.0.attention.attention_bias_net.2.weight'].shape[0]
    max_count = adapter_sd['physics_model.counting_head_classification.6.weight'].shape[0]
    max_objects = max_count - 1

    schema_key = 'physics_model.schema_classifier.3.weight'
    num_schema_classes = adapter_sd[schema_key].shape[0] if schema_key in adapter_sd else 37

    # Infer adapter-side config so older checkpoints load cleanly. Phase 7
    # writes ``llm_name`` (e.g. 'Qwen/Qwen2.5-1.5B-Instruct') into the recipe
    # sidecar so we honor that first; pre-Phase-7 checkpoints fall through
    # to the legacy top-level ``llm_name`` field, then to 'distilgpt2'.
    _early_recipe = adapter_ckpt.get('config', {}) if isinstance(adapter_ckpt, dict) else {}
    llm_name = (_early_recipe.get('llm_name')
                or adapter_ckpt.get('llm_name', 'distilgpt2'))
    has_mcq_head = any(k.startswith('mcq_head.') for k in adapter_sd)
    has_descriptive_head = any(k.startswith('descriptive_head.') for k in adapter_sd)

    # Detect Phase C (per-object prefix) from state_dict. The Phase C adapter
    # has ``adapter_per_object.*`` keys and does NOT have a scene-level
    # ``adapter.*`` MLP (the two are mutually exclusive in V2.__init__).
    # Infer tokens_per_object from the final-Linear output dim:
    #   adapter_per_object output = llm_dim * tokens_per_object
    has_per_object_prefix = any(k.startswith('adapter_per_object.') for k in adapter_sd)
    tokens_per_object = 0
    if has_per_object_prefix:
        # Look up llm_dim from LLM_CONFIGS so Phase 7 (Qwen-1.5B = 1536)
        # works without an extra hardcoded entry. Fallback to 768 for
        # checkpoints with an llm_name that isn't in LLM_CONFIGS (which
        # would raise inside the adapter ctor anyway, but we want a clean
        # error message rather than an arithmetic exception here).
        llm_dim = PhysicsLLMAdapterV2.LLM_CONFIGS.get(
            llm_name, {'dim': 768}).get('dim', 768)
        per_obj_linear_keys = sorted(
            k for k in adapter_sd
            if k.startswith('adapter_per_object.') and k.endswith('.weight')
            and adapter_sd[k].ndim == 2  # 2D => Linear (not LayerNorm's 1D)
        )
        if per_obj_linear_keys:
            final_weight = adapter_sd[per_obj_linear_keys[-1]]
            out_dim = final_weight.shape[0]
            if out_dim % llm_dim != 0:
                raise RuntimeError(
                    f"adapter_per_object final output dim {out_dim} is not a "
                    f"multiple of llm_dim={llm_dim}; cannot infer tokens_per_object"
                )
            tokens_per_object = out_dim // llm_dim
        num_prefix_tokens = max_objects * tokens_per_object
        adapter_mlp_style = "deep"  # per-object always uses deep topology
    else:
        num_prefix_tokens = adapter_ckpt.get('num_prefix_tokens', 64)
        # Detect adapter MLP topology: "deep" has a Linear at index 8 of
        # the Sequential; "shallow" only uses indices 0/1/4/5.
        adapter_mlp_style = "deep" if 'adapter.8.weight' in adapter_sd else "shallow"

    # Detect LoRA tensors in the checkpoint. There are TWO LoRA backends in
    # this codebase, and the saved tensor layouts differ:
    #
    #   - GPT-2 in-house ``LoRAConv1D`` wrapper: saves under
    #     ``llm.transformer.h.<i>.attn.c_attn.lora_A`` / ``.lora_B`` (Parameter,
    #     not nested in a Linear). Keys end exactly in ``.lora_A``.
    #     Reconstruction: call ``adapter.apply_lora(rank, alpha)`` AFTER
    #     ctor and BEFORE load_state_dict.
    #
    #   - peft.LoraConfig (Phase 7+, LLaMA-family Qwen2.5): saves under
    #     ``base_model.model.model.layers.<i>.self_attn.q_proj.lora_A.default.weight``.
    #     Keys contain ``lora_A.`` followed by an adapter name segment.
    #     Reconstruction: pass ``use_lora=True`` into the V2/V3 ctor; the
    #     adapter will internally call ``peft.get_peft_model`` so the
    #     LoRA-suffixed keys exist on the live module before
    #     load_state_dict.
    #
    # We accept either (or neither) for backward compatibility.
    inhouse_lora_a_keys = [k for k in adapter_sd if k.endswith('.lora_A')]
    peft_lora_a_keys = [
        k for k in adapter_sd
        if '.lora_A.' in k and k.endswith('.weight')
        and not k.endswith('.lora_A')  # avoid double-counting in-house keys
    ]
    has_inhouse_lora = bool(inhouse_lora_a_keys)
    has_peft_lora = bool(peft_lora_a_keys)
    has_lora = has_inhouse_lora or has_peft_lora

    if has_inhouse_lora:
        # In-house Conv1D LoRA: lora_A shape is [nx, rank], so dim 1 is rank.
        lora_rank = adapter_sd[inhouse_lora_a_keys[0]].shape[1]
    elif has_peft_lora:
        # peft Linear LoRA: lora_A.default.weight shape is [rank, in_features],
        # so dim 0 is rank.
        lora_rank = adapter_sd[peft_lora_a_keys[0]].shape[0]
    else:
        lora_rank = None
    # Alpha is stored in the recipe sidecar as of Phase 7; legacy fallback
    # is the alpha=2*rank convention from the original GPT-2 LoRA notebook.
    lora_alpha = (
        float(adapter_ckpt.get('lora_alpha', 2 * lora_rank))
        if has_lora else None
    )

    # Read Phase B / Phase C / Phase 7 recipe flags from the saved config sidecar
    # (written by the training notebook's _ff_recipe() helper). Absent for
    # Phase 3 / Phase 4 mixed-format checkpoints -- fall back to defaults that
    # preserve V2-compatible behavior in that case.
    recipe = adapter_ckpt.get('config', {}) if isinstance(adapter_ckpt, dict) else {}
    recipe_inject_scene_text        = bool(recipe.get('inject_scene_text', False))
    recipe_scene_text_style         = str(recipe.get('scene_text_style', 'comma_list'))
    recipe_scene_text_include_color = bool(recipe.get('scene_text_include_color', True))
    recipe_scene_text_include_mat   = bool(recipe.get('scene_text_include_material', True))
    # Phase 7 LoRA + prefix-dropout knobs. Read from recipe; cross-check
    # against state_dict-detected LoRA below so we never silently disagree.
    recipe_use_lora             = bool(recipe.get('use_lora', has_lora))
    recipe_lora_rank            = int(recipe.get('lora_rank', lora_rank or 16))
    recipe_lora_alpha           = float(recipe.get('lora_alpha', lora_alpha or 32.0))
    recipe_lora_dropout         = float(recipe.get('lora_dropout', 0.05))
    recipe_lora_target_modules  = recipe.get('lora_target_modules', None)
    recipe_prefix_dropout       = float(recipe.get('prefix_dropout', 0.0))
    # Phase 8 descriptor-stripping knobs. Stripping is intentionally
    # disabled in adapter.eval() mode (it's a training-time regularizer
    # only), so passing these through to the V3 ctor is purely cosmetic
    # for eval -- it keeps the print banner accurate and means the
    # round-trip ``recipe['descriptor_strip_prob']`` value lives on the
    # reconstructed adapter for any future inference-time ablation.
    recipe_descriptor_strip_prob = float(recipe.get('descriptor_strip_prob', 0.0))
    recipe_descriptor_strip_seed = int(recipe.get('descriptor_strip_seed', 4747))
    # tokens_per_object may come from the recipe OR be inferred from the
    # state_dict; prefer state_dict inference (ground truth) but warn if they disagree.
    if recipe and 'tokens_per_object' in recipe:
        recipe_tpo = int(recipe['tokens_per_object'])
        if recipe_tpo != tokens_per_object:
            print(f"  [WARN] recipe.tokens_per_object={recipe_tpo} but state_dict "
                  f"suggests {tokens_per_object}. Using state_dict value.")

    # Pick class: if this checkpoint was saved by V3 (any Phase B/C flag set or
    # per-object prefix present), instantiate V3 so V3.forward dispatches the
    # variable prefix mask; otherwise fall back to V2 (Phase 3 / Phase 4 mixed).
    use_v3 = has_per_object_prefix or recipe_inject_scene_text or (
        recipe.get('adapter_class') == 'PhysicsLLMAdapterV3'
    )

    print(f"  Physics model config (from adapter): hidden_dim={hidden_dim}, num_layers={num_layers}, num_heads={num_heads}, ff_dim={ff_dim}, max_objects={max_objects}")
    _class_label = "V3" if use_v3 else "V2"
    _lora_backend = (
        'peft' if has_peft_lora else ('in-house' if has_inhouse_lora else None)
    )
    print(
        f"  Adapter config ({_class_label}): llm={llm_name}, "
        f"num_prefix_tokens={num_prefix_tokens}, mlp_style={adapter_mlp_style}, "
        f"mcq_head={has_mcq_head}, descriptive_head={has_descriptive_head}, "
        f"lora={has_lora}"
        + (f" (rank={lora_rank}, alpha={lora_alpha}, backend={_lora_backend})"
           if has_lora else "")
    )
    if has_per_object_prefix:
        print(f"  Per-object prefix: tokens_per_object={tokens_per_object} "
              f"(=> {num_prefix_tokens} total prefix tokens)")
    if recipe_inject_scene_text:
        print(f"  Scene-text injection: style={recipe_scene_text_style} "
              f"(eval scripts prepend at the call site, not inside forward)")
    if recipe_prefix_dropout > 0:
        print(f"  Prefix dropout (Phase 7+): p={recipe_prefix_dropout} "
              f"(stored on adapter for compute_loss; eval is unaffected)")
    if recipe_descriptor_strip_prob > 0:
        print(f"  Descriptor stripping (Phase 8): p={recipe_descriptor_strip_prob} "
              f"(seed={recipe_descriptor_strip_seed}; training-only, "
              f"eval is unaffected)")

    physics_model = FullPhysicsFormer(
        state_dim=35,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_objects=max_objects,
        dropout=0.1,
        num_schema_classes=num_schema_classes
    ).to(device)

    # Common ctor kwargs shared by V2 and V3. The peft LoRA path needs
    # ``freeze_llm=True`` so peft.get_peft_model can correctly mark only the
    # LoRA-suffixed parameters as trainable; the in-house Conv1D path doesn't
    # care because ``apply_lora()`` runs post-ctor and rewrites requires_grad
    # itself. ``freeze_llm=False`` is preserved for non-LoRA checkpoints
    # (Phase 3 default) so the eval semantics for those don't change.
    common_kwargs = dict(
        physics_model=physics_model,
        physics_dim=hidden_dim,
        llm_name=llm_name,
        num_prefix_tokens=num_prefix_tokens,
        freeze_physics=True,
        freeze_llm=(True if has_peft_lora else False),
        build_mcq_head=has_mcq_head,
        build_descriptive_head=has_descriptive_head,
        adapter_mlp_style=adapter_mlp_style,
        tokens_per_object=tokens_per_object,
    )
    # Phase 7 ctor knobs. ``use_lora=True`` triggers peft inside V2.__init__
    # so the LoRA-suffixed keys exist on the live module BEFORE load_state_dict.
    # We only set ``use_lora=True`` when the checkpoint actually contains peft
    # tensors -- forcing it on for in-house checkpoints would create double LoRA.
    if has_peft_lora:
        common_kwargs.update(dict(
            use_lora=True,
            lora_rank=recipe_lora_rank,
            lora_alpha=recipe_lora_alpha,
            lora_dropout=recipe_lora_dropout,
            lora_target_modules=recipe_lora_target_modules,
        ))

    if use_v3:
        from physics_llm_adapter.adapter_v3 import PhysicsLLMAdapterV3
        adapter = PhysicsLLMAdapterV3(
            **common_kwargs,
            include_choices_prob=float(recipe.get('include_choices_prob', 0.0)),
            mixed_format_seed=int(recipe.get('mixed_format_seed', 42)),
            answer_token_dropout=float(recipe.get('answer_token_dropout', 0.0)),
            force_llm_only_routing=bool(recipe.get('force_llm_only_routing', False)),
            inject_scene_text=recipe_inject_scene_text,
            scene_text_style=recipe_scene_text_style,
            scene_text_include_color=recipe_scene_text_include_color,
            scene_text_include_material=recipe_scene_text_include_mat,
            prefix_dropout=recipe_prefix_dropout,
            descriptor_strip_prob=recipe_descriptor_strip_prob,
            descriptor_strip_seed=recipe_descriptor_strip_seed,
        ).to(device)
    else:
        adapter = PhysicsLLMAdapterV2(**common_kwargs).to(device)

    if has_inhouse_lora:
        # GPT-2 in-house Conv1D LoRA: insert wrappers so the checkpoint's
        # ``lora_A`` / ``lora_B`` / ``original.*`` keys map to live params.
        # ``apply_lora()`` is device-aware after the adapter_v2 patch, so
        # calling it post ``.to(device)`` is safe. Skip when peft LoRA was
        # already applied during __init__ -- doing both would stack two
        # LoRA layers on the same modules.
        adapter.apply_lora(rank=lora_rank, alpha=lora_alpha)

    adapter.load_state_dict(adapter_sd)
    adapter.eval()

    print(f"Loaded adapter from {adapter_checkpoint}")
    print(f"  LLM: {llm_name}")
    print(f"  Training samples: {adapter_ckpt.get('training_samples', 'N/A')}")

    return adapter


def map_clevrer_to_adapter_question(
    question: Dict[str, Any],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Map CLEVRER question to adapter format."""
    q_text = question.get('question', '')
    q_type = question.get('question_type', 'descriptive')
    
    adapter_type = 'collision_prediction'
    if 'collide' in q_text.lower():
        adapter_type = 'collision_prediction'
    elif 'what color' in q_text.lower():
        adapter_type = 'object_position'
    elif 'what shape' in q_text.lower():
        adapter_type = 'object_position'
    elif 'how many' in q_text.lower():
        adapter_type = 'object_count'
    elif 'what caused' in q_text.lower() or 'responsible' in q_text.lower():
        adapter_type = 'causal_responsibility'
    elif 'will' in q_text.lower() and 'happen' in q_text.lower():
        adapter_type = 'trajectory_extrapolation'
    elif 'what if' in q_text.lower() or 'without' in q_text.lower():
        adapter_type = 'causal_responsibility'
    elif 'moving' in q_text.lower() or 'direction' in q_text.lower():
        adapter_type = 'motion_direction'
    elif 'faster' in q_text.lower() or 'speed' in q_text.lower():
        adapter_type = 'speed_comparison'
    
    answer = question.get('answer', '')
    if isinstance(answer, list):
        answer = answer[0] if answer else ''
    
    return {
        'question_text': q_text,
        'question_type': adapter_type,
        'clevrer_type': q_type,
        'ground_truth': str(answer).lower(),
        'choices': question.get('choices', [])
    }


def _gen_kwargs_for_mode(gen_mode: str) -> Dict[str, Any]:
    """Translate a CLI ``gen_mode`` into ``adapter.forward`` generation kwargs.

    ``sample``   : stochastic (default; legacy Jan 25 behavior — T=0.7, top_p=0.9)
    ``greedy``   : deterministic argmax at each step
    ``beam4``    : beam search with 4 beams, deterministic
    ``beam8``    : beam search with 8 beams, deterministic
    """
    if gen_mode == 'sample':
        return {'do_sample': True, 'temperature': 0.7, 'top_p': 0.9, 'num_beams': 1}
    if gen_mode == 'greedy':
        return {'do_sample': False, 'num_beams': 1}
    if gen_mode == 'beam4':
        return {'do_sample': False, 'num_beams': 4}
    if gen_mode == 'beam8':
        return {'do_sample': False, 'num_beams': 8}
    raise ValueError(f"Unknown gen_mode={gen_mode!r}")


def _contrastive_score_choices(
    adapter,
    physics_states: torch.Tensor,
    object_mask: torch.Tensor,
    question_text: str,
    choice_texts: List[str],
    alpha: float = 1.0,
    device: str = 'cuda',
) -> torch.Tensor:
    """Score MCQ choices via Contrastive Decoding (Jan 26 published recipe).

    Reproduces ``contrastive_score_choices`` from
    ``clevrer_benchmark_reproducibility/run_contrastive_decoding.py:193-294``
    which produced the 65-67% per-question accuracies in the
    ``archive_20260127/grounded_physics_lm_contrastive*.json`` results.

    For each choice ``c`` the algorithm computes ``score_real(c)`` and
    ``score_zero(c)`` (negative cross-entropy of the choice tokens conditioned
    on the prompt + physics prefix vs. the prompt + zero prefix), and returns

        contrastive_score(c) = score_real(c) + alpha * (score_real(c) - score_zero(c))

    The ``argmax`` of these scores is the choice whose likelihood is *most
    uplifted* by the physics prefix, not just the most-likely choice. This
    amplifies the (typically weak) physics-dependent signal that plain
    perplexity-ranking misses.
    """
    batch_size = physics_states.size(0)
    assert batch_size == 1, "single-question scoring path"

    # Build real and zero physics prefixes once per question.
    real_features = adapter.extract_physics_features(physics_states, object_mask)
    real_prefix = adapter.create_prefix_tokens(real_features)
    zero_states = torch.zeros_like(physics_states)
    zero_features = adapter.extract_physics_features(zero_states, object_mask)
    zero_prefix = adapter.create_prefix_tokens(zero_features)

    real_scores = []
    zero_scores = []
    prefix_len = adapter.num_prefix_tokens

    for choice_text in choice_texts:
        # Format matches the Jan 26 recipe: "<question> Answer: <choice>".
        full_text = question_text + " Answer: " + choice_text
        tokens = adapter.tokenizer(
            [full_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        text_embeds = adapter.llm.transformer.wte(tokens.input_ids)

        real_combined = torch.cat([real_prefix, text_embeds], dim=1)
        zero_combined = torch.cat([zero_prefix, text_embeds], dim=1)

        prefix_mask = torch.ones(batch_size, prefix_len, device=device)
        combined_mask = torch.cat([prefix_mask, tokens.attention_mask], dim=1)

        # Mask the question prompt so cross-entropy only scores the answer tokens.
        q_prompt = question_text + " Answer:"
        q_len = len(adapter.tokenizer.encode(q_prompt, add_special_tokens=False))
        labels = tokens.input_ids.clone()
        labels[:, :q_len] = -100
        prefix_labels = torch.full((batch_size, prefix_len), -100,
                                   dtype=torch.long, device=device)
        labels = torch.cat([prefix_labels, labels], dim=1)

        with torch.no_grad():
            real_out = adapter.llm(
                inputs_embeds=real_combined,
                attention_mask=combined_mask,
                labels=labels,
                return_dict=True,
            )
            zero_out = adapter.llm(
                inputs_embeds=zero_combined,
                attention_mask=combined_mask,
                labels=labels,
                return_dict=True,
            )

        # Per-sample mean cross-entropy over answer tokens (matches
        # ``compute_sample_loss`` in the published script).
        def _mean_loss(logits, lbls):
            logits = logits[:, :-1, :].contiguous()
            shift = lbls[:, 1:].contiguous()
            ce = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                shift.view(-1),
                reduction='none',
            ).view(batch_size, -1)
            valid = (shift != -100).float()
            return (ce * valid).sum(dim=-1) / (valid.sum(dim=-1) + 1e-8)

        real_scores.append(-_mean_loss(real_out.logits, labels))
        zero_scores.append(-_mean_loss(zero_out.logits, labels))

    real_scores = torch.stack(real_scores, dim=1)  # [batch=1, num_choices]
    zero_scores = torch.stack(zero_scores, dim=1)
    return real_scores + alpha * (real_scores - zero_scores)


def answer_with_adapter(
    adapter,
    states: np.ndarray,
    masks: np.ndarray,
    question_data: Dict[str, Any],
    device: str = 'cuda',
    zero_physics: bool = False,
    eval_method: str = 'generate',
    gen_mode: str = 'sample',
    contrastive_alpha: float = 1.0,
    shuffle_choices: bool = False,
) -> str:
    """Get answer from the adapter model.

    CLEVRER explanatory/predictive/counterfactual are multiple-choice. Two methods:

      - ``generate`` (default, **canonical**): prompt the adapter with
        ``"<question> Options: a, b, c"`` and let it generate free-form text,
        then substring-match against the choices. This matches the
        text-generation supervision the adapter was trained with and
        reproduces the Phase 3 79.6% SOTA on the full 5000-scene CLEVRER
        validation split.
      - ``contrastive``: for each choice ``c`` compute
        ``score_real(c) + alpha * (score_real(c) - score_zero(c))`` where
        ``score_zero`` is the log-likelihood with the physics prefix zeroed.
        Amplifies the physics-dependent signal at inference time. Jan 26
        published recipe; reproduces the 65-67% numbers in
        ``archive_20260127/grounded_physics_lm_contrastive*.json``.

    Open-ended questions (no ``choices``) always use generation.

    ``score`` (plain perplexity-rank of choices) was removed: the adapter
    was trained with generation loss, not MCQ-classification loss, so
    perplexity-ranking gives ~50% (near-random) and is misleading. Use
    ``generate`` instead.
    """
    states_tensor = torch.from_numpy(states).float().unsqueeze(0).to(device)

    # Zero out physics vectors for text-only ablation
    if zero_physics:
        states_tensor = torch.zeros_like(states_tensor)

    # Use 2D mask [batch, objects] - take first frame's mask since objects are consistent
    if masks.ndim == 2:
        masks_2d = masks[0]  # [T, N] -> [N]
    else:
        masks_2d = masks
    masks_tensor = torch.from_numpy(masks_2d).float().unsqueeze(0).to(device)

    question_text = question_data['question_text']
    choices = question_data.get('choices')

    # Phase B/C: auto-prepend scene-summary text when the adapter was
    # trained with inject_scene_text=True. Matches training distribution
    # exactly: the V3 training path prepends the same summary before
    # computing loss, so inference without it would be a distribution
    # shift. Uses the FULL-TRAJECTORY mask (union across time) because
    # CLEVRER questions routinely reference objects that exited the
    # camera view by frame 64 -- same rationale as free_form_transfer_test.py.
    if getattr(adapter, 'inject_scene_text', False):
        # Import on first use to avoid a global dependency on the adapter
        # package from the evaluation script's top-level imports.
        try:
            from physics_llm_adapter.scene_summary import build_scene_summary  # type: ignore
        except ImportError:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from physics_llm_adapter.scene_summary import build_scene_summary  # type: ignore
        # Build from numpy states (full trajectory) + union mask, using
        # frame 0 for static attributes (color / shape / mass are all
        # constant across time per scene_converter).
        if masks.ndim == 2:
            union_mask = masks.max(axis=0)
        else:
            union_mask = masks
        scene_summary_states = states[0] if states.ndim == 3 else states
        scene_text = build_scene_summary(
            scene_summary_states, union_mask,
            style=getattr(adapter, 'scene_text_style', 'comma_list'),
            include_color=getattr(adapter, 'scene_text_include_color', True),
            include_material=getattr(adapter, 'scene_text_include_material', True),
        )
        question_text = f"{scene_text} {question_text}"

    # MCQ path
    if isinstance(choices, list) and choices:
        # Optional positional-bias ablation: shuffle choice order per question.
        # Each choice dict keeps its own correct/wrong label, so correctness
        # is preserved while the LLM sees choices in a randomized order.
        if shuffle_choices:
            choices = list(choices)
            random.shuffle(choices)
        choice_texts = [
            c.get('choice', c) if isinstance(c, dict) else str(c)
            for c in choices
        ]
        if eval_method == 'contrastive':
            scores = _contrastive_score_choices(
                adapter, states_tensor, masks_tensor,
                question_text, choice_texts,
                alpha=contrastive_alpha, device=device,
            )
            pred_idx = int(scores[0].argmax().item())
            return str(choice_texts[pred_idx]).strip().lower()
        # Default "generate" path (canonical): free-form generation + substring match.
        full_question = question_text + ' Options: ' + ', '.join(choice_texts)
        gen_kwargs = _gen_kwargs_for_mode(gen_mode)
        with torch.no_grad():
            answers = adapter.forward(
                physics_states=states_tensor,
                object_mask=masks_tensor,
                question_text=[full_question],
                max_length=50,
                **gen_kwargs,
            )
        if isinstance(answers, list) and len(answers) > 0:
            return str(answers[0]).split('\n')[0].strip().lower()
        return str(answers).lower().strip()

    # Open-ended path: use LLM generation.
    gen_kwargs = _gen_kwargs_for_mode(gen_mode)
    with torch.no_grad():
        answers = adapter.forward(
            physics_states=states_tensor,
            object_mask=masks_tensor,
            question_text=[question_text],
            max_length=50,
            **gen_kwargs,
        )

    if isinstance(answers, list) and len(answers) > 0:
        return str(answers[0]).split('\n')[0].strip().lower()
    return str(answers).lower().strip()


def evaluate_answer(predicted: str, ground_truth: str, choices: List = None) -> bool:
    """Check if predicted answer matches ground truth.
    
    Uses strict matching for MCQ - prediction must match the correct choice text.
    """
    pred = predicted.strip().lower()
    gt = ground_truth.strip().lower()
    
    # Exact match
    if pred == gt:
        return True
    
    # For MCQ with choices, check if prediction matches a correct choice
    if choices:
        correct_choices = [c.get('choice', '').lower().strip() for c in choices if c.get('answer') == 'correct']
        wrong_choices = [c.get('choice', '').lower().strip() for c in choices if c.get('answer') == 'wrong']
        
        # Check if prediction matches correct choice (substring in either direction)
        pred_matches_correct = any(
            pred == c or c in pred or pred in c
            for c in correct_choices
        )
        pred_matches_wrong = any(
            pred == c or c in pred or pred in c
            for c in wrong_choices
        )
        
        # Only correct if matches correct AND doesn't match wrong
        if pred_matches_correct and not pred_matches_wrong:
            return True
        
        # If matches both, it's ambiguous - count as wrong
        if pred_matches_correct and pred_matches_wrong:
            return False
    
    # For non-MCQ, allow substring match
    if not choices:
        if gt in pred or pred in gt:
            return True
    
    return False


def _resolve_scene_dir(clevrer_dir: Path, override: Optional[Path]) -> Path:
    """Find the directory of CLEVRER validation scene JSONs.

    Tries (in order): explicit override, ``scenes/validation``,
    ``scenes/clevrer_scenes`` (the released CLEVRER dump layout), and finally
    any first-level subdir of ``scenes/`` that contains ``*.json``.
    """
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"--scene_dir not found: {override}")
        return override
    candidates = [clevrer_dir / 'scenes' / 'validation',
                  clevrer_dir / 'scenes' / 'clevrer_scenes']
    for c in candidates:
        if c.exists():
            return c
    scenes_root = clevrer_dir / 'scenes'
    if scenes_root.exists():
        for sub in sorted(scenes_root.iterdir()):
            if sub.is_dir() and any(sub.glob('*.json')):
                return sub
    raise FileNotFoundError(
        f"Could not locate scene directory under {clevrer_dir}. "
        f"Tried: {[str(c) for c in candidates]}. Pass --scene_dir explicitly."
    )


def _resolve_questions_file(clevrer_dir: Path, override: Optional[Path]) -> Path:
    """Find the CLEVRER validation questions JSON.

    Tries (in order): explicit override, ``questions/validation.json``,
    ``questions/clevrer_validation.json`` (the released CLEVRER dump layout).
    """
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"--questions_file not found: {override}")
        return override
    candidates = [clevrer_dir / 'questions' / 'validation.json',
                  clevrer_dir / 'questions' / 'clevrer_validation.json']
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not locate questions JSON under {clevrer_dir}. "
        f"Tried: {[str(c) for c in candidates]}. Pass --questions_file explicitly."
    )


def _save_progress(output_path: Path, results: 'BenchmarkResults') -> None:
    """Atomically write the in-progress summary JSON.

    Writes to ``<output>.tmp`` first, then ``replace`` to the final path so a
    kill mid-flush leaves either the previous valid summary or the new one --
    never a half-written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(results.to_dict(), f, indent=2)
    tmp.replace(output_path)


def _load_heldout_scene_indices(h5_path: Path) -> set:
    """Wrapper around compute_paper_stats.load_heldout_scenes.

    Lazy-imports the helper so a CLEVRER-only eval doesn't pay the h5py cost
    when --heldout isn't requested. Adds the scripts/ dir to sys.path because
    compute_paper_stats lives there as a top-level module (no package).
    """
    import sys as _sys
    _scripts = Path(__file__).resolve().parent / 'scripts'
    if str(_scripts) not in _sys.path:
        _sys.path.insert(0, str(_scripts))
    from compute_paper_stats import load_heldout_scenes  # type: ignore
    return load_heldout_scenes(Path(h5_path))


def run_evaluation(
    clevrer_dir: Path,
    adapter_checkpoint: str,
    physics_checkpoint: str,
    max_scenes: int = None,
    max_questions: int = None,
    device: str = 'cuda',
    skip_descriptive: bool = False,
    single_frame: int = None,
    zero_physics: bool = False,
    eval_method: str = 'generate',
    gen_mode: str = 'sample',
    gen_seed: Optional[int] = None,
    scene_dir: Optional[Path] = None,
    questions_file: Optional[Path] = None,
    contrastive_alpha: float = 1.0,
    zero_prefix: bool = False,
    shuffle_choices: bool = False,
    filter_malformed: bool = True,
    output_path: Optional[Path] = None,
    save_details: bool = False,
    heldout_scenes: Optional[set] = None,
    save_every: int = 25,
) -> BenchmarkResults:
    """Run CLEVRER evaluation with adapter model.

    When ``output_path`` is provided the function streams per-question records
    to ``<output>.details.jsonl`` (if ``save_details=True``) and atomically
    re-writes the summary JSON every ``save_every`` scenes. A kill mid-run
    therefore loses at most one save_every cohort of scenes, instead of the
    entire run. The final flush happens in a ``finally`` block so even
    exceptions preserve completed work.

    ``heldout_scenes`` (a set of CLEVRER scene_index integers) restricts the
    iteration to scenes the Phase 3 adapter never trained on -- matches the
    paper's primary evaluation pool when combined with --filter_malformed.
    """
    scene_dir = _resolve_scene_dir(clevrer_dir, scene_dir)
    questions_file = _resolve_questions_file(clevrer_dir, questions_file)
    print(f"[data] scene_dir: {scene_dir}")
    print(f"[data] questions: {questions_file}")
    
    print("Loading adapter model...")
    adapter = load_adapter_model(adapter_checkpoint, physics_checkpoint, device)

    # Zero-prefix ablation: physics processes normally, but the 64 adapter prefix
    # tokens are zeroed before they enter the LLM. Tests whether the LLM actually
    # utilizes the physics prefix vs. relying on question text + MCQ structure.
    if zero_prefix:
        print("[ablation] zero_prefix=True: adapter prefix tokens will be zeroed before the LLM")
        original_create_prefix = adapter.create_prefix_tokens
        def _zeroed_create_prefix(physics_features):
            tokens = original_create_prefix(physics_features)
            return torch.zeros_like(tokens)
        adapter.create_prefix_tokens = _zeroed_create_prefix
    if shuffle_choices:
        print("[ablation] shuffle_choices=True: MCQ choice order will be randomized per question")
    if filter_malformed:
        print("[filter] filter_malformed=True: skipping CLEVRER questions where all choices are labeled wrong "
              "(matches LLM baseline validate_question() filter for apples-to-apples comparison)")
    else:
        print("[filter] filter_malformed=False: including all questions, even ones with no correct choice "
              "(these auto-fail under free-form generation eval)")

    if gen_seed is not None:
        torch.manual_seed(gen_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(gen_seed)
        random.seed(gen_seed)
        print(f"[seed] torch.manual_seed({gen_seed}) set for deterministic sampling")
    print(f"[gen] mode={gen_mode}, method={eval_method}")
    
    print("Loading questions...")
    with open(questions_file, 'r') as f:
        all_questions = json.load(f)
    
    questions_by_scene = defaultdict(list)
    for scene_data in all_questions:
        scene_idx = scene_data.get('scene_index', scene_data.get('video_index', 0))
        questions_by_scene[scene_idx] = scene_data.get('questions', [])
    
    scene_files = sorted(scene_dir.glob('*.json'))

    # Held-out filter: restrict to scenes the Phase 3 adapter never trained on.
    # Applied BEFORE max_scenes / max_questions so the cap operates on the
    # already-restricted pool (otherwise --max_scenes 100 would mostly select
    # training scenes and leave very few held-out).
    if heldout_scenes is not None:
        def _scene_num(p: Path) -> int:
            try:
                return int(p.stem.split('_')[-1])
            except (ValueError, IndexError):
                return -1
        before = len(scene_files)
        scene_files = [p for p in scene_files if _scene_num(p) in heldout_scenes]
        print(f"[heldout] {before} -> {len(scene_files)} scenes (intersection with H5 held-out set of {len(heldout_scenes)})")

    if max_questions:
        # Uniformly sample scenes across the full validation set to reach
        # ~max_questions causal questions. Avg ~4 causal q per scene.
        est_scenes_needed = max_questions // 4 + 10  # slight over-estimate
        if est_scenes_needed < len(scene_files):
            step = len(scene_files) / est_scenes_needed
            scene_files = [scene_files[int(i * step)] for i in range(est_scenes_needed)]
            print(f"[sample] Uniformly sampled {len(scene_files)} scenes across full set for ~{max_questions} questions")
    elif max_scenes:
        scene_files = scene_files[:max_scenes]

    # Streaming details writer: per-question record flushed to disk so a kill
    # mid-run preserves everything up to the last completed scene.
    details_handle = None
    if save_details and output_path is not None:
        details_path = Path(output_path).with_suffix('.details.jsonl')
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_handle = open(details_path, 'w', encoding='utf-8')
        print(f"[stream] streaming per-question details to {details_path}")
    if output_path is not None:
        print(f"[stream] periodic summary save every {save_every} scenes -> {output_path}")

    results = BenchmarkResults()

    print(f"\nEvaluating on {len(scene_files)} scenes...")
    
    for idx, scene_file in enumerate(scene_files):
        scene_id = scene_file.stem
        scene_num = int(scene_id.split('_')[-1]) if '_' in scene_id else idx
        
        questions = questions_by_scene.get(scene_num, [])
        if not questions:
            continue
        
        try:
            scene = load_clevrer_scene(str(scene_file))
            states, masks, metadata = clevrer_scene_to_state_tensor(scene)
            
            # If single_frame specified, use only that frame (for fair comparison with LLMs).
            # Negative value means "use all frames" (escape hatch for the default of 64).
            if single_frame is not None and single_frame >= 0:
                frame_idx = min(single_frame, states.shape[0] - 1)
                states = states[frame_idx:frame_idx+1]  # [1, N, 35]
                masks = masks[frame_idx:frame_idx+1] if masks.ndim == 2 else masks
        except Exception as e:
            print(f"\nError loading scene {scene_id}: {e}")
            continue
        
        for q_idx, question in enumerate(questions):
            q_type = question.get('question_type', 'descriptive')
            
            # Skip descriptive questions if requested
            if skip_descriptive and q_type == 'descriptive':
                continue

            # Filter malformed MCQ questions (all choices labeled wrong). This matches the
            # baseline LLM evaluation's validate_question() check -- required for apples-to-apples
            # comparison with LLM baselines. ~7% of CLEVRER val questions are zero-correct "trap"
            # questions that auto-fail under free-form generation eval.
            if filter_malformed:
                choices = question.get('choices') or []
                if choices:
                    has_correct = any(
                        isinstance(c, dict) and c.get('answer') == 'correct'
                        for c in choices
                    )
                    if not has_correct:
                        continue
            
            try:
                q_data = map_clevrer_to_adapter_question(question, metadata)
                predicted = answer_with_adapter(
                    adapter, states, masks, q_data, device,
                    zero_physics=zero_physics, eval_method=eval_method,
                    gen_mode=gen_mode,
                    contrastive_alpha=contrastive_alpha,
                    shuffle_choices=shuffle_choices,
                )
                correct = evaluate_answer(predicted, q_data['ground_truth'], q_data.get('choices'))
                
                result = EvaluationResult(
                    scene_id=scene_id,
                    question_id=f"{scene_id}_q{q_idx}",
                    question_text=q_data['question_text'],
                    clevrer_type=q_type,
                    ground_truth=q_data['ground_truth'],
                    predicted=predicted,
                    correct=correct,
                    choices=q_data.get('choices') or None,
                )
                results.add_result(result)
                if details_handle is not None:
                    details_handle.write(
                        json.dumps(result.to_detail_dict(), ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"\nError on {scene_id} Q{q_idx}: {e}")
                continue

        # Stop early if we've reached the question cap
        if max_questions and results.total_questions >= max_questions:
            print(f"\n[cap] Reached {results.total_questions} questions (target: {max_questions}), stopping.")
            break

        if (idx + 1) % 10 == 0:
            acc = results.get_accuracy() * 100
            print(f"\rProgress: {idx + 1}/{len(scene_files)} scenes, "
                  f"{results.total_questions} questions, "
                  f"Accuracy: {acc:.1f}%", end='', flush=True)

        # Periodic incremental save so a kill mid-run loses at most one cohort.
        if output_path is not None and (idx + 1) % save_every == 0:
            _save_progress(output_path, results)
            if details_handle is not None:
                details_handle.flush()

    print()
    # Final flush + save (covers the partial cohort < save_every and exception paths).
    if details_handle is not None:
        details_handle.flush()
        details_handle.close()
    if output_path is not None:
        _save_progress(output_path, results)
    return results


def print_results(results: BenchmarkResults):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("CLEVRER BENCHMARK RESULTS - Physics-LLM Adapter")
    print("=" * 60)
    
    overall = results.to_dict()['overall']
    print(f"\nOverall Accuracy: {overall['accuracy']:.1%} ({overall['correct']}/{overall['total_questions']})")
    
    print("\n--- By CLEVRER Question Type ---")
    for q_type, metrics in results.to_dict()['by_clevrer_type'].items():
        print(f"  {q_type}: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Run CLEVRER evaluation with Physics-LLM Adapter')
    parser.add_argument('--clevrer_dir', type=str, default='D:\\clevrer',
                        help='Path to CLEVRER dataset directory')
    parser.add_argument('--adapter_checkpoint', type=str, 
                        default='checkpoints/adapter_phase3.pt',
                        help='Path to adapter checkpoint (canonical SOTA: checkpoints/adapter_phase3.pt)')
    parser.add_argument('--physics_checkpoint', type=str,
                        default='D:\\physics-former-data\\checkpoints\\stage1_best.pt',
                        help='Path to physics model checkpoint')
    parser.add_argument('--max_scenes', type=int, default=100,
                        help='Maximum number of scenes to evaluate')
    parser.add_argument('--max_questions', type=int, default=None,
                        help='Target number of causal questions (overrides --max_scenes). '
                             'Uniformly samples scenes across the full validation set.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--skip_descriptive', action='store_true',
                        help='Skip descriptive questions (focus on explanatory/predictive/counterfactual)')
    parser.add_argument('--single_frame', type=int, default=64,
                        help='Use only a single frame for fair comparison with LLMs (default 64 matches the '
                             'canonical Jan 25 / Phase 3 SOTA protocol). Pass e.g. --single_frame -1 to use all frames.')
    parser.add_argument('--zero_physics', action='store_true',
                        help='Zero the physics state tensors at the encoder input (text-only ablation). '
                             'Expected large accuracy drop if the model genuinely depends on physics.')
    parser.add_argument('--zero_prefix', action='store_true',
                        help='Zero the 64 adapter prefix tokens BEFORE they enter the LLM. Physics processes '
                             'normally through the adapter but the LLM sees no physics signal. '
                             'Tests whether the LLM actually utilizes the prefix vs. the question text alone.')
    parser.add_argument('--shuffle_choices', action='store_true',
                        help='Randomly shuffle MCQ choice order for each question. Combined with --zero_prefix '
                             'this gives the true text-only baseline (controls for positional bias like "A is often correct").')
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument('--filter_malformed', dest='filter_malformed', action='store_true',
                              help='Skip CLEVRER questions where every choice is labeled wrong (default; '
                                   'matches LLM baseline run_llm_with_scene.py validate_question() filter '
                                   'for apples-to-apples comparison).')
    filter_group.add_argument('--no_filter_malformed', dest='filter_malformed', action='store_false',
                              help='Include all questions, even ones with no correct choice. These auto-fail '
                                   'under our free-form generation eval (~7%% of CLEVRER val) and depress accuracy '
                                   'unfairly vs. the baselines. Only use this if you specifically want inclusive numbers.')
    parser.set_defaults(filter_malformed=True)
    parser.add_argument('--eval_method', type=str, default='generate',
                        choices=['generate', 'contrastive'],
                        help='MCQ eval. "generate" (default, canonical): free-form generation + substring match, '
                             'matches the adapter\'s text-generation training objective and reproduces the Phase 3 '
                             '79.6%% SOTA. "contrastive" (Jan 26 recipe): ranks choices by physics-amplified likelihood '
                             '(score_real + alpha * (score_real - score_zero)).')
    parser.add_argument('--contrastive_alpha', type=float, default=1.0,
                        help='Alpha for --eval_method=contrastive (default 1.0 matches the Jan 26 published recipe)')
    parser.add_argument('--gen_mode', type=str, default='sample',
                        choices=['sample', 'greedy', 'beam4', 'beam8'],
                        help='Generation decoding (only used when --eval_method=generate). '
                             '"sample" (default) is the Jan 25 stochastic mode (T=0.7, top_p=0.9) used for the '
                             '79.6%% SOTA. "greedy" / "beam4" / "beam8" are deterministic alternatives.')
    parser.add_argument('--gen_seed', type=int, default=42,
                        help='Seed torch RNG before generation for reproducible sampling (default 42 matches the SOTA run)')
    parser.add_argument('--scene_dir', type=str, default=None,
                        help='Override scene directory (default: auto-detect under clevrer_dir/scenes/)')
    parser.add_argument('--questions_file', type=str, default=None,
                        help='Override questions JSON path (default: auto-detect under clevrer_dir/questions/)')
    parser.add_argument('--save_details', action='store_true',
                        help='Write per-question predictions (JSONL) alongside the summary JSON. '
                             'Sidecar path is --output with suffix replaced by .details.jsonl. '
                             'Streamed per question so a kill mid-run preserves completed work.')
    parser.add_argument('--heldout', action='store_true',
                        help='Restrict evaluation to the 501 CLEVRER val scenes the Phase 3 adapter '
                             'never trained on (10%% held-out split derived from the H5 question pool). '
                             'Combine with --filter_malformed (default) to reproduce the paper primary '
                             'evaluation pool of ~1998 questions.')
    parser.add_argument('--h5', type=str, default=None,
                        help='CLEVRER training H5 path (only used with --heldout). '
                             'Default: $CLEVRER_H5 or $CLEVRER_H5.')
    parser.add_argument('--save_every', type=int, default=25,
                        help='Periodic incremental save cadence in scenes (default 25). '
                             'Lower = safer (less work lost on kill) but more disk churn.')

    args = parser.parse_args()

    heldout_scenes = None
    if args.heldout:
        h5_path = Path(args.h5) if args.h5 else Path(
            os.environ.get('CLEVRER_H5', os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5')))
        print(f"[heldout] loading held-out scene_indices from {h5_path}")
        heldout_scenes = _load_heldout_scene_indices(h5_path)
        print(f"[heldout] {len(heldout_scenes)} scenes never seen by Phase 3 training")

    output_path = Path(args.output) if args.output else None

    results = run_evaluation(
        clevrer_dir=Path(args.clevrer_dir),
        adapter_checkpoint=args.adapter_checkpoint,
        physics_checkpoint=args.physics_checkpoint,
        max_scenes=args.max_scenes,
        max_questions=args.max_questions,
        device=args.device,
        skip_descriptive=args.skip_descriptive,
        single_frame=args.single_frame,
        zero_physics=args.zero_physics,
        eval_method=args.eval_method,
        gen_mode=args.gen_mode,
        gen_seed=args.gen_seed,
        contrastive_alpha=args.contrastive_alpha,
        scene_dir=Path(args.scene_dir) if args.scene_dir else None,
        questions_file=Path(args.questions_file) if args.questions_file else None,
        zero_prefix=args.zero_prefix,
        shuffle_choices=args.shuffle_choices,
        filter_malformed=args.filter_malformed,
        output_path=output_path,
        save_details=args.save_details,
        heldout_scenes=heldout_scenes,
        save_every=args.save_every,
    )

    print_results(results)

    if output_path is not None:
        # run_evaluation already streamed/saved; surface the final paths.
        print(f"\nResults saved to: {output_path}")
        if args.save_details:
            details_path = output_path.with_suffix('.details.jsonl')
            print(f"Per-question details saved to: {details_path} ({len(results.individual_results)} records)")


if __name__ == '__main__':
    main()
