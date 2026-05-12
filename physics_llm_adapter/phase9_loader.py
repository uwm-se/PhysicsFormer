"""
Phase 9 / Phase 10 adapter builder.

Constructs a ``PhysicsLLMAdapterV3`` for the diversity-grounding
training tracks. Two entry points are exported:

  * ``build_phase9_adapter``  -- the original Phase 9 cold-start-LoRA
                                  + warm-start-MLP recipe. Used to
                                  produce ``adapter_phase9_diverse_grounding.pt``.
  * ``build_phase10_adapter`` -- Phase 10 warm-start-everything from a
                                  Phase 9+ checkpoint (LoRA + adapter
                                  MLP + slot bias). Used for the
                                  format-cue-augmented retrain that
                                  closes the eval-time prompt-cue gap.

Both share the same architectural build (LoRA r=16, prefix_dropout=0.30,
free-form heads dropped, frozen LLM + physics encoder). The only
difference is whether the LoRA matrices are filtered out of the source
checkpoint at warm-start time:

+----------+--------------------------+----------------+--------------+
|          | Adapter MLP + slot bias  | LoRA q/k/v/o   | Heads        |
+----------+--------------------------+----------------+--------------+
| Phase 9  | warm-start from Phase 7  | COLD (fresh)   | DROPPED      |
| Phase 10 | warm-start from Phase 9  | warm from P9   | DROPPED      |
+----------+--------------------------+----------------+--------------+

Usage
-----
::

    from physics_llm_adapter.phase9_loader import (
        build_phase9_adapter, build_phase10_adapter,
    )

    physics_model = load_physics_former(checkpoint_path)  # existing helper

    # Phase 9 (cold-start LoRA):
    adapter = build_phase9_adapter(
        physics_model=physics_model,
        phase7_checkpoint_path='.../adapter_phase7_qwen_lora_best.pt',
        device='cuda',
    )

    # Phase 10 (warm-start everything):
    adapter = build_phase10_adapter(
        physics_model=physics_model,
        phase9_checkpoint_path='.../adapter_phase9_diverse_grounding.pt',
        device='cuda',
    )

Both print a transparency banner so the training log records exactly
what was warm-started, what was cold-started, and what was dropped.
Reproducibility is preserved by not perturbing any RNG state owned by
the caller.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .adapter_v3 import PhysicsLLMAdapterV3, create_adapter_v3


# ---------------------------------------------------------------------
# Default Phase 9 recipe knobs.
# ---------------------------------------------------------------------
# Captured here so the notebook + the unit tests + the recipe sidecar
# all read the same defaults. Tweak by passing kwargs to
# ``build_phase9_adapter`` -- never edit these in-place.
PHASE9_DEFAULTS: Dict[str, object] = {
    'llm_name':                 'Qwen/Qwen2.5-1.5B-Instruct',
    'physics_dim':              768,
    'tokens_per_object':        4,                # 20 obj * 4 = 80 prefix tokens
    'use_lora':                 True,
    'lora_rank':                16,
    'lora_alpha':               32.0,
    'lora_dropout':             0.05,
    'lora_target_modules':      ('q_proj', 'k_proj', 'v_proj', 'o_proj'),
    'prefix_dropout':           0.30,             # Phase 7 lift, retained
    'inject_scene_text':        True,
    'scene_text_style':         'numbered',
    'scene_text_include_color': True,
    'scene_text_include_material': True,
    # Phase 9 disables descriptor stripping by default. Phase 8's
    # n=200 evidence showed that with monoculture training data,
    # descriptor stripping forced prefix grounding only as a binary
    # ``CLEVRER mode`` switch, not as a per-scene representation. The
    # Phase 9 hypothesis is that question-type DIVERSITY is the
    # missing ingredient, not stripping. Set to >0 only as an ablation.
    'descriptor_strip_prob':    0.0,
    'descriptor_strip_seed':    4747,
    # Free-form QA only -- the descriptive/numerical/MCQ classifier
    # heads are dropped from the model, so ``classify_clevrer_question``
    # routing must be bypassed at every loss call.
    'force_llm_only_routing':   True,
    # Format A every sample: V3 prompts are ``"{question} Answer:"``
    # without an ``Options:`` clause, matching the free-form mode.
    'include_choices_prob':     0.0,
    # Mild noise on the answer-side input tokens to discourage verbatim
    # copying. Same value as Phase 4 free-form ablation.
    'answer_token_dropout':     0.10,
    # Heads not built at all -- saves ~12 MB of params and avoids
    # warm-start ambiguity. Phase 9 is free-form; the heads were
    # CLEVRER-MCQ-specific.
    'build_mcq_head':           False,
    'build_descriptive_head':   False,
}


# Regex captures every key that is part of a peft LoRA layer (both the
# trainable lora_A/lora_B params and any peft-managed bookkeeping like
# ``base_layer.*`` -- those are RE-INSTANTIATED by peft when we apply
# LoRA fresh, so loading them from Phase 7 would just re-clobber the
# fresh state).
_LORA_KEY_PATTERNS = (
    re.compile(r'(^|\.)lora_A\.'),
    re.compile(r'(^|\.)lora_B\.'),
    re.compile(r'(^|\.)lora_embedding'),
    # peft wraps every targeted Linear with ``base_layer.weight`` /
    # ``base_layer.bias``; the base_layer values come from the frozen
    # Qwen download anyway so loading them is a no-op when matched, but
    # we drop them defensively to avoid version-mismatch errors if peft
    # internal naming shifts in a future upgrade.
    re.compile(r'\.base_layer\.'),
)
_HEAD_KEY_PREFIXES: Tuple[str, ...] = (
    'numerical_head.',
    'descriptive_head.',
    'mcq_head.',
)
# Keys that we positively WANT to load from Phase 7. Anything that
# matches these and not any of the drop-patterns is what we warm-start.
_WARM_START_PREFIXES: Tuple[str, ...] = (
    'adapter_per_object.',
    'per_object_slot_bias',
)


def _is_lora_key(key: str) -> bool:
    return any(p.search(key) for p in _LORA_KEY_PATTERNS)


def _is_head_key(key: str) -> bool:
    return any(key.startswith(p) for p in _HEAD_KEY_PREFIXES)


def _is_warm_start_key(key: str) -> bool:
    return any(key.startswith(p) or key == p.rstrip('.')
               for p in _WARM_START_PREFIXES)


def _is_llm_key(key: str) -> bool:
    return key.startswith('llm.')


def _is_physics_model_key(key: str) -> bool:
    return key.startswith('physics_model.')


def _filter_warm_start_state_dict(
    src_state: Dict[str, torch.Tensor],
    *,
    keep_lora: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Return only the keys we want to warm-start.

    Counts kept/dropped keys by category so the loader can print a
    transparent breakdown. ``physics_model.*`` keys are intentionally
    dropped here because the physics encoder is loaded separately
    from ``physics_former_best.pt`` (the same procedure every phase
    has used since Phase 1).

    keep_lora controls whether peft LoRA matrices and base_layer keys
    are retained. Phase 9 sets keep_lora=False (cold-start LoRA from
    Phase 7); Phase 10 sets keep_lora=True (warm-start LoRA from a
    Phase 9+ checkpoint).
    """
    kept: Dict[str, torch.Tensor] = {}
    counts = {
        'kept_adapter_mlp': 0,
        'kept_slot_bias': 0,
        'kept_lora': 0,
        'dropped_lora': 0,
        'dropped_head': 0,
        'dropped_llm': 0,
        'dropped_physics': 0,
        'dropped_other': 0,
    }
    for k, v in src_state.items():
        if _is_warm_start_key(k):
            kept[k] = v
            if k == 'per_object_slot_bias':
                counts['kept_slot_bias'] += 1
            else:
                counts['kept_adapter_mlp'] += 1
        elif _is_lora_key(k):
            if keep_lora:
                kept[k] = v
                counts['kept_lora'] += 1
            else:
                counts['dropped_lora'] += 1
        elif _is_head_key(k):
            counts['dropped_head'] += 1
        elif _is_llm_key(k):
            counts['dropped_llm'] += 1
        elif _is_physics_model_key(k):
            counts['dropped_physics'] += 1
        else:
            counts['dropped_other'] += 1
    return kept, counts


# Backwards-compatibility shim. Older callers and unit tests reach for
# the Phase-7-only name; route them to the generalised helper with the
# Phase 9 default (cold-start LoRA).
def _filter_phase7_state_dict(
    src_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    return _filter_warm_start_state_dict(src_state, keep_lora=False)


def build_phase9_adapter(
    physics_model: torch.nn.Module,
    *,
    phase7_checkpoint_path: Optional[str] = None,
    device: str = 'cuda',
    verbose: bool = True,
    **overrides,
) -> PhysicsLLMAdapterV3:
    """Build a Phase 9 cold-start-LoRA + warm-start-MLP adapter.

    Args
    ----
    physics_model : a ``FullPhysicsFormer`` (or compatible) instance,
        already loaded from ``physics_former_best.pt`` and frozen by
        the caller. Phase 9 does not retrain the encoder.

    phase7_checkpoint_path : path to ``adapter_phase7_qwen_lora_best.pt``
        (or any Phase 7+ V3 checkpoint with the same key layout). When
        ``None``, every component is cold-started -- useful for an
        end-to-end ablation.

    device : where to materialize the adapter. ``'cuda'`` for training,
        ``'cpu'`` for unit tests / integration tests.

    verbose : prints a transparency banner detailing the warm-start /
        cold-start / drop breakdown. Always emitted to stdout for
        training-log capture.

    overrides : any keyword in ``PHASE9_DEFAULTS`` can be overridden
        here. Common ablation knobs: ``descriptor_strip_prob`` (set to
        e.g. 0.5 for a Phase 8 + diversity ablation),
        ``prefix_dropout``, ``include_choices_prob``.

    Returns
    -------
    A ``PhysicsLLMAdapterV3`` ready for the Phase 9 training loop. The
    Phase 7 checkpoint is consumed (read once) but not retained -- the
    adapter is the only thing the caller keeps.
    """
    # Merge defaults with caller overrides (caller wins).
    cfg = dict(PHASE9_DEFAULTS)
    for k, v in overrides.items():
        if k not in cfg:
            raise ValueError(
                f'Unknown Phase 9 override {k!r}. Known knobs: '
                f'{sorted(cfg.keys())}'
            )
        cfg[k] = v

    if verbose:
        print('=' * 72)
        print('Phase 9 adapter build')
        print('=' * 72)
        print(f'  llm_name              : {cfg["llm_name"]}')
        print(f'  tokens_per_object     : {cfg["tokens_per_object"]} '
              f'(80 prefix tokens at 20 max objects)')
        print(f'  use_lora              : {cfg["use_lora"]} '
              f'(rank={cfg["lora_rank"]}, alpha={cfg["lora_alpha"]}, '
              f'targets={list(cfg["lora_target_modules"])})')
        print(f'  prefix_dropout        : {cfg["prefix_dropout"]}')
        print(f'  inject_scene_text     : {cfg["inject_scene_text"]} '
              f'(style={cfg["scene_text_style"]})')
        print(f'  descriptor_strip_prob : {cfg["descriptor_strip_prob"]} '
              f'(0 disables stripping; Phase 9 default off)')
        print(f'  force_llm_only_routing: {cfg["force_llm_only_routing"]}')
        print(f'  build_mcq_head        : {cfg["build_mcq_head"]}')
        print(f'  build_descriptive_head: {cfg["build_descriptive_head"]}')
        print()

    # ------------------------------------------------------------------
    # 1. Construct the adapter. ``use_lora=True`` triggers a FRESH peft
    #    LoraConfig + get_peft_model call inside PhysicsLLMAdapterV2.
    #    LoRA matrices are init'd with peft's defaults: A ~ Kaiming
    #    uniform, B = 0 -> the adapter starts as the identity LLM with
    #    no LoRA influence (and the LLM+LoRA training only changes B
    #    initially; A is symmetric to the rotation choice).
    # ------------------------------------------------------------------
    adapter = create_adapter_v3(
        physics_model=physics_model,
        # Forward all the Phase 9 knobs.
        include_choices_prob=cfg['include_choices_prob'],
        inject_scene_text=cfg['inject_scene_text'],
        scene_text_style=cfg['scene_text_style'],
        scene_text_include_color=cfg['scene_text_include_color'],
        scene_text_include_material=cfg['scene_text_include_material'],
        descriptor_strip_prob=cfg['descriptor_strip_prob'],
        descriptor_strip_seed=cfg['descriptor_strip_seed'],
        # V2 base-class kwargs (passed through **kwargs).
        llm_name=cfg['llm_name'],
        physics_dim=cfg['physics_dim'],
        tokens_per_object=cfg['tokens_per_object'],
        freeze_physics=True,
        freeze_llm=True,
        build_mcq_head=cfg['build_mcq_head'],
        build_descriptive_head=cfg['build_descriptive_head'],
        use_lora=cfg['use_lora'],
        lora_rank=cfg['lora_rank'],
        lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'],
        lora_target_modules=list(cfg['lora_target_modules']),
        prefix_dropout=cfg['prefix_dropout'],
        # answer_token_dropout passed through **kwargs to V3.
        answer_token_dropout=cfg['answer_token_dropout'],
        force_llm_only_routing=cfg['force_llm_only_routing'],
    ).to(device)

    # ------------------------------------------------------------------
    # 1b. Freeze unused heads. With force_llm_only_routing=True the
    #     numerical_head never receives gradients (the loss path skips
    #     it), but its params would still be enumerated by the
    #     optimizer (and AdamW would allocate m/v state) -- wasting
    #     ~2 MB of GPU memory for nothing. Explicit freeze keeps the
    #     trainable-params count clean and lets the training notebook
    #     filter `requires_grad` to find only the truly-active params.
    #
    #     ``mcq_head`` and ``descriptive_head`` are not built (the
    #     ``build_*_head=False`` defaults above), so no freeze needed.
    # ------------------------------------------------------------------
    if hasattr(adapter, 'numerical_head'):
        for p in adapter.numerical_head.parameters():
            p.requires_grad = False
        if verbose:
            print('  numerical_head        : built but FROZEN '
                  '(unused with force_llm_only_routing=True)')

    # ------------------------------------------------------------------
    # 2. Optionally warm-start adapter MLP + slot bias from Phase 7.
    # ------------------------------------------------------------------
    if phase7_checkpoint_path is None:
        if verbose:
            print('  warm-start            : NONE (full cold start ablation)')
            print('  All trainable Phase 9 params start from random init.')
        return adapter

    ckpt_path = Path(phase7_checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'Phase 7 checkpoint not found: {ckpt_path}. Pass '
            f'phase7_checkpoint_path=None for full cold-start ablation.'
        )

    if verbose:
        print(f'  warm-start source     : {ckpt_path}')

    src = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    src_state = src.get('adapter_state_dict')
    if src_state is None:
        raise KeyError(
            f"Checkpoint at {ckpt_path} has no 'adapter_state_dict' top-level "
            f"key. Got: {sorted(src.keys())}"
        )

    filtered, counts = _filter_warm_start_state_dict(src_state, keep_lora=False)

    if not filtered:
        raise RuntimeError(
            f'No warm-start keys recovered from {ckpt_path}. Expected '
            f'adapter_per_object.* and per_object_slot_bias keys. The '
            f'checkpoint may be from an older phase (pre-V3-per-object).'
        )

    # Move tensors to target device before load_state_dict so the
    # adapter's parameters don't end up CPU-resident after the load.
    filtered_dev = {k: v.to(device) for k, v in filtered.items()}

    # strict=False:
    #   missing keys      -> any param the adapter has but Phase 7 didn't
    #                        save (LoRA params, fresh peft state, etc.) --
    #                        they keep their fresh init. Expected.
    #   unexpected keys   -> Phase 7 saved keys the adapter doesn't have
    #                        (numerical_head if heads dropped, etc.) --
    #                        we already filtered those, so this should be
    #                        empty. Defensive.
    incompat = adapter.load_state_dict(filtered_dev, strict=False)

    if verbose:
        print(f'  kept adapter MLP keys : {counts["kept_adapter_mlp"]} '
              f'(adapter_per_object.*)')
        print(f'  kept slot bias keys   : {counts["kept_slot_bias"]} '
              f'(per_object_slot_bias)')
        print(f'  dropped LoRA keys     : {counts["dropped_lora"]} '
              f'(cold-started by peft)')
        print(f'  dropped head keys     : {counts["dropped_head"]} '
              f'(numerical/descriptive/mcq)')
        print(f'  dropped LLM frozen    : {counts["dropped_llm"]} '
              f'(re-loaded fresh from HF)')
        print(f'  dropped physics keys  : {counts["dropped_physics"]} '
              f'(loaded by caller from physics_former_best.pt)')
        if counts['dropped_other']:
            print(f'  dropped other         : {counts["dropped_other"]} '
                  f'(unrecognised key buckets)')

        if incompat.missing_keys:
            # These are EXPECTED -- everything in the adapter that
            # Phase 7 didn't save (fresh peft LoRA, fresh adapter init,
            # etc.). Print a sample for transparency. NOT an error.
            sample = ', '.join(incompat.missing_keys[:3])
            print(f'  fresh-init params     : {len(incompat.missing_keys)} '
                  f'(LoRA matrices etc.; e.g. {sample})')
        if incompat.unexpected_keys:
            # Should be empty after our pre-filter. If non-empty,
            # something in the source ckpt has a layout we don't
            # recognise -- print a warning so the operator can audit.
            print(f'  [WARN] unexpected_keys after filter: '
                  f'{incompat.unexpected_keys[:5]}')

        # Sanity check: trainable params should be LoRA + adapter_per_object
        # + slot bias. Roughly 4M (LoRA) + 4M (adapter) + 0.0001M (slot).
        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        total = sum(p.numel() for p in adapter.parameters())
        print(f'  trainable params      : {trainable:,} / {total:,} '
              f'({100.0 * trainable / max(total, 1):.3f}%)')
        print('=' * 72)
        print()

    return adapter


# ---------------------------------------------------------------------
# Phase 10 recipe defaults.
# ---------------------------------------------------------------------
# Phase 10 retrains from Phase 9's best checkpoint with format-cue-
# augmented training prompts (per-record Bernoulli p=0.5 cue injection,
# see ``data_generation/qa_generation/generate_phase9_diverse_qa.py
# --include_answer_cue mix``). Goal: the LoRA learns to attend to
# parenthetical answer-space hints so the eval-time --answer_cue +
# --constrained_decoding combination beats CD alone (which Phase 9 does
# not -- the n=200 ablation shows cue+CD = 46.5% vs CD = 53.5%).
#
# Architectural deltas vs Phase 9:
#   * Same LoRA rank, same prefix_dropout, same free-form-only routing.
#   * LR is HALVED (2e-4 -> 1e-4) and schedule shortened (12 ep -> 4 ep)
#     because we are fine-tuning a converged checkpoint, not training
#     from scratch.
#   * Warm-start covers EVERYTHING the source ckpt has: LoRA + adapter
#     MLP + slot bias. Heads are still dropped (free-form-only).
PHASE10_DEFAULTS: Dict[str, object] = dict(PHASE9_DEFAULTS)  # inherit base recipe


def build_phase10_adapter(
    physics_model: torch.nn.Module,
    *,
    phase9_checkpoint_path: str,
    device: str = 'cuda',
    verbose: bool = True,
    **overrides,
) -> PhysicsLLMAdapterV3:
    """Build a Phase 10 warm-start-everything adapter.

    Differs from ``build_phase9_adapter`` in exactly one respect: the
    LoRA matrices are loaded from the source checkpoint instead of
    being filtered out. Same architecture, same recipe knobs, same
    head-drop policy. The expected source is
    ``adapter_phase9_diverse_grounding.pt``.

    Args
    ----
    physics_model : a ``FullPhysicsFormer`` (or compatible) instance,
        already loaded from ``physics_former_best.pt`` and frozen by
        the caller.
    phase9_checkpoint_path : path to a Phase 9+ V3 checkpoint that
        carries adapter_per_object.*, per_object_slot_bias, AND peft
        LoRA keys (lora_A.*, lora_B.*, base_layer.*).
    device : where to materialise the adapter.
    verbose : prints a transparency banner.
    overrides : any keyword in ``PHASE10_DEFAULTS`` (which inherits
        from ``PHASE9_DEFAULTS``) can be overridden here.

    Returns
    -------
    A ``PhysicsLLMAdapterV3`` ready for the Phase 10 training loop.
    Trainable params are LoRA + adapter_per_object + slot bias (same
    set as Phase 9), but every one of them starts from the Phase 9
    converged value rather than fresh init.
    """
    # Merge defaults with caller overrides.
    cfg = dict(PHASE10_DEFAULTS)
    for k, v in overrides.items():
        if k not in cfg:
            raise ValueError(
                f'Unknown Phase 10 override {k!r}. Known knobs: '
                f'{sorted(cfg.keys())}'
            )
        cfg[k] = v

    if verbose:
        print('=' * 72)
        print('Phase 10 adapter build (warm-start everything from Phase 9)')
        print('=' * 72)
        print(f'  llm_name              : {cfg["llm_name"]}')
        print(f'  tokens_per_object     : {cfg["tokens_per_object"]}')
        print(f'  use_lora              : {cfg["use_lora"]} '
              f'(rank={cfg["lora_rank"]}, alpha={cfg["lora_alpha"]})')
        print(f'  prefix_dropout        : {cfg["prefix_dropout"]}')
        print(f'  warm-start source     : {phase9_checkpoint_path}')
        print()

    # Construct the adapter (same call as Phase 9 -- the recipe knobs
    # are inherited via PHASE10_DEFAULTS = dict(PHASE9_DEFAULTS)).
    adapter = create_adapter_v3(
        physics_model=physics_model,
        include_choices_prob=cfg['include_choices_prob'],
        inject_scene_text=cfg['inject_scene_text'],
        scene_text_style=cfg['scene_text_style'],
        scene_text_include_color=cfg['scene_text_include_color'],
        scene_text_include_material=cfg['scene_text_include_material'],
        descriptor_strip_prob=cfg['descriptor_strip_prob'],
        descriptor_strip_seed=cfg['descriptor_strip_seed'],
        llm_name=cfg['llm_name'],
        physics_dim=cfg['physics_dim'],
        tokens_per_object=cfg['tokens_per_object'],
        freeze_physics=True,
        freeze_llm=True,
        build_mcq_head=cfg['build_mcq_head'],
        build_descriptive_head=cfg['build_descriptive_head'],
        use_lora=cfg['use_lora'],
        lora_rank=cfg['lora_rank'],
        lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'],
        lora_target_modules=list(cfg['lora_target_modules']),
        prefix_dropout=cfg['prefix_dropout'],
        answer_token_dropout=cfg['answer_token_dropout'],
        force_llm_only_routing=cfg['force_llm_only_routing'],
    ).to(device)

    if hasattr(adapter, 'numerical_head'):
        for p in adapter.numerical_head.parameters():
            p.requires_grad = False

    ckpt_path = Path(phase9_checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'Phase 9 checkpoint not found: {ckpt_path}.'
        )

    src = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    src_state = src.get('adapter_state_dict')
    if src_state is None:
        raise KeyError(
            f"Checkpoint at {ckpt_path} has no 'adapter_state_dict' top-level "
            f"key. Got: {sorted(src.keys())}"
        )

    # Phase 10 difference: keep_lora=True so the peft LoRA matrices
    # come along for the ride.
    filtered, counts = _filter_warm_start_state_dict(src_state, keep_lora=True)

    if not filtered:
        raise RuntimeError(
            f'No warm-start keys recovered from {ckpt_path}. The '
            f'checkpoint layout may not match the V3 schema.'
        )

    filtered_dev = {k: v.to(device) for k, v in filtered.items()}
    incompat = adapter.load_state_dict(filtered_dev, strict=False)

    if verbose:
        print(f'  kept adapter MLP keys : {counts["kept_adapter_mlp"]} '
              f'(adapter_per_object.*)')
        print(f'  kept slot bias keys   : {counts["kept_slot_bias"]} '
              f'(per_object_slot_bias)')
        print(f'  kept LoRA keys        : {counts["kept_lora"]} '
              f'(lora_A/lora_B/base_layer; warm-started)')
        print(f'  dropped head keys     : {counts["dropped_head"]} '
              f'(numerical/descriptive/mcq -- not part of Phase 10)')
        print(f'  dropped LLM frozen    : {counts["dropped_llm"]} '
              f'(re-loaded fresh from HF)')
        print(f'  dropped physics keys  : {counts["dropped_physics"]} '
              f'(loaded by caller from physics_former_best.pt)')
        if counts['dropped_other']:
            print(f'  dropped other         : {counts["dropped_other"]}')

        if incompat.missing_keys:
            sample = ', '.join(incompat.missing_keys[:3])
            print(f'  fresh-init params     : {len(incompat.missing_keys)} '
                  f'(e.g. {sample})')
        if incompat.unexpected_keys:
            print(f'  [WARN] unexpected_keys after filter: '
                  f'{incompat.unexpected_keys[:5]}')

        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        total = sum(p.numel() for p in adapter.parameters())
        print(f'  trainable params      : {trainable:,} / {total:,} '
              f'({100.0 * trainable / max(total, 1):.3f}%)')
        print('=' * 72)
        print()

    return adapter


__all__ = [
    'PHASE9_DEFAULTS',
    'PHASE10_DEFAULTS',
    'build_phase9_adapter',
    'build_phase10_adapter',
]
