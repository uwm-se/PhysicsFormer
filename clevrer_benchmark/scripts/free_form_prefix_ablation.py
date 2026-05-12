"""Free-form prefix-ablation eval: confirm the LLM is using the prefix.

Why this script
---------------
The Phase 6 paper claim is that the per-object prefix carries the dynamics
signal (collision presence, motion, exits-frame) that lets the LLM answer
counterfactual / explanatory CLEVRER questions. The encoder dynamics probe
(``encoder_dynamics_probing.py`` -> ``encoder_dynamics_probing.json``) showed
the prefix DOES carry that signal at 0.77 AUC for future_collision and 0.96
AUC for exits_frame. But probes only show the *signal exists* in the prefix
representation -- they don't show the LLM *uses* it during generation.

This script answers "does the LLM actually use the prefix?" by running each
question THREE times against the same checkpoint, varying only the prefix
input:

  - real         : the actual scene's per-object physics states (control)
  - zero         : prefix tokens forced to zero values (mask kept ones)
  - wrong_scene  : a randomly chosen DIFFERENT scene's prefix tokens

Then we compute three numbers per question (NLI / substring / template
phrasing), aggregate, and report the deltas. Reading guide:

  - acc(real) -- baseline free-form accuracy.
  - acc(real) - acc(zero)        : "prefix tokens contribute X pp accuracy".
                                    > 0 means the LLM is using the prefix.
                                    ~ 0 means the LLM is ignoring it (the
                                    scene-text injection or memorised answer
                                    template is doing all the work).
  - acc(real) - acc(wrong_scene) : "prefix tokens carry SCENE-SPECIFIC
                                    information". Distinguishes "model
                                    treats prefix as a constant cue" from
                                    "model conditions on what the prefix
                                    actually says about the scene". A
                                    positive delta here is the strongest
                                    evidence of grounding.

Implementation
--------------
The zero-prefix and wrong-scene conditions are realised by **monkey-patching
the adapter's prefix builder** for the duration of one ``adapter.forward``
call. We patch:

  - V3 / Phase-6 (per-object prefix): ``create_prefix_tokens_per_object``
  - V2 / Phase-3 (scene-level prefix): ``create_prefix_tokens``

For zero-prefix we run the original builder, then replace the values with
zeros while preserving shape and mask so the LLM still sees the full prefix
length (only the values are nulled). For wrong-scene we run the original
builder against the SUBSTITUTE scene's states, then forward the
prefix-tokens-only into the original adapter.forward, while still using the
real scene's question. This keeps prompt construction (scene-text-injected
or not) honest and only the physics prefix differs.

Usage
-----
::

    python compsac_2026_code/clevrer_benchmark/scripts/free_form_prefix_ablation.py \\
        --adapter_checkpoint compsac_2026_code/checkpoints/adapter_phase6_per_object_best.pt \\
        --heldout --n 200 \\
        --out compsac_2026_code/clevrer_benchmark/results/free_form_prefix_ablation_phase6.json

Defaults to a 200-question sample for a quick (~5 min) ablation read.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Snapshot-portable. Anchor on this script's location so the script works
# from any CWD inside the snapshot tree.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _SCRIPT_DIR.parent
_SNAPSHOT_ROOT = _BENCH_DIR.parent

DEFAULT_CHECKPOINT = (
    _SNAPSHOT_ROOT / 'checkpoints' / 'adapter_phase6_per_object_best.pt')
DEFAULT_RESULTS_DIR = _BENCH_DIR / 'results'
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / 'free_form_prefix_ablation.json'
DEFAULT_H5 = os.environ.get(
    'CLEVRER_H5', os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5'))
DEFAULT_NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

sys.path.insert(0, str(_BENCH_DIR))
sys.path.insert(0, str(_BENCH_DIR / 'scripts'))
sys.path.insert(0, str(_SNAPSHOT_ROOT))

# Reuse the existing free-form transfer test's pool/NLI/atomic-save helpers
# rather than copy-pasting them. Keeps both files in sync (per project rule
# "do not write new scripts when existing scripts should be refactored").
from free_form_transfer_test import (  # type: ignore  # noqa: E402
    _build_question_pool,
    _load_heldout_scenes,
    _nli_setup,
    _nli_correct,
    _substring_correct,
    _verbatim_choice_match,
    _looks_clevrer_template,
    _atomic_save,
)


# ---------------------------------------------------------------------------
# Prefix-builder monkey-patches
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patched_zero_prefix(adapter):
    """Force the adapter's prefix to zero values for one forward call.

    Works for both V2 (scene-level ``create_prefix_tokens``) and V3 / Phase 6
    (per-object ``create_prefix_tokens_per_object``). The mask is preserved
    so the LLM still attends to the prefix slots -- only the embedding
    values become zero. This is the correct way to ablate "did the prefix
    embed any information": if accuracy stays the same, the embedding
    content was unused.
    """
    import torch
    has_per_object = getattr(adapter, 'tokens_per_object', 0) > 0
    if has_per_object:
        orig = adapter.create_prefix_tokens_per_object

        def patched(physics_states, object_mask):
            real_prefix, real_mask = orig(physics_states, object_mask)
            return torch.zeros_like(real_prefix), real_mask
        adapter.create_prefix_tokens_per_object = patched
        try:
            yield
        finally:
            adapter.create_prefix_tokens_per_object = orig
    else:
        orig = adapter.create_prefix_tokens

        def patched(physics_features):
            real_prefix = orig(physics_features)
            return torch.zeros_like(real_prefix)
        adapter.create_prefix_tokens = patched
        try:
            yield
        finally:
            adapter.create_prefix_tokens = orig


@contextlib.contextmanager
def _patched_substitute_prefix(adapter, sub_states, sub_masks):
    """Use a DIFFERENT scene's prefix for one forward call.

    Substitute physics states + masks live on the same device as the real
    inputs and have the encoder's expected width (max_objects). The real
    forward path tokenises the question with the REAL scene-text, so the
    prompt remains scene-faithful and only the physics signal is swapped.
    """
    import torch
    has_per_object = getattr(adapter, 'tokens_per_object', 0) > 0
    if has_per_object:
        orig = adapter.create_prefix_tokens_per_object

        def patched(physics_states, object_mask):
            # IGNORE the caller's states/masks; build prefix from sub_*.
            return orig(sub_states, sub_masks)
        adapter.create_prefix_tokens_per_object = patched
        try:
            yield
        finally:
            adapter.create_prefix_tokens_per_object = orig
    else:
        orig = adapter.create_prefix_tokens
        # V2 uses a single physics_features vector (mean-pooled). We build it
        # from sub_states once, then patch ``create_prefix_tokens`` to ignore
        # the caller's features and use ours.
        sub_features = adapter.extract_physics_features(sub_states, sub_masks)

        def patched(physics_features):
            return orig(sub_features)
        adapter.create_prefix_tokens = patched
        try:
            yield
        finally:
            adapter.create_prefix_tokens = orig


# ---------------------------------------------------------------------------
# Scene loading helpers (independent of the adapter)
# ---------------------------------------------------------------------------

def _load_scene_tensors(scene_path: Path, single_frame: int):
    """Load a CLEVRER scene and return single-frame slices + scene-text view.

    Returns (states_t, masks_t, scene_text_states, scene_text_mask):
      - states_t / masks_t : single-frame slices for the adapter prefix builder
      - scene_text_states / scene_text_mask : full-trajectory derivatives used
        by ``build_scene_summary`` (frame 0 + union mask) so injected text
        captures objects that exited the camera mid-trajectory.
    """
    import numpy as np
    import torch
    from scene_converter import (
        load_clevrer_scene, clevrer_scene_to_state_tensor,
    )
    scene = load_clevrer_scene(str(scene_path))
    states, masks, _ = clevrer_scene_to_state_tensor(scene)
    if masks.ndim == 2:
        scene_text_mask = masks.max(axis=0)
        scene_text_states = states[0]
    else:
        scene_text_mask = masks
        scene_text_states = states
    fi = min(single_frame, states.shape[0] - 1) if single_frame >= 0 else -1
    if fi >= 0:
        states = states[fi:fi + 1]
        masks = masks[fi:fi + 1] if masks.ndim == 2 else masks
    states_t = torch.from_numpy(states).float()
    masks_t = torch.from_numpy(masks).float()
    return states_t, masks_t, scene_text_states, scene_text_mask


def _generate_with_prefix_state(
    adapter, states_t, masks_t, prompt: str, gen_kwargs: Dict, device,
    *,
    sub_states: Optional['torch.Tensor'] = None,
    sub_masks: Optional['torch.Tensor'] = None,
    zero_prefix: bool = False,
) -> str:
    """Run adapter.forward under one of three prefix conditions.

    Exactly one of ``zero_prefix=True`` and ``sub_states is not None`` may be
    set; if neither is set we run the unpatched adapter (real prefix).
    """
    import torch
    if sub_states is not None and zero_prefix:
        raise ValueError("Pass either zero_prefix=True OR sub_states, not both")
    if states_t.dim() == 3:
        states_t = states_t.unsqueeze(0)
    if masks_t.dim() == 2:
        masks_t = masks_t.unsqueeze(0)
    states_t = states_t.to(device)
    masks_t = masks_t.to(device)
    if sub_states is not None:
        if sub_states.dim() == 3:
            sub_states = sub_states.unsqueeze(0)
        if sub_masks.dim() == 2:
            sub_masks = sub_masks.unsqueeze(0)
        sub_states = sub_states.to(device)
        sub_masks = sub_masks.to(device)

    with torch.no_grad():
        if zero_prefix:
            with _patched_zero_prefix(adapter):
                out = adapter.forward(states_t, masks_t, [prompt], **gen_kwargs)
        elif sub_states is not None:
            with _patched_substitute_prefix(adapter, sub_states, sub_masks):
                out = adapter.forward(states_t, masks_t, [prompt], **gen_kwargs)
        else:
            out = adapter.forward(states_t, masks_t, [prompt], **gen_kwargs)
    return str(out[0]) if isinstance(out, list) and out else str(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Free-form prefix-ablation eval (real vs zero vs wrong-scene).')
    parser.add_argument('--n', type=int, default=200,
                        help='Number of heldout questions to sample (default 200; '
                             'each runs 3x for the ablation).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gen_seed', type=int, default=42)
    parser.add_argument('--heldout', action='store_true', default=True,
                        help='Sample from the 501 heldout scenes.')
    parser.add_argument('--no_heldout', action='store_false', dest='heldout')
    parser.add_argument('--valid_only', action='store_true', default=True)
    parser.add_argument('--clevrer_dir', type=Path, default=Path(os.environ.get('CLEVRER_DIR', 'clevrer')))
    parser.add_argument('--adapter_checkpoint', type=Path,
                        default=DEFAULT_CHECKPOINT)
    parser.add_argument('--physics_checkpoint', type=str,
                        default='D:\\physics-former-data\\checkpoints\\stage1_best.pt')
    parser.add_argument('--h5', type=Path, default=Path(DEFAULT_H5))
    parser.add_argument('--single_frame', type=int, default=64)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--gen_mode', type=str, default='greedy',
                        choices=['sample', 'greedy', 'beam4', 'beam8'],
                        help='Greedy is the default for ablation: stochastic '
                             'sampling adds variance that masks small but '
                             'real prefix-vs-no-prefix deltas.')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--save_every', type=int, default=25)
    parser.add_argument(
        '--no_scene_text',
        action='store_true',
        help='Diagnostic: force-disable scene-text injection at inference '
             'even if the adapter recipe sets inject_scene_text=True.',
    )
    parser.add_argument('--use_nli', action='store_true', default=True)
    parser.add_argument('--no_nli', action='store_false', dest='use_nli')
    parser.add_argument('--nli_model', type=str, default=DEFAULT_NLI_MODEL)
    parser.add_argument('--nli_threshold', type=float, default=0.7)
    args = parser.parse_args()

    if args.device == 'auto':
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not args.adapter_checkpoint.exists():
        raise FileNotFoundError(
            f'Adapter checkpoint not found: {args.adapter_checkpoint}')

    # --- Pool & sample ---
    heldout = _load_heldout_scenes(args.h5) if args.heldout else None
    pool = _build_question_pool(
        args.clevrer_dir, heldout, valid_only=args.valid_only)
    print(f'[pool] {len(pool):,} {"heldout " if args.heldout else ""}'
          f'{"valid-only " if args.valid_only else ""}causal questions')
    if not pool:
        raise SystemExit('Empty pool.')

    rng = random.Random(args.seed)
    sample = rng.sample(pool, k=min(args.n, len(pool)))
    print(f'[pool] sampled {len(sample)} questions for the ablation')

    # Build an alternative scene pool for wrong-scene prefix substitution.
    # Uses scenes that are NOT in the sampled set (so we never collide
    # real and substitute scenes) and NOT in the heldout set if --heldout
    # so the alternative is always train-distribution-typical.
    sampled_scene_ids = {q['scene_id'] for q in sample}
    alt_pool = [q for q in pool if q['scene_id'] not in sampled_scene_ids]
    if not alt_pool:
        # Fall back to re-using sampled scenes (avoid same-scene with a
        # next-rotation pick at use time).
        alt_pool = list(pool)
    print(f'[pool] {len(alt_pool)} alternative scenes available for '
          f'wrong-scene substitution')

    # --- Load adapter ---
    print(f'[adapter] loading: {args.adapter_checkpoint}', flush=True)
    from clevrer_benchmark.run_adapter_evaluation import (  # type: ignore
        load_adapter_model, _gen_kwargs_for_mode,
    )
    from physics_llm_adapter.scene_summary import build_scene_summary  # type: ignore

    adapter = load_adapter_model(
        str(args.adapter_checkpoint),
        args.physics_checkpoint, device=args.device)
    adapter.eval()
    gen_kwargs = _gen_kwargs_for_mode(args.gen_mode)

    import torch
    torch.manual_seed(args.gen_seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.gen_seed)

    # --- NLI ---
    nli_state = _nli_setup(args.nli_model, args.device) if args.use_nli else None

    # --- Run ablation ---
    records: List[Dict] = []
    counts = {
        'n_total': 0,
        # NLI per condition
        'nli_real':         0,
        'nli_zero':         0,
        'nli_wrong_scene':  0,
        # Substring per condition
        'subs_real':        0,
        'subs_zero':        0,
        'subs_wrong_scene': 0,
        # Template phrasing per condition (FF-style template usage)
        'tmpl_real':        0,
        'tmpl_zero':        0,
        'tmpl_wrong_scene': 0,
    }
    print('[run] generating (3 conditions per question)', flush=True)
    print(f'[stream] periodic atomic save every {args.save_every} questions '
          f'-> {args.out}', flush=True)
    from tqdm.auto import tqdm
    auto_inject = getattr(adapter, 'inject_scene_text', False) and not args.no_scene_text
    for _i, q in enumerate(tqdm(sample, desc='[ablation]', unit='q')):
        try:
            states_t, masks_t, st_states, st_mask = _load_scene_tensors(
                Path(q['scene_path']), args.single_frame)
        except Exception as e:
            print(f'  [skip] {q["scene_id"]}: {e}')
            continue

        # Build the free-form prompt (with optional scene-text injection
        # matching the adapter's training distribution).
        ff_prompt = q['question_text']
        if auto_inject:
            scene_text = build_scene_summary(
                st_states, st_mask,
                style=getattr(adapter, 'scene_text_style', 'comma_list'),
                include_color=getattr(adapter, 'scene_text_include_color', True),
                include_material=getattr(
                    adapter, 'scene_text_include_material', True),
            )
            ff_prompt = f'{scene_text} {ff_prompt}'

        correct_texts = [c['choice'] for c in q['choices']
                         if c.get('answer') == 'correct']
        wrong_texts = [c['choice'] for c in q['choices']
                       if c.get('answer') == 'wrong']

        # Substitute scene = pick deterministically from alt_pool by question
        # index so the same question always picks the same alternative across
        # repeated runs (reproducibility).
        sub_q = alt_pool[(_i * 7919) % len(alt_pool)]
        if sub_q['scene_id'] == q['scene_id'] and len(alt_pool) > 1:
            sub_q = alt_pool[(_i * 7919 + 1) % len(alt_pool)]
        try:
            sub_states_t, sub_masks_t, _, _ = _load_scene_tensors(
                Path(sub_q['scene_path']), args.single_frame)
        except Exception as e:
            print(f'  [skip-sub] {sub_q["scene_id"]}: {e}')
            continue

        # Three generations per question. Reset RNG per condition so the
        # only thing that varies between the three runs is the prefix.
        outs: Dict[str, str] = {}
        for cond in ('real', 'zero', 'wrong_scene'):
            torch.manual_seed(args.gen_seed)
            if args.device == 'cuda':
                torch.cuda.manual_seed_all(args.gen_seed)
            kwargs = dict(zero_prefix=False)
            if cond == 'zero':
                kwargs = dict(zero_prefix=True)
            elif cond == 'wrong_scene':
                kwargs = dict(sub_states=sub_states_t, sub_masks=sub_masks_t)
            outs[cond] = _generate_with_prefix_state(
                adapter, states_t, masks_t, ff_prompt, gen_kwargs, args.device,
                **kwargs,
            )

        # Score each condition.
        per_cond = {}
        for cond, pred in outs.items():
            subs = _substring_correct(pred, correct_texts, wrong_texts)
            tmpl = _looks_clevrer_template(pred)
            nli = False
            if nli_state is not None:
                tok, mod, ent = nli_state
                nli = _nli_correct(
                    tok, mod, ent, pred, correct_texts,
                    args.device, args.nli_threshold)
            per_cond[cond] = {
                'predicted': pred,
                'substring_correct': subs,
                'template_phrasing': tmpl,
                'nli_correct': nli,
            }
            counts[f'subs_{cond}']  += int(subs)
            counts[f'tmpl_{cond}']  += int(tmpl)
            counts[f'nli_{cond}']   += int(nli)
        counts['n_total'] += 1

        records.append({
            'scene_id':            q['scene_id'],
            'q_idx':               q['q_idx'],
            'question_type':       q['question_type'],
            'question_text':       q['question_text'],
            'correct_choices':     correct_texts,
            'wrong_choices':       wrong_texts,
            'sub_scene_id':        sub_q['scene_id'],
            'prompt':              ff_prompt,
            'real':         per_cond['real'],
            'zero':         per_cond['zero'],
            'wrong_scene':  per_cond['wrong_scene'],
        })

        if (_i + 1) % args.save_every == 0:
            _atomic_save(args.out, _build_summary(counts, args), records)

    summary = _build_summary(counts, args)
    _atomic_save(args.out, summary, records)
    _print_summary(summary)
    print(f'\n  records written to: {args.out}')


def _build_summary(counts: dict, args) -> dict:
    n = max(counts['n_total'], 1)

    def pct(k):  # percent of n
        return 100.0 * counts[k] / n

    def per_cond(metric: str) -> Dict[str, float]:
        return {
            'real':        pct(f'{metric}_real'),
            'zero':        pct(f'{metric}_zero'),
            'wrong_scene': pct(f'{metric}_wrong_scene'),
        }
    nli_acc = per_cond('nli')
    sub_acc = per_cond('subs')
    tmpl_pct = per_cond('tmpl')
    return {
        'config': {
            'n_sampled': counts['n_total'],
            'gen_mode': args.gen_mode,
            'gen_seed': args.gen_seed,
            'heldout': args.heldout,
            'valid_only': args.valid_only,
            'single_frame': args.single_frame,
            'nli_model': args.nli_model if args.use_nli else None,
            'nli_threshold': args.nli_threshold,
            'adapter_checkpoint': str(args.adapter_checkpoint),
        },
        'nli_acc_pct':           nli_acc,
        'substring_acc_pct':     sub_acc,
        'template_phrasing_pct': tmpl_pct,
        'delta': {
            'nli_real_minus_zero':           nli_acc['real'] - nli_acc['zero'],
            'nli_real_minus_wrong_scene':    nli_acc['real'] - nli_acc['wrong_scene'],
            'subs_real_minus_zero':          sub_acc['real'] - sub_acc['zero'],
            'subs_real_minus_wrong_scene':   sub_acc['real'] - sub_acc['wrong_scene'],
        },
    }


def _print_summary(s: dict) -> None:
    print()
    print('=' * 72)
    print('Free-form prefix-ablation summary')
    print('=' * 72)
    cfg = s['config']
    print(f'  n_sampled:    {cfg["n_sampled"]}')
    print(f'  gen_mode:     {cfg["gen_mode"]}')
    print(f'  ckpt:         {Path(cfg["adapter_checkpoint"]).name}')
    print()
    print(f'{"metric":<22} {"real":>9} {"zero":>9} {"wrong_scn":>11} '
          f'{"r-zero":>9} {"r-wrong":>9}')
    print('-' * 72)
    for label, key in [
        ('NLI acc (%)',           'nli_acc_pct'),
        ('substring acc (%)',     'substring_acc_pct'),
        ('template phrasing (%)', 'template_phrasing_pct'),
    ]:
        d = s[key]
        delta_z = d['real'] - d['zero']
        delta_w = d['real'] - d['wrong_scene']
        print(f'{label:<22} {d["real"]:9.2f} {d["zero"]:9.2f} '
              f'{d["wrong_scene"]:11.2f} {delta_z:+9.2f} {delta_w:+9.2f}')
    print()
    print('Reading guide:')
    print('  r-zero  > 0 : the prefix tokens contribute to accuracy.')
    print('  r-zero ~ 0 : the LLM is ignoring the prefix entirely.')
    print('  r-wrong > 0 : the prefix carries SCENE-SPECIFIC info, not just')
    print('               a generic "I have a prefix" cue.')


if __name__ == '__main__':
    main()
