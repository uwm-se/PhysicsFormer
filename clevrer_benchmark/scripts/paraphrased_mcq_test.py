"""Paraphrased-MCQ test: probe whether the Phase 3 adapter does physics-grounded
discrimination over choices, or template-pattern memorization.

Procedure
---------
For each held-out question, we run the adapter four times with identical scene
state but DIFFERENT surface forms of the choice menu:

    control : original CLEVRER MCQ choices (e.g. "the cube collides with the sphere")
    T1      : mild verb swap            ("the cube bumps into the sphere")
    T2      : structural rewording      ("the cube makes contact with the sphere")
    T3      : aggressive rewording      ("the cube ricochets off the sphere")

The paraphrase mappings replace ONLY the CLEVRER template phrases ('collide',
'enters the scene', 'presence of', etc.); object naming, colors, and scene
entities are preserved unchanged. The same mapping is applied uniformly to
EVERY choice in the menu so the test isn't accidentally easier or harder.

Hypotheses
----------
- (A) physics-grounded discrimination: model evaluates each choice's plausibility
  against scene physics. Accuracy should stay near control across all tiers.
- (B) template-pattern memorization: model recognizes the canonical CLEVRER
  template and pattern-matches its memorized correct-choice distribution.
  Accuracy should degrade as paraphrase aggressiveness rises.

Smoking-gun signal: if the model emits the *original* CLEVRER phrasing as its
answer EVEN WHEN the menu only contains paraphrased options, that's direct
evidence the model is regenerating its memorized training-template rather than
selecting from the offered menu. We track this as ``original_template_fallback``.

Usage
-----
::

    python compsac_2026_code/clevrer_benchmark/scripts/paraphrased_mcq_test.py \\
        --n 10 --stratified

Snapshot-portable: paths anchor on ``__file__`` so any CWD works.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Snapshot-portable defaults.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _SCRIPT_DIR.parent
_SNAPSHOT_ROOT = _BENCH_DIR.parent

DEFAULT_CHECKPOINT = _SNAPSHOT_ROOT / 'checkpoints' / 'adapter_phase3.pt'
DEFAULT_OUTPUT = _BENCH_DIR / 'results' / 'paraphrased_mcq_test.json'
DEFAULT_H5 = os.environ.get(
    'CLEVRER_H5',
    os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5'),
)

sys.path.insert(0, str(_BENCH_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))
# Snapshot root needed for ``from physics_llm_adapter.scene_summary``,
# which the inference loop uses to mirror the training-time prompt
# distribution for inject_scene_text=True checkpoints (Phase 7+).
sys.path.insert(0, str(_SNAPSHOT_ROOT))


# =============================================================================
# Paraphrase rules
# =============================================================================
# Each rule is (regex_pattern, T1_replacement, T2_replacement, T3_replacement).
# Patterns operate on lowercase text and are applied left-to-right; longer/more
# specific patterns are listed first so they take precedence over shorter ones.

PARAPHRASE_RULES: List[Tuple[str, str, str, str]] = [
    # "the collision between the X and the Y" -- nominalized form
    (r'\bthe collision between\b',
     'the bump between',
     'the impact involving',
     'the percussive interaction between'),

    # "the X's colliding with the Y" -- gerund possessive
    (r"'s colliding with\b",
     "'s bumping into",
     "'s making contact with",
     "'s ricocheting off"),

    # "the X collides with the Y"
    (r'\bcollides with\b',
     'bumps into',
     'makes contact with',
     'ricochets off'),

    # "X colliding with Y" (non-possessive gerund)
    (r'\bcolliding with\b',
     'bumping into',
     'making contact with',
     'ricocheting off'),

    # "the X and the Y collide"  -- bare verb (no preposition)
    (r'\bcollide\b',
     'bump into each other',
     'make contact with each other',
     'undergo mutual impact'),

    # "the X enters the scene"
    (r'\benters the scene\b',
     'comes onto the scene',
     'appears on the field',
     'materializes within view'),

    # "the X's entering"
    (r"'s entering\b",
     "'s appearing",
     "'s arriving",
     "'s materializing"),

    # "the X's entrance" -- possessive nominalization
    (r"'s entrance\b",
     "'s appearance",
     "'s arrival",
     "'s materialization"),

    # "the presence of the X" -- presence prepositional phrase
    (r'\bthe presence of\b',
     'the existence of',
     'the fact of including',
     'the contribution of having'),

    # "the X exits the scene"
    (r'\bexits the scene\b',
     'leaves the scene',
     'departs from the field',
     'disappears from view'),
]


def apply_paraphrase(text: str, tier: int) -> str:
    """Apply tier-N paraphrase rules to ``text`` (case-preserving for the first letter).

    tier in {0 (control / no-op), 1, 2, 3}.
    Operates on the lowercased text then re-capitalizes the first character
    if the original text started with an uppercase letter.
    """
    if tier == 0 or not text:
        return text
    original = text
    capitalize = original[:1].isupper()
    s = original.lower()
    for pat, t1, t2, t3 in PARAPHRASE_RULES:
        replacement = (t1, t2, t3)[tier - 1]
        s = re.sub(pat, replacement, s)
    if capitalize:
        s = s[:1].upper() + s[1:]
    return s


# =============================================================================
# Pool / scene loading -- mirrors free_form_transfer_test.py
# =============================================================================

# Scene-id parsing is now in scoring/referent_equiv (single source of truth).
# Imported under the original underscore-prefixed name so the rest of this
# script is unchanged.
from scoring.referent_equiv import scene_id_to_num as _scene_id_to_num  # noqa: E402


def _build_question_pool(clevrer_dir: Path,
                         heldout: Optional[set],
                         valid_only: bool) -> List[Dict]:
    scene_dir = None
    for c in [clevrer_dir / 'scenes' / 'validation',
              clevrer_dir / 'scenes' / 'clevrer_scenes',
              clevrer_dir / 'scenes']:
        if c.exists() and any(c.glob('*.json')):
            scene_dir = c
            break
    if scene_dir is None:
        raise FileNotFoundError(f'No CLEVRER scene dir under {clevrer_dir}')

    questions_file = clevrer_dir / 'questions' / 'clevrer_validation.json'
    if not questions_file.exists():
        questions_file = clevrer_dir / 'questions' / 'validation.json'
    if not questions_file.exists():
        raise FileNotFoundError(f'No CLEVRER validation questions in {clevrer_dir/"questions"}')

    with open(questions_file, 'r') as f:
        all_q = json.load(f)
    from collections import defaultdict
    qbs = defaultdict(list)
    for sd in all_q:
        sid = sd.get('scene_index', sd.get('video_index', 0))
        qbs[sid] = sd.get('questions', [])

    causal_types = {'explanatory', 'predictive', 'counterfactual'}
    pool: List[Dict] = []
    for scene_path in sorted(scene_dir.glob('*.json')):
        scene_id = scene_path.stem
        scene_num = _scene_id_to_num(scene_id)
        if scene_num is None:
            continue
        if heldout is not None and scene_num not in heldout:
            continue
        for q_idx, q in enumerate(qbs.get(scene_num, [])):
            qtype = (q.get('question_type') or '').lower()
            if qtype not in causal_types:
                continue
            choices = q.get('choices') or []
            if not choices:
                continue
            if valid_only and not any(c.get('answer') == 'correct' for c in choices):
                continue
            pool.append({
                'scene_path': scene_path,
                'scene_id': scene_id,
                'q_idx': q_idx,
                'question_text': q.get('question', ''),
                'question_type': qtype,
                'choices': choices,
            })
    return pool


def _stratified_sample(pool: List[Dict], n: int, seed: int) -> List[Dict]:
    """Stratified sample across question types for balanced coverage."""
    rng = random.Random(seed)
    by_type: Dict[str, List[Dict]] = {}
    for q in pool:
        by_type.setdefault(q['question_type'], []).append(q)
    types = sorted(by_type.keys())
    per_type = max(1, n // len(types))
    sample: List[Dict] = []
    for t in types:
        bucket = by_type[t][:]
        rng.shuffle(bucket)
        sample.extend(bucket[:per_type])
    # Top up if we under-shot due to integer division; over-shoot is fine.
    if len(sample) < n:
        leftover = [q for q in pool if q not in sample]
        rng.shuffle(leftover)
        sample.extend(leftover[: n - len(sample)])
    return sample[:n]


# =============================================================================
# Scoring helpers
# =============================================================================

# These three helpers are byte-identical across the eval and audit scripts;
# they live in scoring/text_match now. Imported under their original names
# so the rest of this file (and any external script that imports them from
# here) keeps working without changes.
from scoring.text_match import (  # noqa: E402
    _norm,
    _substring_correct,
    _detects_original_template_fallback,
)


def _generate(adapter, states, masks, prompt: str, gen_kwargs, device) -> str:
    import torch
    if states.dim() == 3:
        states = states.unsqueeze(0)
    if masks.dim() == 2:
        masks = masks.unsqueeze(0)
    states = states.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        out = adapter.forward(states, masks, [prompt], **gen_kwargs)
    if isinstance(out, list):
        return str(out[0])
    return str(out)


# =============================================================================
# Main
# =============================================================================

def _atomic_save(path: Path, summary: dict, records: list, paraphrase_rules: list) -> None:
    """Atomically write {summary, records, paraphrase_rules} so a kill mid-flush
    leaves a valid file (either prior or new state, never half-written)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'records': records,
                   'paraphrase_rules': paraphrase_rules},
                  f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _summary_from_counts(counts: dict) -> dict:
    """Build the per-tier summary dict from current counts. Safe to call mid-run."""
    return {tier: {
        'tier_label': ['control', 'T1_mild', 'T2_struct', 'T3_aggressive'][tier],
        'n': counts[tier]['n'],
        'substring_acc_pct': round(100.0 * counts[tier]['substr_correct'] / max(counts[tier]['n'], 1), 1),
        'original_template_fallback_pct': round(100.0 * counts[tier]['orig_template_fallback'] / max(counts[tier]['n'], 1), 1),
        'unknown_pct': round(100.0 * counts[tier]['unknown'] / max(counts[tier]['n'], 1), 1),
    } for tier in (0, 1, 2, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Paraphrased-MCQ test for the Phase 3 adapter.')
    parser.add_argument('--n', type=int, default=10,
                        help='Number of held-out questions to test (default 10).')
    parser.add_argument('--save_every', type=int, default=25,
                        help='Periodic incremental save cadence in questions (default 25). '
                             'Each question runs all 4 tiers, so a kill mid-run loses at most '
                             'save_every*4 generations of work.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stratified', action='store_true', default=True,
                        help='Stratify sample across question types (default on).')
    parser.add_argument('--no_stratified', action='store_false', dest='stratified')
    parser.add_argument('--heldout', action='store_true', default=True,
                        help='Sample only from the 501 held-out scenes (default on).')
    parser.add_argument('--no_heldout', action='store_false', dest='heldout')
    parser.add_argument('--clevrer_dir', type=Path, default=Path(os.environ.get('CLEVRER_DIR', 'clevrer')))
    parser.add_argument('--adapter_checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--physics_checkpoint', type=str,
                        default='D:\\physics-former-data\\checkpoints\\stage1_best.pt')
    parser.add_argument('--h5', type=Path, default=Path(DEFAULT_H5))
    parser.add_argument('--single_frame', type=int, default=64)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--gen_mode', type=str, default='sample',
                        choices=['sample', 'greedy', 'beam4', 'beam8'])
    parser.add_argument('--gen_seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.device == 'auto':
        try:
            import torch
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            args.device = 'cpu'

    if not args.adapter_checkpoint.exists():
        raise FileNotFoundError(f'Adapter checkpoint not found: {args.adapter_checkpoint}')

    # --- Pool & sample ------------------------------------------------------
    heldout = None
    if args.heldout:
        from compute_paper_stats import load_heldout_scenes  # type: ignore
        heldout = load_heldout_scenes(args.h5)
    pool = _build_question_pool(args.clevrer_dir, heldout, valid_only=True)
    print(f'[pool] {len(pool):,} held-out valid-only causal questions')
    sample = (_stratified_sample(pool, args.n, args.seed)
              if args.stratified else
              random.Random(args.seed).sample(pool, k=min(args.n, len(pool))))
    type_dist = {}
    for q in sample:
        type_dist[q['question_type']] = type_dist.get(q['question_type'], 0) + 1
    print(f'[pool] sampled {len(sample)}: {type_dist}')

    # --- Load adapter -------------------------------------------------------
    print('[adapter] loading ...', flush=True)
    from run_adapter_evaluation import load_adapter_model, _gen_kwargs_for_mode
    from scene_converter import load_clevrer_scene, clevrer_scene_to_state_tensor
    # Scene-text injection at inference: mirrors free_form_transfer_test.py.
    # ``PhysicsLLMAdapterV3.forward`` does NOT auto-prepend scene-text
    # (per its docstring); when adapter.inject_scene_text=True (Phase 7+
    # default), the eval caller MUST prepend a deterministic scene summary
    # so the prompt distribution matches training. Skipping this produces
    # incoherent generations (the symptom that hid the same bug in
    # eval_phase9_heldout_type.py for one full eval pass).
    from physics_llm_adapter.scene_summary import build_scene_summary  # type: ignore
    import torch
    adapter = load_adapter_model(str(args.adapter_checkpoint),
                                 args.physics_checkpoint, device=args.device)
    gen_kwargs = _gen_kwargs_for_mode(args.gen_mode)
    torch.manual_seed(args.gen_seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.gen_seed)

    # --- Run all four prompt variants per question --------------------------
    rng = random.Random(args.seed + 1)
    records: List[Dict] = []
    counts = {tier: {'n': 0, 'substr_correct': 0, 'orig_template_fallback': 0,
                     'unknown': 0}
              for tier in (0, 1, 2, 3)}

    print('[run] generating ...', flush=True)
    print(f'[stream] periodic atomic save every {args.save_every} questions -> {args.out}',
          flush=True)
    paraphrase_rules_serialized = [(p, t1, t2, t3) for p, t1, t2, t3 in PARAPHRASE_RULES]
    from tqdm.auto import tqdm
    # Detect once whether the adapter was trained with scene-text injection.
    # Cheap; same value used for every question.
    auto_inject = getattr(adapter, 'inject_scene_text', False)
    inject_style = getattr(adapter, 'scene_text_style', 'comma_list')
    inject_color = getattr(adapter, 'scene_text_include_color', True)
    inject_material = getattr(adapter, 'scene_text_include_material', True)
    if auto_inject:
        print(f'[adapter] inject_scene_text=True (style={inject_style!r}, '
              f'color={inject_color}, material={inject_material}) -- '
              f'prepending scene summary to every prompt to match training '
              f'distribution.', flush=True)
    for _i, q in enumerate(tqdm(sample, desc='[paraphrased]', unit='q')):
        try:
            scene = load_clevrer_scene(str(q['scene_path']))
            states, masks, _ = clevrer_scene_to_state_tensor(scene)
        except Exception as e:
            print(f'  [skip] {q["scene_id"]}: {e}')
            continue
        if args.single_frame is not None and args.single_frame >= 0:
            fi = min(args.single_frame, states.shape[0] - 1)
            states = states[fi:fi + 1]
            masks = masks[fi:fi + 1] if masks.ndim == 2 else masks
        states_t = torch.from_numpy(states).float()
        masks_t = torch.from_numpy(masks).float()

        choices = q['choices']
        original_texts = [c.get('choice', c) if isinstance(c, dict) else str(c)
                          for c in choices]
        labels = [c.get('answer') for c in choices]
        original_correct = [t for t, l in zip(original_texts, labels) if l == 'correct']
        original_wrong = [t for t, l in zip(original_texts, labels) if l == 'wrong']

        # Same shuffle order used for every tier so positional bias is held constant.
        order = list(range(len(original_texts)))
        rng.shuffle(order)
        shuffled_originals = [original_texts[i] for i in order]

        record_q = {
            'scene_id': q['scene_id'],
            'q_idx': q['q_idx'],
            'question_type': q['question_type'],
            'question_text': q['question_text'],
            'original_correct': original_correct,
            'original_wrong': original_wrong,
            'tiers': {},
        }

        # Build the scene-text prefix once per question (the scene is the
        # same across all four tiers) so we don't re-pay the description
        # cost per tier. We use the FULL-trajectory states + union mask
        # exactly as free_form_transfer_test.py does so any object that
        # exits the camera mid-trajectory still ends up in the description.
        # ``states`` and ``masks`` here are the original full-traj numpy
        # arrays from clevrer_scene_to_state_tensor (the single_frame slice
        # made above only affected ``states``/``masks`` if --single_frame
        # was passed; we re-derive st_states/st_mask from the full pair).
        if auto_inject:
            # ``states`` may be the full [T, N, 35] traj or a [1, N, 35]
            # single-frame slice depending on --single_frame. ``masks``
            # similarly. Either way, frame 0 of states + union of masks
            # over time gives the scene-composition view we want.
            st_states = states[0] if states.ndim == 3 else states
            st_mask = (masks.max(axis=0) > 0.5) if masks.ndim == 2 else (masks > 0.5)
            scene_text = build_scene_summary(
                st_states, st_mask.astype('float32'),
                style=inject_style,
                include_color=inject_color,
                include_material=inject_material,
            )
        else:
            scene_text = None

        for tier in (0, 1, 2, 3):
            tier_choices = [apply_paraphrase(c, tier) for c in shuffled_originals]
            tier_correct = [apply_paraphrase(c, tier) for c in original_correct]
            tier_wrong = [apply_paraphrase(c, tier) for c in original_wrong]
            base_prompt = (q['question_text'] + ' Options: ' +
                           ', '.join(tier_choices))
            prompt = (f'{scene_text} {base_prompt}'
                      if scene_text is not None else base_prompt)
            pred = _generate(adapter, states_t, masks_t, prompt, gen_kwargs, args.device)

            substr_ok = _substring_correct(pred, tier_correct, tier_wrong)
            orig_fallback = (tier > 0 and
                             _detects_original_template_fallback(
                                 pred, original_correct, original_wrong,
                                 tier_correct, tier_wrong))
            is_unknown = _norm(pred).startswith('unknown')

            counts[tier]['n'] += 1
            counts[tier]['substr_correct'] += int(substr_ok)
            counts[tier]['orig_template_fallback'] += int(orig_fallback)
            counts[tier]['unknown'] += int(is_unknown)

            record_q['tiers'][f't{tier}'] = {
                'tier_label': ['control', 'T1_mild', 'T2_struct', 'T3_aggressive'][tier],
                'paraphrased_choices': tier_choices,
                'paraphrased_correct': tier_correct,
                'paraphrased_wrong': tier_wrong,
                'prompt': prompt,
                'predicted': pred,
                'substring_correct': substr_ok,
                'original_template_fallback': orig_fallback,
                'is_unknown': is_unknown,
            }
        records.append(record_q)

        # Periodic incremental save: lose at most one save_every cohort on kill.
        if (_i + 1) % args.save_every == 0:
            _atomic_save(args.out, _summary_from_counts(counts), records,
                         paraphrase_rules_serialized)

    # --- Summary -----------------------------------------------------------
    n = max(counts[0]['n'], 1)
    summary = _summary_from_counts(counts)

    print()
    print('=== Paraphrased-MCQ test summary ===')
    print(f'  n = {n}, sample by question type: {type_dist}')
    print()
    print(f'  {"tier":<18} {"substr acc":>12} {"orig-tpl fallback":>20} {"unknown":>10}')
    print('  ' + '-' * 64)
    for tier in (0, 1, 2, 3):
        s = summary[tier]
        ft = s["original_template_fallback_pct"]
        ft_str = '   --' if tier == 0 else f'{ft:>15.1f}%'
        print(f'  {s["tier_label"]:<18} {s["substring_acc_pct"]:>10.1f}%   {ft_str}   {s["unknown_pct"]:>8.1f}%')
    print()

    # Save JSON FIRST so a console-encoding hiccup in the print loop doesn't
    # lose the data (Windows cp1252 sometimes mangles model-generated text).
    _atomic_save(args.out, summary, records, paraphrase_rules_serialized)
    print(f'\n  records written to: {args.out}')

    # Per-question detail dump (ASCII-only marks for Windows cp1252 consoles).
    print()
    print('=== Per-question outputs ===')
    for r in records:
        try:
            print(f'\n  [{r["question_type"]}] scene={r["scene_id"]} q{r["q_idx"]}')
            print(f'  Q: {r["question_text"][:110]}')
            print(f'  correct(orig): {r["original_correct"]}')
            for tkey in ('t0', 't1', 't2', 't3'):
                t = r['tiers'][tkey]
                if t['substring_correct']:
                    mark = '[OK]'
                elif t['original_template_fallback']:
                    mark = '[FALLBACK]'
                elif t['is_unknown']:
                    mark = '[UNKNOWN]'
                else:
                    mark = '[wrong]'
                # Sanitize predicted text for Windows console safety.
                pred_safe = t['predicted'].encode('ascii', errors='replace').decode('ascii')
                print(f'    {t["tier_label"]:<14} -> {pred_safe!r:<55}  {mark}')
        except UnicodeEncodeError:
            # Skip individual records that have un-encodable chars rather than
            # crashing the whole summary dump.
            print('  [skipped record with unencodable chars]')


if __name__ == '__main__':
    main()
