"""Phase 9 held-out-question-type generalization eval.

The North Star metric for Phase 9: can the adapter answer physics
questions of types it NEVER saw during training? Phase 9's training
mix excludes 5 question types (defined in
``physics_llm_adapter/phase9_splits.HELDOUT_QUESTION_TYPES``):

    kinetic_energy        -- numerical, requires velocity AND mass
    collision_prediction  -- boolean, requires future-state reasoning
    speed_comparison      -- relational, picks a specific object
    mass_comparison       -- relational, picks a specific object
    time_to_event         -- numerical, predictive

For each of the 501 held-out CLEVRER scenes we synthesise a question
of each held-out type using ``PhysicsQAGenerator(state_schema='clevrer')``
(the same generator used for training, but invoked with held-out types
that the training data NEVER contained). Ground-truth answers come
from the generator's deterministic answer functions, so scoring
reduces to a substring-match between the model's free-form output and
the categorical / numerical / boolean string the generator emits.

A correct answer here is direct evidence the adapter has learned to
ground physics quantities (energy, momentum, collision likelihood,
speed/mass ranking, temporal prediction) in the prefix tokens --
because the model has never seen the question template, it has no
template to fall back on.

Sibling evals
-------------
- ``free_form_prefix_ablation.py``  -- heldout TRAINING-distribution
   questions, with prefix knockouts. The Phase 7/8 anchor metric.
- ``paraphrased_mcq_test.py``       -- heldout MCQ questions through
   four paraphrase tiers; covers Lever 4 cut #2 (held-out-paraphrase).
- ``eval_phase9_heldout_type.py``   -- THIS file; covers Lever 4 cut #1
   (held-out-question-type). Lever 4 cut #3 (cross-domain) is deferred
   pending Isaac Gym data availability.

Usage
-----
::

    python compsac_2026_code/clevrer_benchmark/scripts/eval_phase9_heldout_type.py \\
        --adapter_checkpoint /path/to/adapter_phase9_diverse_grounding_best.pt \\
        --physics_checkpoint /path/to/physics_former_best.pt \\
        --clevrer_dir $CLEVRER_DIR \\
        --n 200 --gen_mode greedy \\
        --out compsac_2026_code/clevrer_benchmark/results/phase9_diverse_grounding/eval_heldout_type.json
"""
from __future__ import annotations

import os
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Path setup mirrors the sibling eval scripts so any CWD works.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _SCRIPT_DIR.parent
_SNAPSHOT_ROOT = _BENCH_DIR.parent

sys.path.insert(0, str(_SNAPSHOT_ROOT))                          # qa_generator, phase9_splits
sys.path.insert(0, str(_BENCH_DIR))                              # run_adapter_evaluation
sys.path.insert(0, str(_SCRIPT_DIR))                             # free_form_transfer_test helpers
sys.path.insert(0, str(_SNAPSHOT_ROOT / 'data_generation' / 'qa_generation'))

from qa_generator import PhysicsQAGenerator, QuestionType  # type: ignore  # noqa: E402
from physics_llm_adapter.phase9_splits import (  # type: ignore  # noqa: E402
    HELDOUT_QUESTION_TYPES,
    load_heldout_scene_set,
)
# Reuse adapter loader + generation helper + atomic save from the
# established Phase 7/8 eval flow. Keeps Phase 9 evals consistent with
# how Phase 3-8 results were graded.
from run_adapter_evaluation import (  # type: ignore  # noqa: E402
    load_adapter_model,
    _gen_kwargs_for_mode,
)
from free_form_transfer_test import _generate, _atomic_save  # type: ignore  # noqa: E402
# Scene-text injection at inference: ``PhysicsLLMAdapterV3.forward``
# does NOT auto-prepend scene-text (per its docstring); the eval caller
# is expected to do it, matching the training-time prompt distribution.
# Phase 7/8 eval scripts (free_form_transfer_test.py,
# free_form_prefix_ablation.py) follow this convention. Without the
# prepended summary, an inject_scene_text=True checkpoint sees an
# OOD prompt and emits gibberish.
from physics_llm_adapter.scene_summary import (  # type: ignore  # noqa: E402
    build_scene_summary,
)
# Phase 9 output-format fix layers: per-qa_type prompt cues + canned
# 1-shot demos + constrained-decoding LogitsProcessor. All three are
# optional and default off so the eval reproduces the pre-fix Phase 9
# numbers when the flags are absent. See ``scoring/answer_space.py``
# for the docstring rationale.
from clevrer_benchmark.scoring.answer_space import (  # type: ignore  # noqa: E402
    PromptConfig,
    build_prompt,
    make_constrained_processor,
)

DEFAULT_CHECKPOINT = (
    _SNAPSHOT_ROOT / 'checkpoints' / 'adapter_phase9_diverse_grounding_best.pt'
)
DEFAULT_OUTPUT = (
    _BENCH_DIR / 'results' / 'phase9_diverse_grounding' / 'eval_heldout_type.json'
)


# ---------------------------------------------------------------------
# Scoring -- substring match with light normalisation. Exact-match is
# too strict for free-form (the model may say "approximately 5 frames"
# vs. gold "5 frames"); NLI is too loose for short categorical answers
# ("very high" vs. "low" can both entail a vague gold like "moderate").
# Substring match in either direction strikes the right balance for
# the categorical / numerical / boolean answers our generator emits.
# ---------------------------------------------------------------------
def _norm(s: str) -> str:
    return ' '.join(str(s).strip().lower().split())


def _substring_match(pred: str, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return False
    return (p in g) or (g in p)


# ---------------------------------------------------------------------
# Build the eval pool from CLEVRER held-out scenes + held-out question
# types. The generator gives us (question, gold answer) pairs that the
# Phase 9 adapter has never seen during training.
# ---------------------------------------------------------------------
def _build_heldout_type_pool(
    clevrer_dir: Path,
    heldout_scene_set,
    questions_per_scene_per_type: int,
    seed: int,
) -> List[Dict]:
    """For each (scene, held-out type) pair, generate N synthetic QA records.

    Returns a flat list of question dicts, each with the fields the
    inference loop needs:

        scene_path, scene_index, qa_type, question, gold_answer

    The state tensor is loaded later (lazily) so the pool stays small.
    """
    from clevrer_benchmark.scene_converter import (  # noqa: E402
        clevrer_scene_to_state_tensor, load_clevrer_scene,
    )

    # Resolve scene directory same way the other evals do.
    scene_dir = None
    for c in (
        clevrer_dir / 'scenes' / 'validation',
        clevrer_dir / 'scenes' / 'clevrer_scenes',
        clevrer_dir / 'scenes',
    ):
        if c.exists() and any(c.glob('*.json')):
            scene_dir = c
            break
    if scene_dir is None:
        raise FileNotFoundError(f'No CLEVRER scene dir under {clevrer_dir}')

    # Filter scene files to the held-out 501-scene subset.
    scene_paths: List[Path] = []
    for p in sorted(scene_dir.glob('annotation_*.json')):
        try:
            scene_num = int(p.stem.split('_')[-1])
        except ValueError:
            continue
        if scene_num in heldout_scene_set:
            scene_paths.append(p)

    # Materialise QuestionType enum members for the held-out type list.
    heldout_qts = []
    for s in HELDOUT_QUESTION_TYPES:
        try:
            heldout_qts.append(QuestionType(s))
        except ValueError:
            print(f'[WARN] held-out type {s!r} not in QuestionType enum')

    gen = PhysicsQAGenerator(
        question_types=heldout_qts,
        seed=seed,
        state_schema='clevrer',
    )

    pool: List[Dict] = []
    for scene_path in scene_paths:
        scene_num = int(scene_path.stem.split('_')[-1])
        try:
            scene = load_clevrer_scene(str(scene_path))
            states_np, masks_np, _ = clevrer_scene_to_state_tensor(scene)
        except Exception:
            continue

        states_t = torch.from_numpy(states_np[0]).float()       # [N, 35]
        mask_union = (masks_np.max(axis=0) > 0.5).astype('float32')
        mask_t = torch.from_numpy(mask_union).float()           # [N]

        for qt in heldout_qts:
            for _ in range(questions_per_scene_per_type):
                try:
                    qa = gen.generate_qa_pair(
                        states=states_t,
                        mask=mask_t,
                        question_type=qt,
                    )
                except Exception:
                    continue
                pool.append({
                    'scene_path': str(scene_path),
                    'scene_index': scene_num,
                    'qa_type': qt.value,
                    'question': qa.question,
                    'gold_answer': qa.answer,
                })
    return pool


# ---------------------------------------------------------------------
# Inference + scoring loop.
# ---------------------------------------------------------------------
# NOTE: ``_gen_kwargs_for_mode`` is imported from
# ``run_adapter_evaluation.py`` above. Earlier revisions defined a local
# copy that returned ``{do_sample: False, max_new_tokens: 30}``. That was
# silently broken: ``PhysicsLLMAdapterV3.forward`` accepts ``max_length``
# (mapped internally to ``generate(..., max_new_tokens=max_length)``)
# but NOT ``max_new_tokens`` itself, so the kwarg unpacking raised
# ``TypeError: unexpected keyword argument 'max_new_tokens'`` which the
# generation try/except swallowed -> every prediction was the empty
# string -> 0% substring acc across all 5 held-out types. The canonical
# helper at ``run_adapter_evaluation._gen_kwargs_for_mode`` returns
# ``{do_sample: False, num_beams: 1}`` for greedy, which V3.forward
# accepts cleanly (and uses its ctor default ``max_length=50``).
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n', 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--adapter_checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--physics_checkpoint', type=str,
                        default=str(_SNAPSHOT_ROOT / 'checkpoints' / 'physics_former_best.pt'))
    parser.add_argument('--clevrer_dir', type=Path, default=Path(os.environ.get('CLEVRER_DIR', 'clevrer')))
    parser.add_argument('--n', type=int, default=200,
                        help='Cap on number of (scene, type) questions sampled '
                             '(default 200). With 5 types this is ~40 scenes '
                             'per type.')
    parser.add_argument('--questions_per_scene_per_type', type=int, default=1,
                        help='How many synthetic questions to draw per (scene, '
                             'type) pair. Larger = denser eval but linear cost.')
    parser.add_argument('--gen_mode', choices=['greedy', 'sample'],
                        default='greedy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument(
        '--no_scene_text',
        action='store_true',
        help='Diagnostic: force-disable scene-text injection at inference '
             'even if the adapter recipe sets inject_scene_text=True. '
             'Probes whether the prefix tokens carry physics signal on '
             'their own, or whether scene-text was the load-bearing '
             'channel. Pairs with --constrained_decoding to isolate '
             'prefix-vs-text contribution.',
    )
    # ------------------------------------------------------------------
    # Output-format fix layers. Default off => baseline Phase 9 prompt.
    # ------------------------------------------------------------------
    parser.add_argument(
        '--answer_cue',
        action='store_true',
        help='Layer 1 (Fix 1): inject per-qa_type answer-space hint '
             'into the prompt, e.g. "(Choose one: imminent, soon, ...)". '
             'Uses ``scoring.answer_space.ANSWER_SPACE_CUES``. '
             'Zero training cost.',
    )
    parser.add_argument(
        '--in_context_shots',
        type=int,
        default=0,
        choices=[0, 1],
        help='Layer 2 (Fix 2): number of canned 1-shot demonstrations '
             'to prepend per qa_type from '
             '``scoring.answer_space.IN_CONTEXT_DEMOS``. 0 = none, '
             '1 = one demo. Demos are static (not drawn from any eval '
             'scene) so the only information leaked is the answer space.',
    )
    parser.add_argument(
        '--constrained_decoding',
        action='store_true',
        help='Layer 3 (Fix 3): apply a per-qa_type LogitsProcessor that '
             'masks all but the allowed first tokens at the first '
             'generation step. Force-commits the model to a valid '
             'answer-space bucket. Especially useful for ``time_to_event``.',
    )
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build the eval pool.
    # ------------------------------------------------------------------
    print(f'[pool] resolving held-out scene set ...')
    heldout = load_heldout_scene_set()
    print(f'[pool] heldout scenes: {len(heldout):,}')
    print(f'[pool] held-out question types: {sorted(HELDOUT_QUESTION_TYPES)}')
    print(f'[pool] building (scene, type) pool ...')
    full_pool = _build_heldout_type_pool(
        clevrer_dir=args.clevrer_dir,
        heldout_scene_set=heldout,
        questions_per_scene_per_type=args.questions_per_scene_per_type,
        seed=args.seed,
    )
    print(f'[pool] {len(full_pool):,} candidate (scene, type) records')
    if not full_pool:
        raise SystemExit(
            'Empty pool -- check --clevrer_dir and that '
            'heldout_scenes.json is reachable.'
        )

    # Stratified sample: ensure roughly even coverage across types.
    rng = random.Random(args.seed)
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for r in full_pool:
        by_type[r['qa_type']].append(r)
    per_type_n = max(args.n // len(by_type), 1)
    sample: List[Dict] = []
    for qt_str, recs in by_type.items():
        rng.shuffle(recs)
        sample.extend(recs[:per_type_n])
    rng.shuffle(sample)
    sample = sample[:args.n]
    print(f'[pool] {len(sample):,} sampled '
          f'(target ~{per_type_n} per held-out type)')

    # ------------------------------------------------------------------
    # 2. Load adapter.
    # ------------------------------------------------------------------
    if not args.adapter_checkpoint.exists():
        raise FileNotFoundError(
            f'Adapter checkpoint not found: {args.adapter_checkpoint}'
        )
    print(f'[load] adapter from {args.adapter_checkpoint}')
    adapter = load_adapter_model(
        str(args.adapter_checkpoint),
        args.physics_checkpoint,
        device=args.device,
    )
    adapter.eval()
    gen_kwargs = _gen_kwargs_for_mode(args.gen_mode)

    # ------------------------------------------------------------------
    # 3. Inference + scoring loop.
    # ------------------------------------------------------------------
    from clevrer_benchmark.scene_converter import (  # noqa: E402
        clevrer_scene_to_state_tensor, load_clevrer_scene,
    )

    records: List[Dict] = []
    counts = defaultdict(lambda: {'n': 0, 'correct': 0})
    counts_overall = {'n': 0, 'correct': 0}

    t0 = time.time()

    def _summary() -> Dict:
        per_type = {qt: {
            'n': c['n'],
            'substring_acc_pct': round(100.0 * c['correct'] / max(c['n'], 1), 2),
        } for qt, c in counts.items()}
        return {
            'config': {
                'adapter_checkpoint': str(args.adapter_checkpoint),
                'physics_checkpoint': str(args.physics_checkpoint),
                'n_sampled': counts_overall['n'],
                'gen_mode': args.gen_mode,
                'gen_seed': args.seed,
                'heldout_question_types': sorted(HELDOUT_QUESTION_TYPES),
                # Output-format-fix layers active in this run.
                'answer_cue': bool(args.answer_cue),
                'in_context_shots': int(args.in_context_shots),
                'constrained_decoding': bool(args.constrained_decoding),
            },
            'overall': {
                'n': counts_overall['n'],
                'substring_acc_pct': round(
                    100.0 * counts_overall['correct'] / max(counts_overall['n'], 1),
                    2,
                ),
            },
            'per_type': per_type,
        }

    last_flush_n = 0
    for q in tqdm(sample, desc='[heldout-type]', unit='q'):
        try:
            scene = load_clevrer_scene(q['scene_path'])
            states_np, masks_np, _ = clevrer_scene_to_state_tensor(scene)
        except Exception:
            continue
        states_t = torch.from_numpy(states_np[0]).float().unsqueeze(0)  # [1, N, 35]
        masks_t = torch.from_numpy(
            (masks_np.max(axis=0) > 0.5).astype('float32')
        ).float().unsqueeze(0)                                          # [1, N]

        # Phase 9 prompt construction. If the adapter was trained with
        # scene-text injection (Phase 9 default: inject_scene_text=True,
        # style='numbered'), prepend a deterministic scene-summary string
        # to the question so the eval-time prompt matches the training-
        # time distribution. Without this, V3 emits gibberish because the
        # LoRA expects to see ``Object 1: <descriptors>...`` tokens before
        # the question. The adapter object exposes its saved injection
        # config via attributes mirrored from the recipe sidecar at
        # load time, so we read them directly rather than re-passing
        # them via CLI flags.
        if getattr(adapter, 'inject_scene_text', False) and not args.no_scene_text:
            scene_text = build_scene_summary(
                states_np[0],                            # [N, 35] frame 0
                (masks_np.max(axis=0) > 0.5).astype('float32'),  # [N] union mask
                style=getattr(adapter, 'scene_text_style', 'numbered'),
                include_color=getattr(
                    adapter, 'scene_text_include_color', True),
                include_material=getattr(
                    adapter, 'scene_text_include_material', True),
            )
        else:
            scene_text = ''

        # Layered prompt build (cue + shots). With both flags off this
        # collapses byte-for-byte to ``f"{scene_text} {question} Answer:"``
        # so existing baselines reproduce.
        prompt_cfg = PromptConfig(
            answer_cue=bool(args.answer_cue),
            in_context_shots=int(args.in_context_shots),
        )
        prompt = build_prompt(
            qa_type=q['qa_type'],
            scene_text=scene_text,
            question=q['question'],
            cfg=prompt_cfg,
        )

        # Layer 3: constrained decoding. Build a fresh per-call
        # LogitsProcessor (it is stateful: tracks first-call across
        # generation steps). Skip silently if the qa_type has no
        # registered answer-space (returns None).
        per_call_kwargs = dict(gen_kwargs)
        if args.constrained_decoding:
            proc = make_constrained_processor(q['qa_type'], adapter.tokenizer)
            if proc is not None:
                per_call_kwargs['logits_processor'] = proc

        try:
            pred = _generate(
                adapter, states_t, masks_t, prompt,
                per_call_kwargs, args.device,
            )
        except Exception as e:
            # Surface the error to stderr so future generation bugs
            # don't silently produce empty-string predictions across
            # the whole run (this is exactly the failure mode that hid
            # the ``max_new_tokens`` kwarg bug for one full eval pass).
            # We still continue to the next sample so a single-scene
            # CUDA OOM doesn't kill a 200-question run.
            sys.stderr.write(
                f'[gen-error] scene={q.get("scene_index")!r} '
                f'qa_type={q.get("qa_type")!r}: {type(e).__name__}: {e}\n'
            )
            pred = ''

        ok = _substring_match(pred, q['gold_answer'])

        qt = q['qa_type']
        counts[qt]['n'] += 1
        counts[qt]['correct'] += int(ok)
        counts_overall['n'] += 1
        counts_overall['correct'] += int(ok)

        records.append({
            'scene_index': q['scene_index'],
            'qa_type': qt,
            'question': q['question'],
            'gold_answer': q['gold_answer'],
            'predicted': pred,
            'substring_correct': bool(ok),
        })

        if args.save_every > 0 and len(records) - last_flush_n >= args.save_every:
            _atomic_save(args.out, _summary(), records)
            last_flush_n = len(records)

    # Final flush.
    _atomic_save(args.out, _summary(), records)
    dt = time.time() - t0

    # ------------------------------------------------------------------
    # 4. Banner.
    # ------------------------------------------------------------------
    s = _summary()
    print()
    print('=' * 72)
    print(f'Phase 9 held-out-type eval complete in {dt/60:.1f} min')
    print('=' * 72)
    print(f'  ckpt:           {args.adapter_checkpoint.name}')
    print(f'  n sampled:      {s["overall"]["n"]:,}')
    print(f'  overall acc:    {s["overall"]["substring_acc_pct"]:.2f}% '
          f'(substring match)')
    print()
    print('  per-type breakdown:')
    print(f'    {"type":<26s} {"n":>4s} {"acc%":>7s}')
    for qt in sorted(HELDOUT_QUESTION_TYPES):
        c = s['per_type'].get(qt, {'n': 0, 'substring_acc_pct': 0.0})
        print(f'    {qt:<26s} {c["n"]:>4d} {c["substring_acc_pct"]:>6.2f}%')
    print()
    print(f'  records written to: {args.out}')


if __name__ == '__main__':
    main()
