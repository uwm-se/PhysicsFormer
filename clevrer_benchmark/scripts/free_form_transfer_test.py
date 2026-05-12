"""Free-form transfer test: probe whether the Phase 3 adapter can answer
CLEVRER physics questions WITHOUT being shown the MCQ choices.

The paraphrase audit (paraphrase_audit.py) showed the adapter emits a verbatim
copy of one of the labeled choices on 99%+ of held-out questions -- meaning at
inference time the model behaves as an MCQ classifier with physics priors
rather than as an open-ended physics reasoner. This script tests the natural
follow-up question: when we strip the ``Options: ...`` suffix from the prompt,
does the adapter still produce coherent CLEVRER-style physics descriptions, or
does its competence collapse?

Procedure
---------
1. Load the canonical Phase 3 adapter (same loader as run_adapter_evaluation.py).
2. Sample ``--n`` questions from the held-out valid-only pool (default 50).
3. For each, build the standard single-frame state tensor from frame 64.
4. Generate twice:
   - ``mcq``       : prompt = ``"<question> Options: <c1, c2, c3, ...>"``  (control)
   - ``free_form`` : prompt = ``"<question>"``                             (test)
5. Score each generation against the labeled correct choices using the same
   substring rule as ``evaluate_answer()`` and an NLI bidirectional-entailment
   rule (DeBERTa-v3-base-MNLI, threshold 0.7 in both directions).
6. Report bucket counts, accuracy under both rules, sample outputs, and the
   gap ``mcq_acc - free_form_acc`` -- the size of that gap quantifies how
   much of the adapter's CLEVRER score depends on the choice menu being
   visible at inference time.

Usage
-----
::

    python compsac_2026_code/clevrer_benchmark/scripts/free_form_transfer_test.py \\
        --n 50 --heldout

The script auto-resolves the adapter checkpoint, the held-out scene set, and
result paths from this script's location, so it works from any CWD inside the
snapshot.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Snapshot-portable defaults.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _SCRIPT_DIR.parent
_SNAPSHOT_ROOT = _BENCH_DIR.parent

DEFAULT_CHECKPOINT = _SNAPSHOT_ROOT / 'checkpoints' / 'adapter_phase3.pt'
DEFAULT_RESULTS_DIR = _BENCH_DIR / 'results'
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / 'free_form_transfer_test.json'
DEFAULT_H5 = os.environ.get(
    'CLEVRER_H5',
    os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5'),
)
DEFAULT_NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

# Make the eval-script and stats-script imports resolvable.
sys.path.insert(0, str(_BENCH_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

# Make ``physics_llm_adapter`` (the snapshot-internal package containing
# scene_summary.py) importable. The eval scripts already live inside the
# compsac_2026_code snapshot, but they sit at depth 2 (clevrer_benchmark/
# scripts/...) so adding the snapshot root to sys.path is the cleanest
# way to ``from physics_llm_adapter import scene_summary`` without
# refactoring the existing import paths used elsewhere in this file.
sys.path.insert(0, str(_SNAPSHOT_ROOT))
from physics_llm_adapter.scene_summary import build_scene_summary  # noqa: E402

# Surface-form scoring helpers were extracted into the shared scoring package
# so this script, paraphrased_mcq_test.py, free_form_prefix_ablation.py, and
# the Phase 9 lenient rescore all share one definition of ``substring_correct``
# / ``verbatim_choice_match`` / ``looks_clevrer_template``. Behaviour is byte
# identical -- the names below are the original underscore-prefixed helpers,
# kept as aliases so ``free_form_prefix_ablation.py`` (which imports them
# from this module) continues to work unmodified.
from scoring.text_match import (  # noqa: E402
    _norm,
    _substring_correct,
    _verbatim_choice_match,
    _looks_clevrer_template,
)


def _load_heldout_scenes(h5_path: Path) -> set:
    """Resolve the canonical 501-scene held-out subset used in the paper.

    Resolution order (first hit wins):

    1. ``{repo}/clevrer_benchmark/results/heldout_scenes.json`` -- the
       authoritative sidecar produced once by ``compute_paper_stats.py``
       and committed to the repo. This is what Phase 3-7 evals actually
       read against (verified: 100% scene-id overlap with Phase 7
       prefix-ablation/transfer-test result records). Free-standing,
       no h5py dependency, no risk of H5-schema drift.

    2. ``compute_paper_stats.load_heldout_scenes(h5_path)`` -- recompute
       from the training H5. Only works if ``h5_path`` happens to use
       native CLEVRER question_types (``counterfactual``/``explanatory``
       /``predictive``); fails silently with an empty set if the H5 was
       built with the synthetic-style labels (``causal_chain``/
       ``future_prediction``/``counterfactual_reasoning``) that the
       physics-LLM training pipeline emits. We keep this as a fallback
       so a fresh checkout without the JSON sidecar still has a path,
       but loudly warn so the operator knows something is off.
    """
    sidecar = _BENCH_DIR / 'results' / 'heldout_scenes.json'
    if sidecar.exists():
        with open(sidecar, 'r') as f:
            ids = json.load(f)
        return set(ids)

    from compute_paper_stats import load_heldout_scenes  # type: ignore
    heldout = load_heldout_scenes(h5_path)
    if not heldout:
        print(f'[WARN] _load_heldout_scenes: H5 fallback returned 0 scenes. '
              f'The H5 at {h5_path} likely has synthetic-style '
              f"question_types (e.g. 'causal_chain') instead of the "
              f"native CLEVRER labels ('counterfactual'/'explanatory'/"
              f"'predictive') that compute_paper_stats expects. The "
              f'canonical heldout_scenes.json sidecar is also absent at '
              f'{sidecar}. The eval pool will be empty.', flush=True)
    return heldout


# Scene-id parsing also lives in scoring/referent_equiv now. The two callsites
# below import the public ``scene_id_to_num`` and re-bind it to the original
# private name so the rest of this script doesn't change.
from scoring.referent_equiv import scene_id_to_num as _scene_id_to_num  # noqa: E402


def _build_question_pool(
    clevrer_dir: Path,
    heldout: Optional[set],
    valid_only: bool,
) -> List[Dict]:
    """Mirror ``run_evaluation()``'s scene/question discovery so the sampled
    pool is exactly the evaluation pool. We collect (scene_path, question_data)
    triples so the caller can sample without redoing scene IO.
    """
    # Replicate the scene/questions path resolution from run_evaluation().
    scene_dir = None
    candidates = [
        clevrer_dir / 'scenes' / 'validation',
        clevrer_dir / 'scenes' / 'clevrer_scenes',
        clevrer_dir / 'scenes',
    ]
    for c in candidates:
        if c.exists() and any(c.glob('*.json')):
            scene_dir = c
            break
    if scene_dir is None:
        raise FileNotFoundError(f'No CLEVRER scene dir under {clevrer_dir}')

    questions_file = clevrer_dir / 'questions' / 'clevrer_validation.json'
    if not questions_file.exists():
        questions_file = clevrer_dir / 'questions' / 'validation.json'
    if not questions_file.exists():
        raise FileNotFoundError(f'No CLEVRER validation questions at {clevrer_dir/"questions"}')

    with open(questions_file, 'r') as f:
        all_questions = json.load(f)

    from collections import defaultdict
    questions_by_scene = defaultdict(list)
    for sd in all_questions:
        sid = sd.get('scene_index', sd.get('video_index', 0))
        questions_by_scene[sid] = sd.get('questions', [])

    causal_types = {'explanatory', 'predictive', 'counterfactual'}
    pool: List[Dict] = []
    for scene_path in sorted(scene_dir.glob('*.json')):
        scene_id = scene_path.stem
        scene_num = _scene_id_to_num(scene_id)
        if scene_num is None:
            continue
        if heldout is not None and scene_num not in heldout:
            continue
        for q_idx, q in enumerate(questions_by_scene.get(scene_num, [])):
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
                'scene_num': scene_num,
                'q_idx': q_idx,
                'question_text': q.get('question', ''),
                'question_type': qtype,
                'choices': choices,
            })
    return pool


def _generate(adapter, states, masks, prompt: str, gen_kwargs: Dict, device,
              adapter_type: str = 'v2') -> str:
    """Run a single forward generation against the chosen adapter API.

    ``v2``  -> ``adapter.forward(states, masks, [prompt], **gen_kwargs)`` returns
                a list of strings (PhysicsLLMAdapterV2 / V3 family).
    ``grounded`` -> ``adapter.generate(states, masks, prompt=p, max_new_tokens=,
                temperature=)`` returns a single string
                (grounded_physics_adapter.GroundedPhysicsAdapter).
    """
    import torch
    if states.dim() == 3:
        states = states.unsqueeze(0)
    if masks.dim() == 2:
        masks = masks.unsqueeze(0)
    states = states.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        if adapter_type == 'grounded':
            out = adapter.generate(
                physics_states=states,
                object_mask=masks,
                prompt=prompt,
                max_new_tokens=gen_kwargs.get('max_new_tokens', 30),
                temperature=gen_kwargs.get('temperature', 0.7),
            )
        else:
            out = adapter.forward(states, masks, [prompt], **gen_kwargs)
    if isinstance(out, list):
        return str(out[0])
    return str(out)


# NLI helpers (online per-prediction grading) live in scoring/nli_paraphrase
# now. Same byte-for-byte semantics; the originals expected (model_id, device)
# positionally and so does the imported version.
from scoring.nli_paraphrase import (  # noqa: E402
    _nli_setup,
    _nli_paraphrase,
    _nli_correct,
)


def _atomic_save(path: Path, summary: dict, records: list) -> None:
    """Atomically write {summary, records} so a kill mid-flush leaves a valid file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'records': records}, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _summary_from_counts(counts: dict, args) -> dict:
    """Build the summary dict from current counts. Safe to call mid-run for a
    partial summary -- ``n_sampled`` reflects only completed questions."""
    n = max(counts['n_total'], 1)
    pct = lambda k: 100.0 * counts[k] / n  # noqa: E731
    return {
        'config': {
            'n_sampled': counts['n_total'],
            'heldout_only': args.heldout,
            'valid_only': args.valid_only,
            'single_frame': args.single_frame,
            'gen_mode': args.gen_mode,
            'gen_seed': args.gen_seed,
            'nli_model': args.nli_model if args.use_nli else None,
            'nli_threshold': args.nli_threshold if args.use_nli else None,
            # Phase A scene-text injection config. Captured here so the
            # JSON output is self-describing -- a downstream A/B
            # aggregator can tell from the file alone whether injection
            # was on, in which style, and which attributes were dropped.
            'inject_scene_text':       getattr(args, 'inject_scene_text',      False),
            'scene_text_style':        getattr(args, 'scene_text_style',       'comma_list'),
            'scene_text_no_color':     getattr(args, 'scene_text_no_color',    False),
            'scene_text_no_material':  getattr(args, 'scene_text_no_material', False),
        },
        'mcq': {
            'substring_acc_pct': round(pct('mcq_substring_correct'), 2),
            'nli_acc_pct': round(pct('mcq_nli_correct'), 2),
            'verbatim_correct_pct': round(pct('mcq_verbatim_correct'), 2),
            'verbatim_wrong_pct': round(pct('mcq_verbatim_wrong'), 2),
            'free_form_text_pct': round(pct('mcq_free_form_text'), 2),
        },
        'free_form': {
            'substring_acc_pct': round(pct('free_form_substring_correct'), 2),
            'nli_acc_pct': round(pct('free_form_nli_correct'), 2),
            'clevrer_template_phrasing_pct': round(pct('ff_template_phrasing'), 2),
            'clevrer_choice_membership_pct': round(pct('ff_clevrer_choice_membership'), 2),
        },
        'gap': {
            'substring_pp': round(pct('mcq_substring_correct') - pct('free_form_substring_correct'), 2),
            'nli_pp': round(pct('mcq_nli_correct') - pct('free_form_nli_correct'), 2),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Free-form vs MCQ transfer test for the Phase 3 adapter.')
    parser.add_argument('--n', type=int, default=50,
                        help='Number of heldout questions to sample (default 50).')
    parser.add_argument('--save_every', type=int, default=25,
                        help='Periodic incremental save cadence in questions (default 25). '
                             'Lower = safer (less work lost on kill) but more disk churn.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--heldout', action='store_true',
                        help='Sample only from the 501 heldout scenes (paper primary).')
    parser.add_argument('--valid_only', action='store_true', default=True,
                        help='Drop zero-correct MCQ trap items (default on).')
    parser.add_argument('--clevrer_dir', type=Path, default=Path(os.environ.get('CLEVRER_DIR', 'clevrer')))
    parser.add_argument('--adapter_checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        '--adapter_type', choices=['v2', 'grounded'], default='v2',
        help="'v2': PhysicsLLMAdapterV2/V3 (compsac Phase 3/4). "
             "'grounded': GroundedPhysicsAdapter from grounded_physics_adapter/."
    )
    parser.add_argument(
        '--causal_encoder_path', type=Path,
        default=Path('physics_llm_adapter/causal_encoder_best.pt'),
        help='Frozen CausalEncoder checkpoint (only used when adapter_type=grounded).'
    )
    parser.add_argument(
        '--physics_tokenizer_path', type=Path,
        default=Path('physics_tokenizer/physics_tokenizer_finetuned.pt'),
        help='PhysicsConceptTokenizer + LanguageAligner checkpoint (only used when adapter_type=grounded).'
    )
    parser.add_argument('--physics_checkpoint', type=str,
                        default='D:\\physics-former-data\\checkpoints\\stage1_best.pt')
    parser.add_argument('--h5', type=Path, default=Path(DEFAULT_H5),
                        help='CLEVRER training H5; only used with --heldout.')
    parser.add_argument('--single_frame', type=int, default=64,
                        help='Frame index (canonical 64). Pass -1 for full trajectory.')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--gen_mode', type=str, default='sample',
                        choices=['sample', 'greedy', 'beam4', 'beam8'])
    parser.add_argument('--gen_seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT,
                        help='Output JSON with full per-question records.')
    parser.add_argument('--use_nli', action='store_true', default=True,
                        help='Run NLI semantic-match scoring (default on).')
    parser.add_argument('--no_nli', action='store_false', dest='use_nli')
    parser.add_argument('--nli_model', type=str, default=DEFAULT_NLI_MODEL)
    parser.add_argument('--nli_threshold', type=float, default=0.7)
    # ------------------------------------------------------------------
    # Phase A: scene-text injection (eval-time only).
    # The encoder OOD probe (encoder_ood_probing_v2_surface.json) showed
    # that the frozen PhysicsFormer encoder destroys per-object color
    # (R^2 1.00 input -> -0.004 encoder_per_obj on CLEVRER) and that the
    # adapter prefix loses scene-presence color signal entirely (AUC 0.81
    # input_sum -> 0.52 prefix_pool ~ random). When --inject_scene_text
    # is set, this script prepends a deterministic textual summary of
    # surface attributes to every prompt, e.g.
    #   "Scene contains: red metal cube, blue rubber sphere. <question>"
    # built from the per-object state vector via scene_summary.py. This
    # bypasses the encoder for color/shape/material, letting us A/B
    # whether text injection alone (no retrain) lifts free-form NLI on
    # the existing checkpoint -- if yes, the retrain in Phase B is gravy.
    # ------------------------------------------------------------------
    parser.add_argument('--inject_scene_text', action='store_true',
                        help='Prepend a deterministic scene-summary string '
                             '("Scene contains: red metal cube, ...") to '
                             'both the MCQ and free-form prompts. Useful as '
                             'an eval-time A/B against the same adapter.')
    parser.add_argument(
        '--no_scene_text',
        action='store_true',
        help='Diagnostic: force-disable scene-text injection even if the '
             'adapter recipe sets inject_scene_text=True. Overrides both '
             '--inject_scene_text and the auto-detected adapter attribute.',
    )
    parser.add_argument('--scene_text_style', type=str, default='comma_list',
                        choices=['comma_list', 'numbered'],
                        help="Format of the injected scene text. "
                             "'comma_list' = 'Scene contains: red cube, blue sphere.', "
                             "'numbered'   = 'Object 1: red cube. Object 2: blue sphere.'")
    parser.add_argument('--scene_text_no_color', action='store_true',
                        help='Drop the color word from the injected text. '
                             'Useful for ablating which surface attribute '
                             'the LLM actually relies on at inference.')
    parser.add_argument('--scene_text_no_material', action='store_true',
                        help='Drop the material (rubber/metal) word from the '
                             'injected text.')
    args = parser.parse_args()

    if args.device == 'auto':
        try:
            import torch
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            args.device = 'cpu'

    if not args.adapter_checkpoint.exists():
        raise FileNotFoundError(
            f'Adapter checkpoint not found: {args.adapter_checkpoint}')

    # --- Pool & sample ---
    heldout = _load_heldout_scenes(args.h5) if args.heldout else None
    pool = _build_question_pool(args.clevrer_dir, heldout, valid_only=args.valid_only)
    print(f'[pool] {len(pool):,} heldout valid-only causal questions available')
    if not pool:
        raise SystemExit('Empty pool -- check --clevrer_dir and --heldout / --h5.')

    rng = random.Random(args.seed)
    sample = rng.sample(pool, k=min(args.n, len(pool)))
    print(f'[pool] sampled {len(sample)} for the transfer test')

    # --- Load adapter ---
    print(f'[adapter] loading (type={args.adapter_type}) ...', flush=True)
    from run_adapter_evaluation import (
        load_adapter_model,
        _gen_kwargs_for_mode,
    )
    from scene_converter import load_clevrer_scene, clevrer_scene_to_state_tensor
    import torch
    if args.adapter_type == 'grounded':
        # Lazy-import the workspace-root grounded_physics_adapter package.
        # It does its own sys.path manipulation for physics_tokenizer/.
        import sys as _sys
        _proj_root = Path(__file__).resolve().parents[3]
        for _p in (_proj_root, _proj_root / 'physics_tokenizer'):
            if str(_p) not in _sys.path:
                _sys.path.insert(0, str(_p))
        from grounded_physics_adapter.adapter import load_grounded_adapter
        adapter = load_grounded_adapter(
            causal_encoder_path=str(args.causal_encoder_path),
            tokenizer_path=str(args.physics_tokenizer_path),
            adapter_checkpoint=str(args.adapter_checkpoint),
            device=args.device,
        )
        adapter.eval()
        # Grounded API ignores HF generation kwargs; use its own knobs.
        gen_kwargs = {'max_new_tokens': 30, 'temperature': 0.7}
    else:
        adapter = load_adapter_model(str(args.adapter_checkpoint),
                                     args.physics_checkpoint, device=args.device)
        gen_kwargs = _gen_kwargs_for_mode(args.gen_mode)
    torch.manual_seed(args.gen_seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.gen_seed)

    # --- NLI (optional) ---
    nli_state = None
    if args.use_nli:
        nli_state = _nli_setup(args.nli_model, args.device)

    # --- Run both prompt formats per question ---
    records = []
    counts = {
        'n_total': 0,
        'mcq_substring_correct': 0,
        'free_form_substring_correct': 0,
        'mcq_nli_correct': 0,
        'free_form_nli_correct': 0,
        'mcq_verbatim_correct': 0,
        'mcq_verbatim_wrong': 0,
        'mcq_free_form_text': 0,
        'ff_template_phrasing': 0,
        'ff_clevrer_choice_membership': 0,
    }
    print('[run] generating ...', flush=True)
    print(f'[stream] periodic atomic save every {args.save_every} questions -> {args.out}',
          flush=True)
    from tqdm.auto import tqdm
    for _i, q in enumerate(tqdm(sample, desc='[ff_test]', unit='q')):
        try:
            scene = load_clevrer_scene(str(q['scene_path']))
            states, masks, _ = clevrer_scene_to_state_tensor(scene)
        except Exception as e:
            print(f'  [skip] {q["scene_id"]}: {e}')
            continue

        # Capture full-trajectory states + mask BEFORE single-frame slicing.
        # The scene-text summary needs every object that was EVER in the
        # scene (CLEVRER question prompts routinely name objects that have
        # already exited the camera view by frame 64, e.g. "If the cylinder
        # is removed...") whereas the adapter consumes the single-frame
        # slice. Using mask.max(axis=0) gives the union of per-frame
        # visibility, i.e. an object is "in the scene" if it was ever
        # visible. Static attributes (color, shape, mass / material) are
        # constant across time per scene_converter, so frame 0 is a safe
        # source for the description.
        states_full = states
        if masks.ndim == 2:
            scene_text_mask = masks.max(axis=0)                    # [N]
            scene_text_states = states_full[0]                     # [N, 35]
        else:
            scene_text_mask = masks                                # already [N]
            scene_text_states = states_full                        # [N, 35]

        if args.single_frame is not None and args.single_frame >= 0:
            fi = min(args.single_frame, states.shape[0] - 1)
            states = states[fi:fi + 1]
            masks = masks[fi:fi + 1] if masks.ndim == 2 else masks
        states_t = torch.from_numpy(states).float()
        masks_t = torch.from_numpy(masks).float()

        choices = q['choices']
        choice_texts = [c.get('choice', c) if isinstance(c, dict) else str(c)
                        for c in choices]
        correct_texts = [c['choice'] for c in choices if c.get('answer') == 'correct']
        wrong_texts = [c['choice'] for c in choices if c.get('answer') == 'wrong']

        # Shuffle choices to match the primary protocol.
        order = list(range(len(choice_texts)))
        rng.shuffle(order)
        choice_texts_shuf = [choice_texts[i] for i in order]

        # Build base prompts. mcq_prompt has the choice list, ff_prompt
        # does not -- we generate against both formats per question.
        mcq_prompt = q['question_text'] + ' Options: ' + ', '.join(choice_texts_shuf)
        ff_prompt = q['question_text']

        # Scene-text injection. Triggered by EITHER:
        #   (a) the --inject_scene_text CLI flag (Phase A: eval-time-only
        #       A/B against an existing checkpoint), OR
        #   (b) the adapter being a Phase B/C checkpoint with
        #       adapter.inject_scene_text=True (the adapter itself reports
        #       how it was trained, so the eval caller auto-matches the
        #       training distribution without requiring the operator to
        #       pass --inject_scene_text explicitly). The auto-detected
        #       path uses the adapter's saved style; the CLI path uses
        #       the user-supplied style (since the goal there is A/B).
        # We build the summary from the FULL-TRAJECTORY states + the
        # union-across-time mask rather than the single-frame slice,
        # because CLEVRER questions often reference objects that have
        # already exited the camera view by frame 64 (e.g. "If the
        # cylinder is removed..." when the cylinder bounced out at
        # frame 40). Using the union mask makes the summary describe the
        # scene's COMPOSITION, not its frame-64 visibility.
        auto_inject = getattr(adapter, 'inject_scene_text', False)
        if (args.inject_scene_text or auto_inject) and not args.no_scene_text:
            if args.inject_scene_text:
                style     = args.scene_text_style
                no_color  = args.scene_text_no_color
                no_mat    = args.scene_text_no_material
            else:
                style     = getattr(adapter, 'scene_text_style', 'comma_list')
                # Auto-detected path uses the adapter's saved include
                # flags verbatim -- anything else would be an OOD prompt
                # vs. what training saw.
                no_color  = not getattr(adapter, 'scene_text_include_color', True)
                no_mat    = not getattr(adapter, 'scene_text_include_material', True)
            scene_text = build_scene_summary(
                scene_text_states, scene_text_mask,
                style=style,
                include_color=not no_color,
                include_material=not no_mat,
            )
            mcq_prompt = scene_text + ' ' + mcq_prompt
            ff_prompt = scene_text + ' ' + ff_prompt

        mcq_pred = _generate(adapter, states_t, masks_t, mcq_prompt, gen_kwargs,
                             args.device, adapter_type=args.adapter_type)
        ff_pred = _generate(adapter, states_t, masks_t, ff_prompt, gen_kwargs,
                            args.device, adapter_type=args.adapter_type)

        mcq_bucket = _verbatim_choice_match(mcq_pred, correct_texts, wrong_texts)
        ff_bucket = _verbatim_choice_match(ff_pred, correct_texts, wrong_texts)

        mcq_subs_correct = _substring_correct(mcq_pred, correct_texts, wrong_texts)
        ff_subs_correct = _substring_correct(ff_pred, correct_texts, wrong_texts)

        mcq_nli_correct = ff_nli_correct = False
        if nli_state is not None:
            tok, mod, ent = nli_state
            mcq_nli_correct = _nli_correct(tok, mod, ent, mcq_pred, correct_texts,
                                           args.device, args.nli_threshold)
            ff_nli_correct = _nli_correct(tok, mod, ent, ff_pred, correct_texts,
                                          args.device, args.nli_threshold)

        counts['n_total'] += 1
        counts['mcq_substring_correct'] += int(mcq_subs_correct)
        counts['free_form_substring_correct'] += int(ff_subs_correct)
        counts['mcq_nli_correct'] += int(mcq_nli_correct)
        counts['free_form_nli_correct'] += int(ff_nli_correct)
        counts['mcq_verbatim_correct'] += int(mcq_bucket == 'verbatim_correct')
        counts['mcq_verbatim_wrong'] += int(mcq_bucket == 'verbatim_wrong')
        counts['mcq_free_form_text'] += int(mcq_bucket == 'free_form')
        counts['ff_template_phrasing'] += int(_looks_clevrer_template(ff_pred))
        counts['ff_clevrer_choice_membership'] += int(ff_bucket != 'free_form')

        records.append({
            'scene_id': q['scene_id'],
            'q_idx': q['q_idx'],
            'question_type': q['question_type'],
            'question_text': q['question_text'],
            'correct_choices': correct_texts,
            'wrong_choices': wrong_texts,
            'mcq': {
                'prompt': mcq_prompt,
                'predicted': mcq_pred,
                'substring_correct': mcq_subs_correct,
                'nli_correct': mcq_nli_correct,
                'bucket': mcq_bucket,
            },
            'free_form': {
                'prompt': ff_prompt,
                'predicted': ff_pred,
                'substring_correct': ff_subs_correct,
                'nli_correct': ff_nli_correct,
                'bucket': ff_bucket,
                'looks_clevrer_template': _looks_clevrer_template(ff_pred),
            },
        })

        # Periodic incremental save: a kill mid-run leaves the file at the
        # last cohort boundary, never half-written.
        if (_i + 1) % args.save_every == 0:
            _atomic_save(args.out, _summary_from_counts(counts, args), records)

    summary = _summary_from_counts(counts, args)

    # Pretty-print summary.
    print()
    print('=== Free-form transfer test summary ===')
    print(f"  n sampled:                         {summary['config']['n_sampled']}")
    print(f"  heldout / valid-only:              {summary['config']['heldout_only']} / {summary['config']['valid_only']}")
    print()
    print(f"  MCQ prompt (control):")
    print(f"    substring accuracy:              {summary['mcq']['substring_acc_pct']:5.1f}%")
    if args.use_nli:
        print(f"    NLI semantic accuracy:           {summary['mcq']['nli_acc_pct']:5.1f}%")
    print(f"    verbatim-correct picks:          {summary['mcq']['verbatim_correct_pct']:5.1f}%")
    print(f"    verbatim-wrong picks:            {summary['mcq']['verbatim_wrong_pct']:5.1f}%")
    print(f"    free-form text:                  {summary['mcq']['free_form_text_pct']:5.1f}%")
    print()
    print(f"  Free-form prompt (test):")
    print(f"    substring accuracy:              {summary['free_form']['substring_acc_pct']:5.1f}%")
    if args.use_nli:
        print(f"    NLI semantic accuracy:           {summary['free_form']['nli_acc_pct']:5.1f}%")
    print(f"    output uses CLEVRER templates:   {summary['free_form']['clevrer_template_phrasing_pct']:5.1f}%")
    print(f"    output is verbatim CLEVRER chc:  {summary['free_form']['clevrer_choice_membership_pct']:5.1f}%")
    print()
    print(f"  Gap (MCQ - free_form):")
    print(f"    substring:                       {summary['gap']['substring_pp']:+.1f} pp")
    if args.use_nli:
        print(f"    NLI:                             {summary['gap']['nli_pp']:+.1f} pp")
    print()
    if summary['gap']['substring_pp'] > 30:
        print('  Interpretation: large gap -- the adapter is heavily MCQ-format-bound.')
    elif summary['gap']['substring_pp'] > 10:
        print('  Interpretation: moderate gap -- physics priors transfer partially without choices.')
    else:
        print('  Interpretation: small gap -- the adapter generalizes well to free-form prompts.')

    _atomic_save(args.out, summary, records)
    print(f"\n  records written to: {args.out}")


if __name__ == '__main__':
    main()
