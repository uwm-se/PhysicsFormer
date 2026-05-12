"""Lenient rescoring of Phase 9 eval outputs.

The Phase 9 eval scripts (``eval_phase9_heldout_type``, ``paraphrased_mcq_test``,
``free_form_prefix_ablation``, ``free_form_transfer_test``) grade with a
strict-substring rule that mirrors the Phase 3-7 paper baseline. That rule
is deliberately conservative for apples-to-apples comparison with the LLM
baselines, but it under-credits Phase 9's free-form generations whenever
the model emits a paraphrase, a synonym, or a referentially-equivalent
description rather than parroting the canonical CLEVRER template.

This driver applies the audit-grade lenient rubrics that are already
validated on Phase 3 (and now sit in ``clevrer_benchmark/scoring/``) to
every Phase 9 eval output. It emits ONE consolidated summary JSON with:

  - **Strict substring** (re-confirmed from saved predictions; should
    match the published per-eval header numbers byte-for-byte).
  - **Scene-aware referent-equivalence reclaim** for the three CLEVRER
    MCQ-style evals (paraphrase tiers, prefix ablation, free-form
    transfer). Loads the actual CLEVRER scene-object inventory and
    reclaims wrong predictions iff the prediction's parsed event +
    descriptors uniquely identify the same entities as a labeled
    correct choice and not a labeled wrong choice.
  - **Bidirectional-NLI paraphrase reclaim** at a five-step threshold
    sweep ``[0.4, 0.5, 0.6, 0.7, 0.8]`` so the paper figure can pick
    the threshold that's most defensible against the strict floor and
    show how sensitive the lenient number is to the cutoff.
  - **Categorical synonym + ordinal-adjacency match** for the held-out
    QUESTION TYPES (``kinetic_energy`` / ``collision_prediction`` /
    ``mass_comparison`` / ``speed_comparison`` / ``time_to_event``).
    The substring rule alone misses ``"medium"`` <-> ``"moderate"``,
    ``"the third object"`` <-> ``"3"``, etc. The ordinal-tol=1 row
    additionally credits one-bucket-off predictions on ordinal scales
    (e.g. gold=``moderate`` pred=``high`` is a near miss, not a wrong).
  - **Per-condition roll-ups** so the multi-condition evals (paraphrase
    has 4 tiers, prefix-ablation has 3 conditions, free-form transfer
    has 2 prompt formats) split lenient accuracy along the axis the
    eval was designed to measure.

This script is built entirely on the shared ``clevrer_benchmark.scoring``
package -- there is no scoring logic inline. Compare its 200 lines to
the >1000 lines of duplicated logic that previously lived across the
five eval / audit scripts.

Usage::

    python compsac_2026_code/clevrer_benchmark/scripts/eval_phase9_lenient_rescore.py \\
        [--input_dir compsac_2026_code/clevrer_benchmark/results/phase9_diverse_grounding] \\
        [--clevrer_dir $CLEVRER_DIR] \\
        [--out compsac_2026_code/clevrer_benchmark/results/phase9_lenient_rescore.json] \\
        [--nli_thresholds 0.4 0.5 0.6 0.7 0.8] \\
        [--no_nli]   # skip the NLI sweep (substring + referent + categorical only)

The NLI sweep is the expensive step (~3-5 min per eval on a 5080 mobile
batch_size=32). Pass ``--no_nli`` for a quick directional read using the
free rubrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Snapshot-portable defaults: anchor on this file so ``cd``-anywhere works.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _SCRIPT_DIR.parent
_SNAPSHOT_ROOT = _BENCH_DIR.parent
sys.path.insert(0, str(_BENCH_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SNAPSHOT_ROOT))

from scoring import (  # noqa: E402
    norm,
    substring_correct,
    is_semantic_match,
    parse_event,
    load_scene_objects,
    scene_id_to_num,
    build_pairs,
    run_nli_batched,
    evaluate_flips,
    categorical_correct,
    HELDOUT_TYPE_BUCKETS,
)
from scoring.io import (  # noqa: E402
    detect_phase9_eval_kind,
    iter_legacy_records_paraphrase,
    iter_legacy_records_prefix_ablation,
    iter_legacy_records_free_form_transfer,
)


DEFAULT_INPUT_DIR = _BENCH_DIR / 'results' / 'phase9_diverse_grounding'
DEFAULT_OUTPUT = _BENCH_DIR / 'results' / 'phase9_lenient_rescore.json'
DEFAULT_CLEVRER = Path(os.environ.get('CLEVRER_DIR', 'clevrer'))
DEFAULT_NLI_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
DEFAULT_NLI_MARGIN = 0.05
DEFAULT_NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

PHASE9_FLATTENERS = {
    'paraphrase':         iter_legacy_records_paraphrase,
    'prefix_ablation':    iter_legacy_records_prefix_ablation,
    'free_form_transfer': iter_legacy_records_free_form_transfer,
}


# ---------------------------------------------------------------------------
# Per-rubric scorers
# ---------------------------------------------------------------------------

def _strict_count(records: List[Dict]) -> int:
    """Number of records whose ``correct`` field (set by the eval script
    at generation time) is True. Mirrors the published header number
    so a delta of 0 confirms our converter is faithful."""
    return sum(1 for r in records if r.get('correct'))


def _referent_reclaim(records: List[Dict],
                      clevrer_dir: Path) -> Dict:
    """Scene-aware referent-equivalence reclaim across the wrong records.

    For every record where ``correct`` is False, parse the prediction's
    event template, load the scene's object inventory, and check whether
    the prediction's referent set uniquely identifies the same entities
    as some labeled correct choice (and NOT any labeled wrong choice).

    Returns a dict with reclaim counts and diagnostic counters identical
    in spirit to ``semantic_equivalence_audit.py``'s output.
    """
    n_total = len(records)
    n_strict = _strict_count(records)
    wrongs = [r for r in records if not r.get('correct')]

    n_no_scene = 0
    n_no_pred_event = 0
    n_reclaimed = 0
    n_blocked_by_wrong = 0
    reclaim_by_meta_key: Counter = Counter()

    for r in wrongs:
        scene_num = scene_id_to_num(r.get('scene_id', ''))
        if scene_num is None:
            continue
        inventory = load_scene_objects(clevrer_dir, scene_num)
        if inventory is None:
            n_no_scene += 1
            continue

        pred = norm(r.get('predicted', ''))
        if parse_event(pred) is None:
            n_no_pred_event += 1
            continue

        correct_texts = [c['choice'] for c in r['choices'] if c.get('answer') == 'correct']
        wrong_texts = [c['choice'] for c in r['choices'] if c.get('answer') == 'wrong']

        matched_correct = [c for c in correct_texts if is_semantic_match(pred, c, inventory)]
        matched_wrong = [w for w in wrong_texts if is_semantic_match(pred, w, inventory)]

        if matched_correct and not matched_wrong:
            n_reclaimed += 1
            r['_referent_reclaim'] = True
            meta = r.get('_phase9_meta') or {}
            # Use the eval's per-condition discriminator (tier / condition /
            # prompt_format) as the roll-up key.
            for k in ('tier', 'condition', 'prompt_format'):
                if k in meta:
                    reclaim_by_meta_key[(k, meta[k])] += 1
        elif matched_correct and matched_wrong:
            n_blocked_by_wrong += 1

    return {
        'n_total': n_total,
        'strict_correct': n_strict,
        'strict_acc_pct': round(100.0 * n_strict / max(n_total, 1), 2),
        'reclaimed': n_reclaimed,
        'blocked_by_ambiguity': n_blocked_by_wrong,
        'no_scene_file': n_no_scene,
        'no_pred_event_template': n_no_pred_event,
        'lenient_correct': n_strict + n_reclaimed,
        'lenient_acc_pct': round(100.0 * (n_strict + n_reclaimed) / max(n_total, 1), 2),
        'delta_pp': round(100.0 * n_reclaimed / max(n_total, 1), 2),
        'per_condition_reclaim': {
            f'{k}={v}': c for (k, v), c in reclaim_by_meta_key.items()
        },
    }


def _nli_threshold_sweep(records: List[Dict],
                         thresholds: List[float],
                         margin: float,
                         model_id: str,
                         device: str,
                         batch_size: int) -> Dict:
    """Run the bidirectional-NLI paraphrase audit on the wrong records,
    once, and report flip counts at every requested threshold.

    Single NLI pass amortised across all thresholds -- the per-(record,
    choice) bidirectional scores are independent of the cutoff. This
    keeps an N=200 sweep at 5 thresholds to ~3-5 min on a single GPU
    rather than 5x that.

    Records are mutated in-place to add ``_audit_*`` fields keyed by
    threshold so a downstream aggregator can roll up the flips by tier
    / condition / prompt_format.
    """
    wrongs = [r for r in records if not r.get('correct')]

    # build_pairs expects to walk the full records list (so the ``ri``
    # indices line up). We pass the wrong-only subset and accept that
    # ri indexes into ``wrongs`` for evaluate_flips' lookup.
    pairs, meta = build_pairs(wrongs, skip_verbatim_wrong=False)

    n_to_score = sum(1 for r in wrongs if not r.get('_audit_skipped'))
    out: Dict = {
        'n_total': len(records),
        'strict_correct': _strict_count(records),
        'pairs_scored': len(pairs),
        'records_audited': n_to_score,
        'thresholds': {},
    }

    if not pairs:
        return out

    scores = run_nli_batched(pairs,
                             model_id=model_id,
                             device=device,
                             batch_size=batch_size)

    # Capture once per threshold. evaluate_flips mutates record fields,
    # so we snapshot the per-record max-correct score and reuse it for
    # every cutoff rather than re-running.
    n_total = out['n_total']
    n_strict = out['strict_correct']

    # First pass: populate ``_audit_max_correct_paraphrase`` /
    # ``_audit_max_wrong_paraphrase`` on each wrong record. We use a
    # threshold of -1 + margin 0 so EVERY paraphrase score is captured;
    # the per-threshold counts come from a second-pass comparison
    # against the captured score, no extra NLI work.
    evaluate_flips(wrongs, meta, scores, threshold=-1.0, margin=0.0)

    # Per-condition roll-up scaffold: meta_key -> {threshold -> int}.
    per_cond: Dict[Tuple[str, str], Dict[float, int]] = defaultdict(lambda: defaultdict(int))

    for thr in thresholds:
        n_flips = 0
        for r in wrongs:
            if r.get('_audit_skipped'):
                continue
            mx = r.get('_audit_max_correct_paraphrase', -1.0)
            mw = r.get('_audit_max_wrong_paraphrase', -1.0)
            if mx >= thr and mx > mw + margin:
                n_flips += 1
                m = r.get('_phase9_meta') or {}
                for k in ('tier', 'condition', 'prompt_format'):
                    if k in m:
                        per_cond[(k, m[k])][thr] += 1
        out['thresholds'][f'{thr:.2f}'] = {
            'flips': n_flips,
            'lenient_correct': n_strict + n_flips,
            'lenient_acc_pct': round(100.0 * (n_strict + n_flips) / max(n_total, 1), 2),
            'delta_pp': round(100.0 * n_flips / max(n_total, 1), 2),
        }

    out['per_condition_flips'] = {
        f'{k}={v}': {f'{t:.2f}': c for t, c in counts.items()}
        for (k, v), counts in per_cond.items()
    }
    return out


def _per_condition_strict(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Pre-aggregate strict-substring acc by tier / condition / prompt_format.

    Mirrors the per-condition layout the existing eval-summary blocks
    print so the lenient-vs-strict comparison can be presented under
    one schema.
    """
    by_cond: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {'n': 0, 'correct': 0})
    for r in records:
        meta = r.get('_phase9_meta') or {}
        for k in ('tier', 'condition', 'prompt_format'):
            if k in meta:
                bucket = by_cond[(k, meta[k])]
                bucket['n'] += 1
                if r.get('correct'):
                    bucket['correct'] += 1
    out: Dict[str, Dict[str, float]] = {}
    for (k, v), b in by_cond.items():
        out[f'{k}={v}'] = {
            'n': b['n'],
            'strict_correct': b['correct'],
            'strict_acc_pct': round(100.0 * b['correct'] / max(b['n'], 1), 2),
        }
    return out


# ---------------------------------------------------------------------------
# Held-out-type categorical rescoring
# ---------------------------------------------------------------------------

def rescore_heldout_type(eval_json_path: Path) -> Dict:
    """Lenient rescoring for ``eval_phase9_heldout_type_n200.json``.

    The held-out-type eval doesn't ship an MCQ choice menu -- it only
    has ``gold_answer`` (a categorical / index / ordinal-bucket label
    from the QA generator) and the model's free-form ``predicted``.
    The strict scorer is a directional substring; the lenient rubric
    accepts:

      - ``'strict'``           : substring match (already in saved
                                 ``substring_correct`` flag).
      - ``'index'``            : pred resolves to the same 1-based
                                 object index as the gold (only for
                                 mass_comparison / speed_comparison
                                 index branch).
      - ``'synonym'``          : pred and gold map to the same
                                 canonical bucket via the per-qtype
                                 synonym table.
      - ``'ordinal_adjacent'`` : ordinal-bucket types only;
                                 pred is one bucket off from gold.

    Returns counts at three rigor levels: strict, synonym (incl. index),
    and ordinal-tol=1 (synonym OR one-bucket-off). All three are
    reported per qa_type plus overall.
    """
    with open(eval_json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    records = payload.get('records') or []

    # Three counters per (qa_type, rubric).
    counts: Dict[str, Dict[str, Counter]] = {}
    overall: Dict[str, Counter] = {
        'strict': Counter(),
        'synonym': Counter(),
        'ordinal_tol1': Counter(),
    }

    for r in records:
        qa_type = r.get('qa_type', '')
        pred = r.get('predicted', '')
        gold = r.get('gold_answer', '')
        ok_strict = bool(r.get('substring_correct', False))
        ok_synonym, _reason_s = categorical_correct(qa_type, pred, [gold], (), ordinal_tol=0)
        ok_ord, _reason_o = categorical_correct(qa_type, pred, [gold], (), ordinal_tol=1)

        bucket = counts.setdefault(qa_type, {
            'strict': Counter(),
            'synonym': Counter(),
            'ordinal_tol1': Counter(),
        })

        for rubric, ok in (('strict', ok_strict),
                           ('synonym', ok_synonym),
                           ('ordinal_tol1', ok_ord)):
            bucket[rubric]['n'] += 1
            if ok:
                bucket[rubric]['correct'] += 1
            overall[rubric]['n'] += 1
            if ok:
                overall[rubric]['correct'] += 1

    def _pct(c: Counter) -> Dict:
        n = max(c.get('n', 0), 1)
        return {
            'n': c.get('n', 0),
            'correct': c.get('correct', 0),
            'acc_pct': round(100.0 * c.get('correct', 0) / n, 2),
        }

    return {
        'eval_kind': 'heldout_type',
        'source_path': str(eval_json_path),
        'n_records': len(records),
        'overall': {rubric: _pct(c) for rubric, c in overall.items()},
        'per_qa_type': {
            qa_type: {rubric: _pct(c) for rubric, c in bucket.items()}
            for qa_type, bucket in counts.items()
        },
        'rubric_legend': {
            'strict': 'Saved substring_correct flag (eval-time strict rule).',
            'synonym': 'Strict OR pred-bucket equals gold-bucket via per-type synonym table OR pred-index equals gold-index for index-style answers.',
            'ordinal_tol1': 'Synonym OR pred-bucket is within 1 ordinal step of gold-bucket on ordinal-scale types (kinetic_energy, time_to_event, speed_comparison ordinal branch).',
        },
        'supported_qa_types': sorted(HELDOUT_TYPE_BUCKETS.keys()),
    }


# ---------------------------------------------------------------------------
# CLEVRER MCQ-style rescoring
# ---------------------------------------------------------------------------

def rescore_clevrer_eval(eval_json_path: Path,
                         kind: str,
                         clevrer_dir: Path,
                         nli_thresholds: List[float],
                         nli_margin: float,
                         nli_model: str,
                         nli_device: str,
                         nli_batch: int,
                         skip_nli: bool) -> Dict:
    """Run all rubrics on a Phase 9 CLEVRER MCQ-style eval JSON.

    Flattens the per-question multi-condition records via
    ``scoring.io.iter_legacy_records_*`` so each (question, condition)
    becomes one legacy record, then applies referent-equivalence and
    NLI threshold sweep.
    """
    with open(eval_json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    src_records = payload.get('records') or []
    flatten = PHASE9_FLATTENERS[kind]
    legacy = list(flatten(src_records))

    summary: Dict = {
        'eval_kind': kind,
        'source_path': str(eval_json_path),
        'n_source_records': len(src_records),
        'n_flattened_records': len(legacy),
        'strict_correct': _strict_count(legacy),
        'strict_acc_pct': round(100.0 * _strict_count(legacy) / max(len(legacy), 1), 2),
        'per_condition_strict': _per_condition_strict(legacy),
    }

    # Rubric 2: scene-aware referent equivalence.
    summary['referent_equivalence'] = _referent_reclaim(legacy, clevrer_dir)

    # Rubric 3: bidirectional-NLI paraphrase threshold sweep.
    if not skip_nli:
        summary['nli_paraphrase'] = _nli_threshold_sweep(
            legacy,
            thresholds=nli_thresholds,
            margin=nli_margin,
            model_id=nli_model,
            device=nli_device,
            batch_size=nli_batch,
        )
    else:
        summary['nli_paraphrase'] = {'skipped': True, 'reason': '--no_nli flag'}

    return summary


# ---------------------------------------------------------------------------
# Discovery + driver
# ---------------------------------------------------------------------------

def _discover_phase9_evals(input_dir: Path) -> Dict[str, Path]:
    """Find every recognised Phase 9 eval JSON under ``input_dir``.

    Returns ``{eval_kind: path}``. Multiple files of the same kind get
    deduplicated by lexicographic max (so ``eval_paraphrase_n5000.json``
    wins over ``eval_paraphrase_n200.json``).
    """
    if not input_dir.exists():
        raise FileNotFoundError(f'Phase 9 results dir not found: {input_dir}')
    found: Dict[str, Path] = {}
    for p in sorted(input_dir.glob('eval_*.json')):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        kind = detect_phase9_eval_kind(payload)
        if kind is None:
            continue
        # If the same eval kind appears more than once, keep the
        # alphabetically-greater path (largest n_NNN suffix wins).
        prev = found.get(kind)
        if prev is None or str(p) > str(prev):
            found[kind] = p
    return found


def _print_summary(payload: Dict) -> None:
    """Pretty-print the consolidated summary to stdout."""
    print()
    print('=' * 78)
    print('Phase 9 lenient rescoring -- summary')
    print('=' * 78)

    # Held-out type eval first (different schema).
    ht = payload.get('heldout_type')
    if ht:
        print()
        print('-- Held-out QUESTION TYPES (categorical match) -----------')
        print(f"  {'qa_type':<26}  {'n':>4}  {'strict':>8}  {'synonym':>9}  {'ord_tol1':>9}")
        for qt, b in sorted(ht['per_qa_type'].items()):
            print(f"  {qt:<26}  {b['strict']['n']:>4}  "
                  f"{b['strict']['acc_pct']:>7.2f}%  "
                  f"{b['synonym']['acc_pct']:>8.2f}%  "
                  f"{b['ordinal_tol1']['acc_pct']:>8.2f}%")
        oa = ht['overall']
        print(f"  {'OVERALL':<26}  {oa['strict']['n']:>4}  "
              f"{oa['strict']['acc_pct']:>7.2f}%  "
              f"{oa['synonym']['acc_pct']:>8.2f}%  "
              f"{oa['ordinal_tol1']['acc_pct']:>8.2f}%")

    # Three CLEVRER MCQ-style evals.
    for kind in ('paraphrase', 'prefix_ablation', 'free_form_transfer'):
        s = payload.get(kind)
        if not s:
            continue
        print()
        print(f'-- {kind} (CLEVRER MCQ-style) -----------')
        n = s['n_flattened_records']
        print(f"  n records (flattened):   {n:,}")
        print(f"  strict substring acc:    {s['strict_acc_pct']:>6.2f}%  "
              f"({s['strict_correct']:,}/{n:,})")
        re_ = s.get('referent_equivalence', {})
        if re_:
            print(f"  + referent reclaim:      {re_.get('lenient_acc_pct', 0):>6.2f}%  "
                  f"(+{re_.get('reclaimed', 0):,} reclaimed, "
                  f"{re_.get('blocked_by_ambiguity', 0)} ambiguity-blocked, "
                  f"{re_.get('no_scene_file', 0)} no-scene)")
        nli = s.get('nli_paraphrase', {})
        if 'thresholds' in nli:
            print(f"  + NLI paraphrase threshold sweep:")
            for t, row in sorted(nli['thresholds'].items()):
                print(f"      thr={t}  flips={row['flips']:>4}  "
                      f"lenient_acc={row['lenient_acc_pct']:>6.2f}%  "
                      f"delta=+{row['delta_pp']:>5.2f} pp")
        elif nli.get('skipped'):
            print(f"  + NLI paraphrase:        SKIPPED ({nli['reason']})")

        # Per-condition strict (sanity check that the converter preserved
        # the eval's intended condition split).
        pc = s.get('per_condition_strict', {})
        if pc:
            print(f"  per-condition strict:")
            for k, v in sorted(pc.items()):
                print(f"      {k:<30}  n={v['n']:>4}  acc={v['strict_acc_pct']:>6.2f}%")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Lenient rescoring of Phase 9 eval outputs (referent equivalence + NLI paraphrase + categorical synonym).')
    parser.add_argument('--input_dir', type=Path, default=DEFAULT_INPUT_DIR,
                        help=f'Phase 9 results dir (default: {DEFAULT_INPUT_DIR}).')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT,
                        help=f'Where to write the consolidated summary (default: {DEFAULT_OUTPUT}).')
    parser.add_argument('--clevrer_dir', type=Path, default=DEFAULT_CLEVRER,
                        help=f'CLEVRER root for scene loading (default: {DEFAULT_CLEVRER}).')
    parser.add_argument('--nli_thresholds', type=float, nargs='+',
                        default=DEFAULT_NLI_THRESHOLDS,
                        help=f'NLI paraphrase thresholds to sweep '
                             f'(default: {DEFAULT_NLI_THRESHOLDS}).')
    parser.add_argument('--nli_margin', type=float, default=DEFAULT_NLI_MARGIN,
                        help=f'NLI flip-rule margin over best-wrong score '
                             f'(default: {DEFAULT_NLI_MARGIN}).')
    parser.add_argument('--nli_model', type=str, default=DEFAULT_NLI_MODEL,
                        help=f'NLI model id (default: {DEFAULT_NLI_MODEL}).')
    parser.add_argument('--nli_batch', type=int, default=32,
                        help='NLI batch size (default: 32).')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Compute device (default: auto-select cuda if available).')
    parser.add_argument('--no_nli', action='store_true',
                        help='Skip the NLI threshold sweep (substring + referent + categorical only).')
    args = parser.parse_args()

    # Resolve device.
    device = args.device
    if device == 'auto':
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'

    print(f'[input_dir]  {args.input_dir}')
    print(f'[output]     {args.out}')
    print(f'[clevrer]    {args.clevrer_dir}')
    print(f'[nli_model]  {args.nli_model}  device={device}  batch={args.nli_batch}')
    print(f'[nli_thr]    {args.nli_thresholds}  margin={args.nli_margin}')
    print(f'[skip_nli]   {args.no_nli}')

    found = _discover_phase9_evals(args.input_dir)
    if not found:
        raise SystemExit(f'No recognised Phase 9 eval JSONs in {args.input_dir}')
    print(f'[found]      {sorted(found.keys())}')

    summary: Dict = {
        'config': {
            'input_dir': str(args.input_dir),
            'clevrer_dir': str(args.clevrer_dir),
            'nli_model': args.nli_model,
            'nli_thresholds': args.nli_thresholds,
            'nli_margin': args.nli_margin,
            'nli_batch': args.nli_batch,
            'device': device,
            'skip_nli': args.no_nli,
        },
        'sources': {k: str(v) for k, v in found.items()},
    }

    # Held-out-type eval first -- the only one that uses the categorical
    # rubric and runs without NLI / scene IO, so it's the cheapest sanity
    # check.
    if 'heldout_type' in found:
        print()
        print('=== heldout_type ===')
        summary['heldout_type'] = rescore_heldout_type(found['heldout_type'])

    # Three CLEVRER MCQ-style evals.
    for kind in ('paraphrase', 'prefix_ablation', 'free_form_transfer'):
        if kind not in found:
            continue
        print()
        print(f'=== {kind} ===')
        summary[kind] = rescore_clevrer_eval(
            eval_json_path=found[kind],
            kind=kind,
            clevrer_dir=args.clevrer_dir,
            nli_thresholds=args.nli_thresholds,
            nli_margin=args.nli_margin,
            nli_model=args.nli_model,
            nli_device=device,
            nli_batch=args.nli_batch,
            skip_nli=args.no_nli,
        )

    # Atomic save (a kill mid-flush leaves the previous file intact).
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    tmp.replace(args.out)

    _print_summary(summary)
    print(f'[saved]      {args.out}')


if __name__ == '__main__':
    main()
