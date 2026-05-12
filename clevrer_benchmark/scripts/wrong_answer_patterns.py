"""Pattern analysis of Phase 3 wrong predictions on CLEVRER held-out.

Reads ``phase3_BASELINE_SHUFFLE_FULL5000.details.jsonl`` (the paper's primary
run) and dissects every wrong record along multiple axes -- question type,
choice count, question length, presence of negation, scene object count, what
kind of confusion the model made -- so reviewers see a clean failure-mode
breakdown without re-running the eval.

What this script answers
------------------------
1.  Is the wrong rate concentrated in certain question types?
2.  Does it correlate with the number of choices offered?
3.  Does negation in the question (e.g. "what will *not* happen") hurt?
4.  When the model picks a wrong choice verbatim, what *kind* of confusion
    is it making? (object-pair swap / color swap / shape swap / cause swap)
5.  Are there hallucinated objects -- objects the model emitted that are
    not present in the scene's correct or wrong choices?
6.  Are some scenes systematically harder than others?

Usage::

    python compsac_2026_code/clevrer_benchmark/scripts/wrong_answer_patterns.py \\
        [--heldout] [--valid_only] [--details PATH]

Snapshot-portable: paths anchor on ``__file__``. Defaults to the held-out
valid-only pool (paper primary).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Snapshot-portable defaults.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _SCRIPT_DIR.parent
_RESULTS_DIR = _BENCH_DIR / 'results'
DEFAULT_DETAILS = _RESULTS_DIR / 'phase3_BASELINE_SHUFFLE_FULL5000.details.jsonl'
DEFAULT_OUTPUT = _RESULTS_DIR / 'wrong_answer_patterns.json'
DEFAULT_H5 = os.environ.get(
    'CLEVRER_H5',
    os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5'),
)

sys.path.insert(0, str(_SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return str(s).strip().lower()


CLEVRER_COLORS = (
    'gray', 'red', 'blue', 'green', 'brown', 'cyan', 'purple', 'yellow',
    'metal', 'rubber',  # treated as colors for token-extraction purposes
)
CLEVRER_SHAPES = ('cube', 'sphere', 'cylinder', 'object')
CLEVRER_MATERIALS = ('metal', 'rubber')
NEGATION_TOKENS = (
    'not ', 'without', "won't", "wouldn't", 'except', 'no longer',
    'fail to', 'never',
)


def _tokens_present(text: str, vocab) -> Set[str]:
    """Return the subset of ``vocab`` that appears as a token (word boundary)."""
    t = ' ' + _norm(text) + ' '
    return {v for v in vocab if (' ' + v + ' ') in t or (' ' + v + 's ') in t}


def _has_negation(question: str) -> bool:
    q = ' ' + _norm(question) + ' '
    return any(tok in q for tok in NEGATION_TOKENS)


def _scene_id_to_num(scene_id: str) -> Optional[int]:
    try:
        return int(scene_id.replace('annotation_', '').split('_')[-1])
    except (ValueError, IndexError):
        return None


def _wilson(correct: int, total: int) -> Tuple[float, float, float]:
    """Wilson 95% CI for a proportion."""
    z = 1.959963984540054
    if total <= 0:
        return (float('nan'), float('nan'), float('nan'))
    p = correct / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (p, max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# Confusion-type classifier for verbatim-wrong predictions
# ---------------------------------------------------------------------------

def _classify_confusion(pred: str,
                        correct_choices: List[str],
                        wrong_choices: List[str]) -> Dict[str, object]:
    """For a verbatim-wrong prediction, identify the closest correct choice
    and label what kind of single-attribute substitution turns one into the
    other. Returns a dict with the chosen correct, the chosen wrong (=pred),
    and a confusion category.
    """
    pred_n = _norm(pred)
    if not correct_choices:
        return {'category': 'no_correct_choice', 'closest_correct': None}

    # Tokenize correct choices and pred by whitespace; find the correct choice
    # with the largest token-overlap to the prediction.
    def toks(s: str) -> Set[str]:
        return set(_norm(s).split())

    pred_toks = toks(pred_n)
    best = None
    best_overlap = -1
    for c in correct_choices:
        c_toks = toks(c)
        overlap = len(pred_toks & c_toks)
        if overlap > best_overlap:
            best_overlap = overlap
            best = c

    closest_correct = best
    correct_toks = toks(closest_correct or '')
    diff_in_pred = pred_toks - correct_toks  # tokens in pred but not in correct
    diff_in_correct = correct_toks - pred_toks  # tokens in correct but not in pred

    pred_colors = _tokens_present(pred_n, CLEVRER_COLORS)
    correct_colors = _tokens_present(closest_correct or '', CLEVRER_COLORS)
    pred_shapes = _tokens_present(pred_n, CLEVRER_SHAPES)
    correct_shapes = _tokens_present(closest_correct or '', CLEVRER_SHAPES)
    pred_materials = _tokens_present(pred_n, CLEVRER_MATERIALS)
    correct_materials = _tokens_present(closest_correct or '', CLEVRER_MATERIALS)

    # Heuristic categories (most specific first).
    category = 'other'
    if pred_n == _norm(closest_correct or ''):
        category = 'identical_to_correct_bug'
    elif (pred_colors != correct_colors and
          pred_shapes == correct_shapes and
          pred_materials == correct_materials):
        category = 'color_swap'
    elif (pred_shapes != correct_shapes and
          pred_colors == correct_colors and
          pred_materials == correct_materials):
        category = 'shape_swap'
    elif (pred_materials != correct_materials and
          pred_colors == correct_colors and
          pred_shapes == correct_shapes):
        category = 'material_swap'
    elif (pred_colors != correct_colors and pred_shapes != correct_shapes):
        category = 'object_pair_swap'  # whole referent different
    elif ('collision' in pred_n) != ('collision' in (closest_correct or '')):
        category = 'event_form_swap'  # collision-event vs entrance/presence
    elif ('presence' in pred_n) != ('presence' in (closest_correct or '')):
        category = 'event_form_swap'
    elif ('entrance' in pred_n or "'s entering" in pred_n) != (
        'entrance' in (closest_correct or '') or "'s entering" in (closest_correct or '')):
        category = 'event_form_swap'

    return {
        'category': category,
        'closest_correct': closest_correct,
        'pred_unique_tokens': sorted(diff_in_pred),
        'correct_unique_tokens': sorted(diff_in_correct),
    }


# ---------------------------------------------------------------------------
# Hallucinated-object detector
# ---------------------------------------------------------------------------

def _scene_object_vocabulary(record: Dict) -> Set[str]:
    """Reconstruct the set of (color, shape, material) words that appear in the
    record's correct + wrong choices. The model has hallucinated when it emits
    a color/shape word outside this set.
    """
    vocab: Set[str] = set()
    for c in record['choices']:
        text = c.get('choice', '') if isinstance(c, dict) else str(c)
        vocab |= _tokens_present(text, CLEVRER_COLORS)
        vocab |= _tokens_present(text, CLEVRER_SHAPES)
        vocab |= _tokens_present(text, CLEVRER_MATERIALS)
    return vocab


def _hallucinated_tokens(pred: str, scene_vocab: Set[str]) -> Set[str]:
    pred_vocab = (_tokens_present(pred, CLEVRER_COLORS)
                  | _tokens_present(pred, CLEVRER_SHAPES)
                  | _tokens_present(pred, CLEVRER_MATERIALS))
    return pred_vocab - scene_vocab


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _pct(num: int, denom: int) -> str:
    if denom <= 0:
        return '   --'
    return f'{100.0 * num / denom:5.1f}%'


def _ci_str(correct: int, total: int) -> str:
    if total <= 0:
        return '--'
    p, lo, hi = _wilson(correct, total)
    return f'{p*100:5.1f}% [{lo*100:4.1f}, {hi*100:4.1f}]'


def _section(title: str) -> None:
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pattern analysis of Phase 3 wrong predictions.')
    parser.add_argument('--details', type=Path, default=DEFAULT_DETAILS,
                        help=f'Phase 3 details.jsonl to analyze '
                             f'(default: {DEFAULT_DETAILS.name}).')
    parser.add_argument('--heldout', action='store_true', default=True,
                        help='Restrict to held-out 501 scenes (default on).')
    parser.add_argument('--no_heldout', action='store_false', dest='heldout')
    parser.add_argument('--valid_only', action='store_true', default=True,
                        help='Drop zero-correct MCQ trap items (default on).')
    parser.add_argument('--no_valid_only', action='store_false', dest='valid_only')
    parser.add_argument('--h5', type=Path, default=Path(DEFAULT_H5),
                        help=f'CLEVRER training H5 path (only for --heldout). '
                             f'Default: {DEFAULT_H5} (also reads CLEVRER_H5).')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--samples_per_bucket', type=int, default=3,
                        help='How many sample wrongs to print per confusion '
                             'category (default 3).')
    args = parser.parse_args()

    if not args.details.exists():
        raise FileNotFoundError(f'Details file not found: {args.details}')

    heldout: Optional[Set[int]] = None
    if args.heldout:
        from compute_paper_stats import load_heldout_scenes  # type: ignore
        heldout = load_heldout_scenes(args.h5)
        print(f'[heldout] {len(heldout)} scenes loaded')

    # ---- Load and bucket records --------------------------------------
    n_total = 0
    n_correct = 0
    n_wrong = 0
    n_filtered_invalid = 0
    n_filtered_heldout = 0

    wrongs: List[Dict] = []
    correct_records: List[Dict] = []
    by_qtype: Dict[str, Dict[str, int]] = defaultdict(lambda: {'correct': 0, 'wrong': 0})
    by_nchoices: Dict[int, Dict[str, int]] = defaultdict(lambda: {'correct': 0, 'wrong': 0})
    by_negation: Dict[bool, Dict[str, int]] = defaultdict(lambda: {'correct': 0, 'wrong': 0})
    by_qlen_quartile: Dict[int, Dict[str, int]] = defaultdict(lambda: {'correct': 0, 'wrong': 0})
    by_scene: Dict[str, Dict[str, int]] = defaultdict(lambda: {'correct': 0, 'wrong': 0})

    print(f'[load] {args.details}')
    raw_records: List[Dict] = []
    with open(args.details, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if args.valid_only and not (r.get('correct_choices') or []):
                n_filtered_invalid += 1
                continue
            if heldout is not None:
                num = _scene_id_to_num(r.get('scene_id', ''))
                if num is None or num not in heldout:
                    n_filtered_heldout += 1
                    continue
            raw_records.append(r)

    # Compute question-length quartile thresholds across the kept pool.
    qlens = [len((r.get('question_text') or '').split()) for r in raw_records]
    qlens_sorted = sorted(qlens)
    if qlens_sorted:
        q1 = qlens_sorted[len(qlens_sorted) // 4]
        q2 = qlens_sorted[len(qlens_sorted) // 2]
        q3 = qlens_sorted[3 * len(qlens_sorted) // 4]
    else:
        q1 = q2 = q3 = 0

    def _qlen_quartile(n_words: int) -> int:
        if n_words <= q1:
            return 1
        if n_words <= q2:
            return 2
        if n_words <= q3:
            return 3
        return 4

    for r in raw_records:
        n_total += 1
        is_correct = bool(r.get('correct'))
        if is_correct:
            n_correct += 1
            correct_records.append(r)
        else:
            n_wrong += 1
            wrongs.append(r)
        bucket = 'correct' if is_correct else 'wrong'

        qtype = r.get('clevrer_type', 'unknown')
        choices = r.get('choices') or []
        n_choices = len(choices)
        question_text = r.get('question_text') or ''
        has_neg = _has_negation(question_text)

        by_qtype[qtype][bucket] += 1
        by_nchoices[n_choices][bucket] += 1
        by_negation[has_neg][bucket] += 1
        by_qlen_quartile[_qlen_quartile(len(question_text.split()))][bucket] += 1
        by_scene[r.get('scene_id', '?')][bucket] += 1

    _section('Pool')
    print(f'  total kept records:      {n_total:,}')
    print(f'  correct:                 {n_correct:,}  ({_pct(n_correct, n_total)})')
    print(f'  wrong:                   {n_wrong:,}  ({_pct(n_wrong, n_total)})')
    print(f'  filtered (zero-correct): {n_filtered_invalid:,}')
    print(f'  filtered (non-heldout):  {n_filtered_heldout:,}')

    # ---- Wrong rate by question type ----------------------------------
    _section('Wrong rate by question type')
    print(f'  {"type":<15} {"n":>6} {"correct":>9} {"wrong":>7} {"wrong %":>22}')
    print('  ' + '-' * 65)
    for qt in sorted(by_qtype.keys()):
        d = by_qtype[qt]
        n = d['correct'] + d['wrong']
        print(f'  {qt:<15} {n:>6} {d["correct"]:>9} {d["wrong"]:>7} '
              f'{_ci_str(d["wrong"], n):>22}')

    # ---- Wrong rate by number of choices ------------------------------
    _section('Wrong rate by number of MCQ choices')
    print(f'  {"# choices":<10} {"n":>6} {"wrong":>7} {"wrong %":>22}')
    print('  ' + '-' * 60)
    for k in sorted(by_nchoices.keys()):
        d = by_nchoices[k]
        n = d['correct'] + d['wrong']
        print(f'  {k:<10} {n:>6} {d["wrong"]:>7} {_ci_str(d["wrong"], n):>22}')

    # ---- Wrong rate with vs without negation in question --------------
    _section('Wrong rate with vs without question negation')
    print(f'  {"has negation":<15} {"n":>6} {"wrong":>7} {"wrong %":>22}')
    print('  ' + '-' * 60)
    for k in (False, True):
        d = by_negation[k]
        n = d['correct'] + d['wrong']
        print(f'  {str(k):<15} {n:>6} {d["wrong"]:>7} {_ci_str(d["wrong"], n):>22}')

    # ---- Wrong rate by question length quartile -----------------------
    _section('Wrong rate by question length quartile (in words)')
    print(f'  thresholds: q1<={q1}, q2<={q2}, q3<={q3}')
    print(f'  {"quartile":<15} {"n":>6} {"wrong":>7} {"wrong %":>22}')
    print('  ' + '-' * 60)
    for k in sorted(by_qlen_quartile.keys()):
        d = by_qlen_quartile[k]
        n = d['correct'] + d['wrong']
        print(f'  Q{k:<14} {n:>6} {d["wrong"]:>7} {_ci_str(d["wrong"], n):>22}')

    # ---- Confusion-type analysis on verbatim-wrong predictions --------
    _section('Confusion type on verbatim-wrong predictions')
    confusion_counts: Counter = Counter()
    confusion_samples: Dict[str, List[Dict]] = defaultdict(list)
    n_verbatim_wrong = 0
    n_freeform = 0
    n_substr_only = 0
    hallucinated_examples: List[Dict] = []
    for r in wrongs:
        pred = _norm(r.get('predicted', ''))
        correct_choices_t = [c['choice'] for c in r['choices'] if c.get('answer') == 'correct']
        wrong_choices_t = [c['choice'] for c in r['choices'] if c.get('answer') == 'wrong']
        is_verbatim_wrong = any(pred == _norm(w) for w in wrong_choices_t)
        is_verbatim_correct = any(pred == _norm(c) for c in correct_choices_t)
        if is_verbatim_correct:
            continue  # would be a scoring bug -- correct match yet flagged wrong
        if is_verbatim_wrong:
            n_verbatim_wrong += 1
            classification = _classify_confusion(pred, correct_choices_t, wrong_choices_t)
            cat = classification['category']
            confusion_counts[cat] += 1
            if len(confusion_samples[cat]) < args.samples_per_bucket:
                confusion_samples[cat].append({
                    'qtype': r.get('clevrer_type'),
                    'question': r.get('question_text'),
                    'closest_correct': classification['closest_correct'],
                    'predicted_wrong': pred,
                })
            continue
        # Free-form (no exact choice match).
        n_freeform += 1
        scene_vocab = _scene_object_vocabulary(r)
        halls = _hallucinated_tokens(pred, scene_vocab)
        if halls and len(hallucinated_examples) < 8:
            hallucinated_examples.append({
                'qtype': r.get('clevrer_type'),
                'predicted': pred,
                'scene_vocab': sorted(scene_vocab),
                'hallucinated': sorted(halls),
                'correct_choices': correct_choices_t,
            })
        # Could also be substring-overlap with a wrong (rare).
        if any((pred in _norm(w)) or (_norm(w) in pred) for w in wrong_choices_t):
            n_substr_only += 1

    print(f'  total wrongs:            {len(wrongs):,}')
    print(f'    verbatim-wrong picks:  {n_verbatim_wrong:,}  ({_pct(n_verbatim_wrong, len(wrongs))})')
    print(f'    free-form text:        {n_freeform:,}  ({_pct(n_freeform, len(wrongs))})')
    print(f'    substr-only on wrong:  {n_substr_only:,}')
    print()
    print(f'  Confusion category breakdown (verbatim-wrong only, n={n_verbatim_wrong:,}):')
    print(f'  {"category":<28} {"n":>6} {"% of vw":>10}')
    print('  ' + '-' * 50)
    for cat, n in confusion_counts.most_common():
        print(f'  {cat:<28} {n:>6}  {_pct(n, n_verbatim_wrong):>9}')

    # ---- Sample wrongs from each confusion category -------------------
    _section('Sample wrongs by confusion category')
    for cat, samples in confusion_samples.items():
        print(f'\n  --- {cat} ({confusion_counts[cat]:,} total) ---')
        for s in samples:
            print(f'   [{s["qtype"]}] {str(s["question"])[:100]}')
            print(f'      closest correct: {s["closest_correct"]!r}')
            print(f'      predicted (wrong): {s["predicted_wrong"]!r}')

    # ---- Hallucinated objects in free-form predictions ----------------
    _section('Hallucinated objects in free-form predictions')
    if not hallucinated_examples:
        print('  none detected (free-form predictions did not introduce out-of-scene tokens)')
    else:
        for h in hallucinated_examples:
            print(f'  [{h["qtype"]}] pred: {h["predicted"]!r}')
            print(f'    scene_vocab : {h["scene_vocab"]}')
            print(f'    hallucinated: {h["hallucinated"]}')
            print(f'    correct(s)  : {h["correct_choices"]}')
            print()

    # ---- Per-scene hardness distribution ------------------------------
    _section('Per-scene hardness (top hardest scenes)')
    scene_stats = []
    for sid, d in by_scene.items():
        n = d['correct'] + d['wrong']
        if n < 2:  # ignore singletons; need at least 2 questions for a rate
            continue
        scene_stats.append((sid, d['wrong'], n, d['wrong'] / n))
    scene_stats.sort(key=lambda t: (-t[3], -t[2]))
    print(f'  {"scene":<22} {"wrong":>7} {"n":>5} {"wrong %":>9}')
    print('  ' + '-' * 50)
    for sid, w, n, rate in scene_stats[:15]:
        print(f'  {sid:<22} {w:>7} {n:>5} {rate*100:>8.1f}%')

    # ---- Negation Fisher's exact test (rough significance) ------------
    try:
        from scipy.stats import fisher_exact  # type: ignore
        cw = by_negation[True]
        nw = by_negation[False]
        n_with_w = cw['wrong']
        n_with_c = cw['correct']
        n_no_w = nw['wrong']
        n_no_c = nw['correct']
        _, p = fisher_exact([[n_with_w, n_with_c], [n_no_w, n_no_c]],
                            alternative='two-sided')
        _section('Negation Fisher\'s exact test (pooled across types)')
        print(f'  with-negation  wrong/correct: {n_with_w}/{n_with_c}')
        print(f'  no-negation    wrong/correct: {n_no_w}/{n_no_c}')
        print(f'  two-sided p-value: {p:.4g}')
    except ImportError:
        pass

    # ===== FOLLOW-UP ANALYSES (control for question-type confound) =====

    # ---- Within-question-type breakdowns ----
    _section('Within-type breakdowns (control for question-type confound)')
    types_present = sorted({r.get('clevrer_type', 'unknown') for r in raw_records})
    within_type_summary: Dict[str, Dict] = {}
    for qt in types_present:
        type_records = [r for r in raw_records if r.get('clevrer_type') == qt]
        if not type_records:
            continue
        n_t = len(type_records)
        wrong_t = sum(1 for r in type_records if not r.get('correct'))
        print(f'\n  --- {qt} (n={n_t}, wrong={wrong_t}, {_ci_str(wrong_t, n_t)}) ---')

        # by # choices within type
        ncs: Dict[int, Dict[str, int]] = defaultdict(lambda: {'correct': 0, 'wrong': 0})
        for r in type_records:
            k = len(r.get('choices') or [])
            ncs[k]['correct' if r.get('correct') else 'wrong'] += 1
        print(f'    by # choices:')
        print(f'      {"k":<6} {"n":>5} {"wrong":>6} {"wrong %":>22}')
        for k in sorted(ncs.keys()):
            d = ncs[k]
            n = d['correct'] + d['wrong']
            print(f'      {k:<6} {n:>5} {d["wrong"]:>6} {_ci_str(d["wrong"], n):>22}')

        # by negation within type
        negs: Dict[bool, Dict[str, int]] = defaultdict(lambda: {'correct': 0, 'wrong': 0})
        for r in type_records:
            has_neg = _has_negation(r.get('question_text') or '')
            negs[has_neg]['correct' if r.get('correct') else 'wrong'] += 1
        print(f'    by negation:')
        print(f'      {"neg?":<6} {"n":>5} {"wrong":>6} {"wrong %":>22}')
        for k in (False, True):
            d = negs[k]
            n = d['correct'] + d['wrong']
            print(f'      {str(k):<6} {n:>5} {d["wrong"]:>6} {_ci_str(d["wrong"], n):>22}')
        # Fisher within type, if scipy available.
        try:
            from scipy.stats import fisher_exact  # type: ignore
            tw = negs[True]
            fw = negs[False]
            if (tw['correct'] + tw['wrong']) > 0 and (fw['correct'] + fw['wrong']) > 0:
                _, pv = fisher_exact([[tw['wrong'], tw['correct']],
                                       [fw['wrong'], fw['correct']]],
                                      alternative='two-sided')
                print(f'      negation Fisher p (within {qt}): {pv:.4g}')
        except ImportError:
            pass

        # by within-type length quartile (recompute thresholds per type)
        type_qlens = sorted(
            len((r.get('question_text') or '').split()) for r in type_records)
        if type_qlens:
            tq1 = type_qlens[len(type_qlens) // 4]
            tq2 = type_qlens[len(type_qlens) // 2]
            tq3 = type_qlens[3 * len(type_qlens) // 4]
        else:
            tq1 = tq2 = tq3 = 0

        def _bucket(n_words: int, q1=tq1, q2=tq2, q3=tq3) -> int:
            if n_words <= q1:
                return 1
            if n_words <= q2:
                return 2
            if n_words <= q3:
                return 3
            return 4

        ql: Dict[int, Dict[str, int]] = defaultdict(lambda: {'correct': 0, 'wrong': 0})
        for r in type_records:
            qb = _bucket(len((r.get('question_text') or '').split()))
            ql[qb]['correct' if r.get('correct') else 'wrong'] += 1
        print(f'    by within-type length quartile (q1<={tq1}, q2<={tq2}, q3<={tq3}):')
        print(f'      {"qrt":<6} {"n":>5} {"wrong":>6} {"wrong %":>22}')
        for k in sorted(ql.keys()):
            d = ql[k]
            n = d['correct'] + d['wrong']
            print(f'      Q{k:<5} {n:>5} {d["wrong"]:>6} {_ci_str(d["wrong"], n):>22}')

        within_type_summary[qt] = {
            'n': n_t,
            'wrong': wrong_t,
            'by_nchoices': {k: {'n': v['correct'] + v['wrong'], 'wrong': v['wrong']}
                            for k, v in ncs.items()},
            'by_negation': {str(k): {'n': v['correct'] + v['wrong'], 'wrong': v['wrong']}
                             for k, v in negs.items()},
            'qlen_thresholds': [tq1, tq2, tq3],
            'by_qlen_quartile': {k: {'n': v['correct'] + v['wrong'], 'wrong': v['wrong']}
                                  for k, v in ql.items()},
        }

    # ---- Confusion-type cross-tab by question type ----
    _section('Confusion type x question type cross-tab')
    confusion_xtab: Dict[str, Counter] = defaultdict(Counter)
    type_vw_totals: Counter = Counter()
    for r in wrongs:
        pred = _norm(r.get('predicted', ''))
        correct_choices_t = [c['choice'] for c in r['choices'] if c.get('answer') == 'correct']
        wrong_choices_t = [c['choice'] for c in r['choices'] if c.get('answer') == 'wrong']
        is_verbatim_correct = any(pred == _norm(c) for c in correct_choices_t)
        if is_verbatim_correct:
            continue
        is_verbatim_wrong = any(pred == _norm(w) for w in wrong_choices_t)
        qt = r.get('clevrer_type', 'unknown')
        if not is_verbatim_wrong:
            confusion_xtab[qt]['free_form'] += 1
            type_vw_totals[qt] += 0  # don't bump verbatim-wrong total for free-form
            continue
        cls = _classify_confusion(pred, correct_choices_t, wrong_choices_t)
        confusion_xtab[qt][cls['category']] += 1
        type_vw_totals[qt] += 1

    cat_keys = sorted({c for cats in confusion_xtab.values() for c in cats.keys()})
    print(f'  rows = question type, columns = confusion category (% of within-type verbatim-wrong)')
    header = f'  {"qtype":<16}' + ''.join(f'{c[:14]:>15}' for c in cat_keys) + f'{"vw_total":>11}'
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for qt in sorted(confusion_xtab.keys()):
        row = f'  {qt:<16}'
        for c in cat_keys:
            n = confusion_xtab[qt].get(c, 0)
            denom = max(type_vw_totals[qt], 1) if c != 'free_form' else max(
                sum(confusion_xtab[qt].values()), 1)
            row += f'{n:>5} ({100.0*n/denom:>4.1f}%)'
        row += f'{type_vw_totals[qt]:>11}'
        print(row)

    # ---- Catastrophic-scene deep dive ----
    _section('Catastrophic-scene deep dive (worst 5 scenes by wrong rate)')
    # Pull every record for the worst scenes (where 100% of questions wrong, n>=2).
    worst_scenes = [sid for sid, w, n, rate in scene_stats[:5]
                    if rate >= 0.99 and n >= 2]
    scene_records: Dict[str, List[Dict]] = defaultdict(list)
    for r in raw_records:
        sid = r.get('scene_id', '?')
        if sid in worst_scenes:
            scene_records[sid].append(r)
    for sid in worst_scenes:
        recs = scene_records[sid]
        print(f'\n  --- {sid}  (n={len(recs)} questions, all wrong) ---')
        # Reconstruct scene-vocab to see object count
        scene_vocab_all: Set[str] = set()
        for r in recs:
            scene_vocab_all |= _scene_object_vocabulary(r)
        n_color_tokens = len(scene_vocab_all & set(CLEVRER_COLORS))
        print(f'    scene-token vocab: {sorted(scene_vocab_all)}  '
              f'({n_color_tokens} color/material tokens)')
        for r in recs[:6]:  # cap printout at 6 questions
            qt = r.get('clevrer_type', '?')
            qtxt = (r.get('question_text') or '')[:90]
            corr = [c['choice'] for c in r['choices'] if c.get('answer') == 'correct']
            pred = r.get('predicted', '')
            print(f'    [{qt}] Q: {qtxt}')
            print(f'      correct(s): {corr}')
            print(f'      predicted:  {pred!r}')

    # ---- Capture follow-up analyses in the saved summary ----
    follow_up_summary = {
        'within_type_breakdowns': within_type_summary,
        'confusion_type_by_question_type': {
            qt: dict(cnts) for qt, cnts in confusion_xtab.items()
        },
        'verbatim_wrong_totals_by_qtype': dict(type_vw_totals),
        'catastrophic_scenes': [
            {'scene_id': sid,
             'n_questions': len(scene_records[sid]),
             'records': [
                 {'qtype': r.get('clevrer_type'),
                  'question': r.get('question_text'),
                  'correct_choices': [c['choice'] for c in r['choices']
                                       if c.get('answer') == 'correct'],
                  'wrong_choices': [c['choice'] for c in r['choices']
                                     if c.get('answer') == 'wrong'],
                  'predicted': r.get('predicted')}
                 for r in scene_records[sid]
             ]}
            for sid in worst_scenes
        ],
    }

    # ---- Save full summary ---------------------------------------------
    summary = {
        'pool': {
            'total_kept': n_total,
            'correct': n_correct,
            'wrong': n_wrong,
            'filtered_invalid': n_filtered_invalid,
            'filtered_non_heldout': n_filtered_heldout,
            'heldout_only': args.heldout,
            'valid_only': args.valid_only,
        },
        'wrong_rate_by_qtype': {
            qt: {
                'n': d['correct'] + d['wrong'],
                'wrong': d['wrong'],
                'wrong_pct': round(100.0 * d['wrong'] / max(d['correct'] + d['wrong'], 1), 2),
            } for qt, d in by_qtype.items()
        },
        'wrong_rate_by_nchoices': {
            k: {
                'n': d['correct'] + d['wrong'],
                'wrong': d['wrong'],
                'wrong_pct': round(100.0 * d['wrong'] / max(d['correct'] + d['wrong'], 1), 2),
            } for k, d in by_nchoices.items()
        },
        'wrong_rate_by_negation': {
            str(k): {
                'n': d['correct'] + d['wrong'],
                'wrong': d['wrong'],
                'wrong_pct': round(100.0 * d['wrong'] / max(d['correct'] + d['wrong'], 1), 2),
            } for k, d in by_negation.items()
        },
        'verbatim_split': {
            'verbatim_wrong': n_verbatim_wrong,
            'free_form': n_freeform,
            'substr_only_on_wrong': n_substr_only,
        },
        'confusion_counts': dict(confusion_counts),
        'confusion_samples': {k: v for k, v in confusion_samples.items()},
        'hallucinated_examples': hallucinated_examples,
        'top_hardest_scenes': [
            {'scene_id': sid, 'wrong': w, 'n': n, 'wrong_pct': round(rate * 100, 1)}
            for sid, w, n, rate in scene_stats[:25]
        ],
    }

    summary['follow_up'] = follow_up_summary

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\n  full summary written to: {args.out}')


if __name__ == '__main__':
    main()
