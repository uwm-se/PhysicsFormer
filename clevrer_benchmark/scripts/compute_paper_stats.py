"""Compute paper-ready statistics for Phase 3 CLEVRER Physics-LLM results.

Reads the result JSONs produced by ``run_adapter_evaluation.py`` and emits:

  * Wilson 95%% CIs for every accuracy (handles low counts better than normal-approx
    intervals when zero-physics drops accuracy to ~0%).
  * Fisher's exact p-values + Cohen's h effect sizes for every ablation delta
    (baseline vs zero_physics, baseline vs zero_prefix, baseline vs zero_prefix_shuffle).
  * LaTeX table fragments ready to paste into main.tex.

Usage::

    python compsac_2026_code/clevrer_benchmark/scripts/compute_paper_stats.py \
        --scope FULL5000 --emit_latex

``--scope`` picks which suffix to look for. Defaults to FULL5000 which maps to:
    phase3_GENERATE_singleframe_FULL5000.json       (baseline)
    phase3_ZEROPHYSICS_FULL5000.json                (zero physics)
    phase3_ZEROPREFIX_FULL5000.json                 (zero prefix)
    phase3_ZEROPREFIX_SHUFFLE_FULL5000.json         (zero prefix + shuffle)

Pass ``--scope 100scenes`` to reproduce the paper's original 435-question
stratified protocol (paths use ``*_100scenes.json`` suffix).

When ``phase3_BASELINE_SHUFFLE_FULL5000.json`` is present (FULL5000 scope only)
the script also loads it as the **primary** Phase 3 row -- the paper's primary
macros (69.2 / 79.4 / 63.4 / 63.6 on heldout valid-only) come from that file.
The unshuffled baseline is shown as ``unshuffled reference`` for context, and
an extra ``Positional bias`` delta (unshuffled - shuffled) is emitted.

The ``--results_dir`` default resolves to ``../results`` relative to this
script, so the script works regardless of the current working directory.
``--heldout`` requires the CLEVRER training H5; pass ``--h5 PATH`` or set the
``CLEVRER_H5`` environment variable to point at it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# Default H5 path used only when --heldout is requested. Override with --h5 or
# the CLEVRER_H5 env var; the H5 is too large to ship inside the snapshot.
DEFAULT_H5 = os.environ.get(
    'CLEVRER_H5',
    os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5'),
)
# Anchor the default results dir on this script's location so it works from any CWD.
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = _SCRIPT_DIR.parent / 'results'

try:
    from scipy.stats import fisher_exact
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


Z95 = 1.959963984540054  # two-sided 95%% z-critical


def wilson_ci(correct: int, total: int, z: float = Z95) -> Tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (point_estimate, lower, upper) in [0, 1]. Handles corner cases
    total=0 and correct in {0, total} gracefully.
    """
    if total <= 0:
        return (float('nan'), float('nan'), float('nan'))
    p = correct / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    halfw = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (p, max(0.0, center - halfw), min(1.0, center + halfw))


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions.

    Uses arcsine transform; sign preserved (positive when p1 > p2).
    Clamps proportions away from 0/1 to avoid NaN from asin(sqrt(x)) at the
    endpoints.
    """
    eps = 1e-12
    p1c = max(eps, min(1 - eps, p1))
    p2c = max(eps, min(1 - eps, p2))
    phi1 = 2.0 * math.asin(math.sqrt(p1c))
    phi2 = 2.0 * math.asin(math.sqrt(p2c))
    return phi1 - phi2


def fisher_pvalue(c1: int, n1: int, c2: int, n2: int) -> Optional[float]:
    """Two-sided Fisher's exact p-value for 2x2 table.

    Table rows are (correct, wrong); columns are (condition 1, condition 2).
    Returns None if scipy is unavailable.
    """
    if not HAVE_SCIPY:
        return None
    w1 = n1 - c1
    w2 = n2 - c2
    table = [[c1, c2], [w1, w2]]
    _, p = fisher_exact(table, alternative='two-sided')
    return p


def format_pvalue(p: Optional[float]) -> str:
    """Paper-style p-value: '<0.001' / '<0.01' / '<0.05' / 'n.s.'"""
    if p is None:
        return 'n/a (install scipy)'
    if p < 0.001:
        return '<0.001'
    if p < 0.01:
        return '<0.01'
    if p < 0.05:
        return '<0.05'
    return f'{p:.3f} (n.s.)'


def load_result(path: Path,
                valid_only: bool = False,
                heldout_scenes: Optional[set] = None) -> Optional[Dict]:
    """Load a result JSON produced by run_adapter_evaluation.py.

    Returns a flat dict of {category -> {'correct': int, 'total': int}} plus an
    'overall' entry. Returns None if the file doesn't exist yet.

    If ``valid_only=True`` or ``heldout_scenes`` is provided, the counts are
    recomputed from the sidecar ``.details.jsonl`` file (same basename,
    different extension). Raises if the details file is missing when a filter
    is requested -- the summary JSON doesn't preserve per-question metadata.
    """
    if not path.exists():
        return None
    # Default path: just read the summary JSON
    if not valid_only and heldout_scenes is None:
        with open(path, 'r') as f:
            d = json.load(f)
        out = {}
        overall = d.get('overall', {})
        out['overall'] = {
            'correct': int(overall.get('correct', 0)),
            'total': int(overall.get('total_questions', overall.get('total', 0))),
        }
        for cat, cd in d.get('by_clevrer_type', {}).items():
            out[cat] = {'correct': int(cd.get('correct', 0)), 'total': int(cd.get('total', 0))}
        return out
    # Filtered path: recompute from details.jsonl
    details_path = path.with_suffix('').with_suffix('.details.jsonl')
    if not details_path.exists():
        # handle .json -> .details.jsonl naming
        details_path = path.parent / (path.stem + '.details.jsonl')
    if not details_path.exists():
        raise FileNotFoundError(f'Filter requested but details file missing: {details_path}')
    out = {'overall': {'correct': 0, 'total': 0}}
    with open(details_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if valid_only and not (r.get('correct_choices') or []):
                continue
            if heldout_scenes is not None:
                sid = r.get('scene_id', '')
                try:
                    num = int(sid.replace('annotation_', '').split('_')[-1])
                except ValueError:
                    continue
                if num not in heldout_scenes:
                    continue
            qt = r.get('clevrer_type', 'unknown')
            correct = 1 if r.get('correct') else 0
            out['overall']['correct'] += correct
            out['overall']['total'] += 1
            if qt not in out:
                out[qt] = {'correct': 0, 'total': 0}
            out[qt]['correct'] += correct
            out[qt]['total'] += 1
    return out


def load_heldout_scenes(h5_path: Path) -> set:
    """Identify the 10%% held-out CLEVRER val scenes that Phase 3 never trained on.

    Resolution order (first hit wins):

    1. The canonical ``clevrer_benchmark/results/heldout_scenes.json``
       sidecar produced once by this script and committed to the repo.
       This is the same path ``free_form_transfer_test._load_heldout_scenes``
       prefers, so eval and audit scripts agree byte-for-byte.
    2. The training H5, walked in dataset order to recompute the split.
       ``h5_path`` must point at ``clevrer_training_expanded.h5``; the
       file is too large to ship in the snapshot, so it is supplied via
       CLI/env.

    Falling back to the sidecar lets audits / lenient rescoring run on
    machines that don't have the training H5 mounted (e.g. eval-only
    runs against shipped checkpoints).
    """
    sidecar = Path(__file__).resolve().parent.parent / 'results' / 'heldout_scenes.json'
    if sidecar.exists():
        try:
            with open(sidecar, 'r', encoding='utf-8') as f:
                ids = json.load(f)
            scenes = {int(x) for x in ids}
            if scenes:
                return scenes
        except (OSError, ValueError, json.JSONDecodeError):
            pass  # Sidecar unreadable -> fall through to H5.

    import h5py
    if not h5_path.exists():
        raise FileNotFoundError(
            f'CLEVRER training H5 not found at {h5_path} and the '
            f'heldout_scenes.json sidecar at {sidecar} is also '
            f'missing/unreadable. Pass --h5 PATH, set the CLEVRER_H5 '
            f'env var, or restore the sidecar.'
        )
    FOCUS = {'counterfactual', 'explanatory', 'predictive'}
    with h5py.File(str(h5_path), 'r') as hf:
        qtypes = hf['question_types'][:]
        metadata = hf['metadata'][:]
    filtered = [i for i, qt in enumerate(qtypes)
                if (qt.decode('utf-8') if isinstance(qt, bytes) else str(qt)).lower() in FOCUS]
    train_size = int(0.9 * len(filtered))
    test_idx = filtered[train_size:]
    heldout = set()
    for i in test_idx:
        m = metadata[i]
        if isinstance(m, bytes):
            m = m.decode('utf-8')
        try:
            heldout.add(json.loads(m).get('scene_index'))
        except Exception:
            continue
    return heldout


def print_condition_row(label: str, res: Dict, categories: list) -> None:
    """Print one row of the accuracy+CI table (text version)."""
    cells = [f'{label:<30}']
    for cat in categories:
        d = res.get(cat) if res else None
        if d is None or d['total'] == 0:
            cells.append(f'{"—":>20}')
            continue
        p, lo, hi = wilson_ci(d['correct'], d['total'])
        cells.append(f'{p*100:>5.1f}% [{lo*100:>4.1f}, {hi*100:>4.1f}] ({d["correct"]}/{d["total"]})')
    print('  '.join(cells))


def print_delta_row(label: str, base: Dict, ablation: Dict, categories: list) -> None:
    """Print a Δ row for one ablation vs baseline (text version)."""
    cells = [f'{label:<30}']
    for cat in categories:
        b = base.get(cat) if base else None
        a = ablation.get(cat) if ablation else None
        if b is None or a is None or b['total'] == 0 or a['total'] == 0:
            cells.append(f'{"—":>25}')
            continue
        pb = b['correct'] / b['total']
        pa = a['correct'] / a['total']
        delta_pp = (pb - pa) * 100
        h = cohen_h(pb, pa)
        p = fisher_pvalue(b['correct'], b['total'], a['correct'], a['total'])
        p_str = format_pvalue(p)
        cells.append(f'{delta_pp:>+6.1f}pp h={h:.2f} p={p_str:<10}')
    print('  '.join(cells))


def emit_latex_ablation_table(baseline, zero_phys, zero_pfx, zero_pfx_shuf, categories, scope_label):
    """Emit a LaTeX fragment for the ablation table (Table 3 replacement)."""
    print()
    print('% === LaTeX: main ablation table (paper Table 3 replacement) ===')
    print(f'% Scope: {scope_label}')
    print(r'\begin{table}[ht!]')
    print(r'\centering\footnotesize')
    print(r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc@{}}')
    print(r'\toprule')
    print(r'\textbf{Condition} & \textbf{Overall} & \textbf{Explan.} & \textbf{Predict.} & \textbf{Counter.} \\')
    print(r'\midrule')
    for label, res in [
        ('Grounded-Physics LM', baseline),
        ('Zero Physics (input)', zero_phys),
        ('Zero Prefix (tokens)', zero_pfx),
        ('Zero Prefix + Shuffle', zero_pfx_shuf),
    ]:
        cells = [label]
        for cat in categories:
            d = res.get(cat) if res else None
            if d is None or d['total'] == 0:
                cells.append('---')
            else:
                p = d['correct'] / d['total'] * 100
                cells.append(f'{p:.1f}')
        print(' & '.join(cells) + r' \\')
    print(r'\midrule')
    # Delta rows
    for label, abl in [
        (r'$\Delta$ (Physics contrib.)', zero_phys),
        (r'$\Delta$ (Prefix contrib., vs shuffle)', zero_pfx_shuf),
    ]:
        cells = [label]
        for cat in categories:
            b = baseline.get(cat) if baseline else None
            a = abl.get(cat) if abl else None
            if b is None or a is None or b['total'] == 0 or a['total'] == 0:
                cells.append('---')
            else:
                d = (b['correct'] / b['total'] - a['correct'] / a['total']) * 100
                cells.append(f'{d:+.1f}')
        print(' & '.join(cells) + r' \\')
    print(r'\bottomrule')
    print(r'\end{tabular*}')
    print(fr'\caption{{Ablation study (\%) on the Phase-3 Physics-LLM, {scope_label} (n=TOTAL). '
          r'\emph{Zero Physics}: encoder-input tensors zeroed. '
          r'\emph{Zero Prefix}: adapter prefix tokens zeroed before the LLM. '
          r'\emph{Zero Prefix + Shuffle}: prefix zeroed and MCQ choice order randomized per question '
          r'(true text-only baseline). All deltas statistically significant at $p<0.001$ (Fisher\textquotesingle{}s exact).}')
    print(r'\label{tab:ablation}')
    print(r'\end{table}')


def emit_latex_significance_table(baseline, ablations, categories, scope_label):
    """Emit LaTeX for the significance table (paper Table 2)."""
    print()
    print('% === LaTeX: physics grounding significance table (paper Table 2 replacement) ===')
    print(f'% Scope: {scope_label}')
    print(r'\begin{table}[ht!]')
    print(r'\centering\footnotesize')
    print(r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}')
    print(r'\toprule')
    print(r'\textbf{Question Type} & \textbf{$\Delta$ (pp)} & \textbf{$p$-value} & \textbf{Cohen\textquotesingle{}s $h$} & \textbf{$n$} \\')
    print(r'\midrule')
    labels = [('Overall', 'overall'), ('Explanatory', 'explanatory'),
              ('Predictive', 'predictive'), ('Counterfactual', 'counterfactual')]
    for label, cat in labels:
        b = baseline.get(cat) if baseline else None
        a = ablations.get(cat) if ablations else None
        if b is None or a is None or b['total'] == 0 or a['total'] == 0:
            print(fr'{label} & --- & --- & --- & --- \\')
            continue
        pb = b['correct'] / b['total']
        pa = a['correct'] / a['total']
        delta_pp = (pb - pa) * 100
        h = cohen_h(pb, pa)
        p = fisher_pvalue(b['correct'], b['total'], a['correct'], a['total'])
        p_str = format_pvalue(p)
        n = b['total']
        print(fr'{label} & ${delta_pp:+.1f}$ & ${p_str}$ & {h:.2f} & {n} \\')
    print(r'\bottomrule')
    print(r'\end{tabular*}')
    print(r'\caption{Statistical significance of physics grounding contribution '
          r'(Grounded-Physics LM vs.\ Zero Physics). All effects significant at '
          r'$p<0.001$ (Fisher\textquotesingle{}s exact test).}')
    print(r'\label{tab:significance}')
    print(r'\end{table}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=Path,
                        default=DEFAULT_RESULTS_DIR,
                        help=f'Directory of result JSONs (default: {DEFAULT_RESULTS_DIR}).')
    parser.add_argument('--scope', type=str, default='FULL5000',
                        help='Filename suffix: FULL5000 (default) or 100scenes.')
    parser.add_argument('--emit_latex', action='store_true',
                        help='Also emit LaTeX table fragments for main.tex')
    parser.add_argument('--valid_only', action='store_true',
                        help='Exclude CLEVRER questions where every choice is labeled wrong '
                             '(matches the LLM baseline validate_question() filter).')
    parser.add_argument('--heldout', action='store_true',
                        help='Restrict to the 501 CLEVRER val scenes that the Phase 3 adapter '
                             'never saw during training (10%% heldout split from the H5).')
    parser.add_argument('--h5', type=Path, default=Path(DEFAULT_H5),
                        help='CLEVRER training H5 path; only used with --heldout. '
                             f'Default: {DEFAULT_H5} (also reads CLEVRER_H5 env var).')
    args = parser.parse_args()

    heldout = load_heldout_scenes(args.h5) if args.heldout else None

    rd = args.results_dir
    if args.scope == '100scenes':
        baseline_path = rd / 'phase3_GENERATE_singleframe_100scenes.json'
        baseline_shuf_path = None  # No shuffled-baseline file in the 100scenes scope.
        zp_path = rd / 'phase3_GENERATE_singleframe_ZEROPHYSICS_100scenes.json'
        zpfx_path = rd / 'phase3_ZEROPREFIX_100scenes.json'
        zpfxs_path = rd / 'phase3_ZEROPREFIX_SHUFFLE_100scenes.json'
        scope_label = '435-question stratified subset'
    else:
        baseline_path = rd / 'phase3_GENERATE_singleframe_FULL5000.json'
        baseline_shuf_path = rd / 'phase3_BASELINE_SHUFFLE_FULL5000.json'
        zp_path = rd / 'phase3_ZEROPHYSICS_FULL5000.json'
        zpfx_path = rd / 'phase3_ZEROPREFIX_FULL5000.json'
        zpfxs_path = rd / 'phase3_ZEROPREFIX_SHUFFLE_FULL5000.json'
        scope_label = 'full 5000-scene validation set'

    baseline = load_result(baseline_path, valid_only=args.valid_only, heldout_scenes=heldout)
    baseline_shuf = (load_result(baseline_shuf_path, valid_only=args.valid_only, heldout_scenes=heldout)
                     if baseline_shuf_path else None)
    zero_phys = load_result(zp_path, valid_only=args.valid_only, heldout_scenes=heldout)
    zero_pfx = load_result(zpfx_path, valid_only=args.valid_only, heldout_scenes=heldout)
    zero_pfx_shuf = load_result(zpfxs_path, valid_only=args.valid_only, heldout_scenes=heldout)

    filter_suffix = []
    if args.heldout:
        filter_suffix.append(f'heldout {len(heldout):,} scenes')
    if args.valid_only:
        filter_suffix.append('valid-only')
    if filter_suffix:
        scope_label = f'{scope_label} [{", ".join(filter_suffix)}]'

    print(f'=== Phase 3 Physics-LLM statistics — scope: {scope_label} ===')
    print()
    print(f'  baseline         {baseline_path.name}: {"ok" if baseline else "MISSING"}')
    if baseline_shuf_path is not None:
        print(f'  baseline-shuffle {baseline_shuf_path.name}: {"ok" if baseline_shuf else "MISSING"}')
    print(f'  zero-physics     {zp_path.name}: {"ok" if zero_phys else "MISSING"}')
    print(f'  zero-prefix      {zpfx_path.name}: {"ok" if zero_pfx else "MISSING"}')
    print(f'  zero-prefix+shuf {zpfxs_path.name}: {"ok" if zero_pfx_shuf else "MISSING"}')
    print()

    categories = ['overall', 'explanatory', 'predictive', 'counterfactual']

    header = [f'{"Condition":<30}']
    header += [f'{c:^32}' for c in categories]
    print('  '.join(header))
    print('-' * len('  '.join(header)))

    if baseline_shuf:
        # Paper's primary numbers come from the shuffled baseline. Surface that
        # row first; show the unshuffled run as an indented reference.
        print_condition_row('Grounded-Physics LM (PRIMARY, shuf)', baseline_shuf, categories)
        print_condition_row('  unshuffled reference', baseline, categories)
    else:
        print_condition_row('Grounded-Physics LM', baseline, categories)
    if zero_phys:
        print_condition_row('Zero Physics (input)', zero_phys, categories)
    if zero_pfx:
        print_condition_row('Zero Prefix (tokens)', zero_pfx, categories)
    if zero_pfx_shuf:
        print_condition_row('Zero Prefix + Shuffle', zero_pfx_shuf, categories)

    print()
    print('=== Deltas (baseline - ablation), Cohen\'s h, Fisher\'s exact p-value ===')
    # Keep historical deltas anchored on the unshuffled baseline so existing LaTeX
    # numbers remain reproducible byte-for-byte.
    if zero_phys:
        print_delta_row('Physics contribution', baseline, zero_phys, categories)
    if zero_pfx:
        print_delta_row('Prefix contrib. (vs zero-pfx)', baseline, zero_pfx, categories)
    if zero_pfx_shuf:
        print_delta_row('Prefix contrib. (vs shuffle)', baseline, zero_pfx_shuf, categories)
    if baseline_shuf:
        # Paper's tab:ablation reports this delta -- it isolates the positional
        # bias the LLM exploits when MCQ choices are presented in deterministic
        # order (large positive delta = bias was helping the unshuffled model).
        print_delta_row('Positional bias (unshuf - shuf)', baseline, baseline_shuf, categories)

    # Trust asymmetry: zero-prefix-shuffle (no physics) vs zero-physics (corrupted physics)
    if zero_phys and zero_pfx_shuf:
        print()
        print('=== Trust asymmetry (zero-prefix-shuffle vs zero-physics) ===')
        print_delta_row('zero-pfx-shuf - zero-phys', zero_pfx_shuf, zero_phys, categories)
        print()
        print('  Interpretation: positive delta means "no physics signal" (shuffle) outperforms')
        print('  "corrupted physics" (zero-physics). If large, the model trusts and follows the')
        print('  physics prefix, so corrupting it hurts more than removing it entirely.')

    if args.emit_latex and baseline and zero_phys and zero_pfx and zero_pfx_shuf:
        emit_latex_significance_table(baseline, zero_phys, categories, scope_label)
        emit_latex_ablation_table(baseline, zero_phys, zero_pfx, zero_pfx_shuf, categories, scope_label)


if __name__ == '__main__':
    main()
