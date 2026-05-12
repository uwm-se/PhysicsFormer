"""Single-command eval driver for the Phase 4 / free-form QA ablation.

Runs the full Tier 1a Phase A evaluation suite on a trained adapter
checkpoint and emits a paper-ready comparison table against the existing
Phase 3 SOTA and Phase 4 mixed-format results.

Stages
------
1. **MCQ** -- canonical held-out MCQ accuracy via
   ``run_adapter_evaluation.py --heldout --filter_malformed --shuffle_choices
   --save_details --single_frame 64``. Phase 3 = 69.2%, Phase 4 mixed = 72.5%.
2. **Free-form transfer** -- ``free_form_transfer_test.py --heldout --n N``.
   Phase 3 = 0% NLI, 99.45% "unknown". Phase 4 mixed = 0% NLI. **Target metric.**
3. **Paraphrased MCQ** -- ``paraphrased_mcq_test.py --no_stratified --n N``
   across tiers 0-3. Phase 3 = 0.0-0.2%, fallback >88%.
4. **Encoder OOD probing** (optional) -- ``encoder_ood_probing.py``. Cheap
   sanity check that the encoder still has its CLEVRER + Isaac signal
   intact (it should, since Phase 4 only retrains the adapter, not the
   encoder).

Each stage runs the corresponding script as a subprocess; if the output
file already exists the stage is skipped (override with
``--no_skip_existing``). After all stages, the script reads each output
JSON and prints a multi-column comparison table.

Usage
-----
::

    # On the host that has the trained checkpoint and CLEVRER on disk:
    python compsac_2026_code/clevrer_benchmark/scripts/eval_phase4_freeform_qa.py \\
        --adapter_checkpoint compsac_2026_code/checkpoints/adapter_phase4_freeform_qa_best.pt \\
        --label freeform_ablation \\
        --n 1998

The script writes to
``compsac_2026_code/clevrer_benchmark/results/{label}/{label}_{STAGE}.json``
mirroring the Phase 4 mixed-format directory layout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Snapshot-portable defaults.
_HERE = Path(__file__).resolve().parent
_BENCH_DIR = _HERE.parent
_SNAPSHOT_ROOT = _BENCH_DIR.parent
_RESULTS_DIR = _BENCH_DIR / 'results'

# Reuse compute_paper_stats helpers for MCQ filtering + Wilson CI.
sys.path.insert(0, str(_HERE))
from compute_paper_stats import (  # noqa: E402
    load_result, load_heldout_scenes, wilson_ci, fisher_pvalue, format_pvalue,
)

DEFAULT_CHECKPOINT = (
    _SNAPSHOT_ROOT / 'checkpoints' / 'adapter_phase4_freeform_qa_best.pt'
)
DEFAULT_PHYSICS_CKPT = 'D:\\physics-former-data\\checkpoints\\stage1_best.pt'
DEFAULT_CLEVRER_DIR = 'D:\\clevrer'
DEFAULT_H5 = os.environ.get(
    'CLEVRER_H5',
    os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5'),
)
DEFAULT_LABEL = 'freeform_ablation'

# Existing baselines pulled from disk for the comparison table. These
# files are written by prior eval runs (Phase 3 + Phase 4 mixed-format)
# and are NOT regenerated here.
PHASE3_MCQ_REF      = _RESULTS_DIR / 'phase3_BASELINE_SHUFFLE_FULL5000.json'
PHASE3_FREEFORM_REF = _RESULTS_DIR / 'free_form_transfer_full_heldout.json'
PHASE3_PARAPH_REF   = _RESULTS_DIR / 'paraphrased_mcq_test_full_heldout.json'

PHASE4_MCQ_REF      = _RESULTS_DIR / 'phase4' / 'phase4_BASELINE_SHUFFLE_HELDOUT.json'
PHASE4_FREEFORM_REF = _RESULTS_DIR / 'phase4' / 'phase4_FREEFORM_HELDOUT.json'
PHASE4_PARAPH_REF   = _RESULTS_DIR / 'phase4' / 'phase4_PARAPHRASED_HELDOUT.json'
PHASE4_OOD_REF      = _RESULTS_DIR / 'phase4' / 'phase4_ENCODER_PROBE.json'


# ---------------------------------------------------------------------------
# Stage runners (subprocess.run wrappers)
# ---------------------------------------------------------------------------


def _run_step(name: str, cmd: List[str], output_path: Path,
              skip_existing: bool, dry_run: bool = False) -> bool:
    """Run a stage subprocess. Returns True if the stage produced (or already had) output."""
    if skip_existing and output_path.exists():
        size_kb = output_path.stat().st_size / 1e3
        print(f"  [SKIP] {name}: output exists at {output_path} ({size_kb:.1f} KB)")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [RUN]  {name}")
    print(f"         cmd: {' '.join(str(c) for c in cmd)}")
    if dry_run:
        print(f"         (dry-run; not executing)")
        return False

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_SNAPSHOT_ROOT))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  [FAIL] {name}: exit {proc.returncode} after {elapsed:.1f}s")
        return False
    if not output_path.exists():
        print(f"  [WARN] {name}: subprocess succeeded but no output at {output_path}")
        return False
    size_kb = output_path.stat().st_size / 1e3
    print(f"  [OK]   {name}: wrote {output_path} ({size_kb:.1f} KB) in {elapsed:.1f}s")
    return True


def run_mcq(args, out_dir: Path) -> Optional[Path]:
    """Held-out MCQ eval. Output JSON has overall + by_clevrer_type accuracy."""
    out_path = out_dir / f'{args.label}_BASELINE_SHUFFLE_HELDOUT.json'
    cmd = [
        sys.executable,
        str(_BENCH_DIR / 'run_adapter_evaluation.py'),
        '--clevrer_dir', args.clevrer_dir,
        '--adapter_checkpoint', str(args.adapter_checkpoint),
        '--physics_checkpoint', args.physics_checkpoint,
        '--max_scenes', '99999',
        '--single_frame', '64',
        '--heldout',
        '--filter_malformed',
        '--shuffle_choices',
        '--save_details',
        '--skip_descriptive',
        '--output', str(out_path),
        '--h5', args.h5,
    ]
    ok = _run_step('MCQ (heldout, shuffled)', cmd, out_path,
                   args.skip_existing, args.dry_run)
    return out_path if ok else None


def run_freeform(args, out_dir: Path) -> Optional[Path]:
    """Free-form-vs-MCQ transfer. Output JSON has summary.{mcq,free_form,gap}."""
    out_path = out_dir / f'{args.label}_FREEFORM_HELDOUT.json'
    cmd = [
        sys.executable,
        str(_HERE / 'free_form_transfer_test.py'),
        '--clevrer_dir', args.clevrer_dir,
        '--adapter_checkpoint', str(args.adapter_checkpoint),
        '--n', str(args.n),
        '--heldout',
        '--out', str(out_path),
    ]
    ok = _run_step('Free-form transfer', cmd, out_path,
                   args.skip_existing, args.dry_run)
    return out_path if ok else None


def run_paraphrased(args, out_dir: Path) -> Optional[Path]:
    """Paraphrased-MCQ across tiers 0-3. Output JSON has summary[tier]={substring_acc_pct,...}."""
    out_path = out_dir / f'{args.label}_PARAPHRASED_HELDOUT.json'
    cmd = [
        sys.executable,
        str(_HERE / 'paraphrased_mcq_test.py'),
        '--clevrer_dir', args.clevrer_dir,
        '--adapter_checkpoint', str(args.adapter_checkpoint),
        '--physics_checkpoint', args.physics_checkpoint,
        '--n', str(args.n),
        '--no_stratified',
        '--out', str(out_path),
        '--h5', args.h5,
    ]
    ok = _run_step('Paraphrased MCQ', cmd, out_path,
                   args.skip_existing, args.dry_run)
    return out_path if ok else None


def run_ood(args, out_dir: Path) -> Optional[Path]:
    """Encoder OOD probing on CLEVRER + Isaac. Output JSON has results[domain][feature]."""
    out_path = out_dir / f'{args.label}_ENCODER_PROBE.json'
    cmd = [
        sys.executable,
        str(_HERE / 'encoder_ood_probing.py'),
        '--adapter_ckpt', str(args.adapter_checkpoint),
        '--physics_ckpt', args.physics_checkpoint,
        '--n_per_domain', str(args.ood_n),
        '--out', str(out_path),
    ]
    ok = _run_step('Encoder OOD probing', cmd, out_path,
                   args.skip_existing, args.dry_run)
    return out_path if ok else None


# ---------------------------------------------------------------------------
# Summary loaders
# ---------------------------------------------------------------------------


def _safe_load(path: Optional[Path]) -> Optional[Dict]:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  [WARN] could not parse {path}: {e}")
        return None


def summary_mcq(path: Optional[Path],
                heldout: Optional[set] = None,
                valid_only: bool = True) -> Dict[str, Tuple[int, int, float]]:
    """Read an MCQ result file and return per-category (correct, total, acc).

    Uses ``load_result`` from ``compute_paper_stats`` for files that have
    a ``.details.jsonl`` sidecar (the canonical heldout-filterable shape).
    Falls back to the simpler {overall, by_clevrer_type} layout otherwise.
    """
    if path is None or not path.exists():
        return {}

    details = path.with_suffix('.details.jsonl')
    if details.exists():
        try:
            d = load_result(path, valid_only=valid_only,
                            heldout_scenes=heldout)
            out = {}
            for cat in ('overall', 'explanatory', 'predictive', 'counterfactual'):
                if cat in d and d[cat]['total'] > 0:
                    c, t = d[cat]['correct'], d[cat]['total']
                    out[cat] = (c, t, c / t)
            return out
        except Exception as e:
            print(f"  [WARN] load_result failed on {path.name}: {e}")

    # Fallback: parse the simple summary shape directly.
    raw = _safe_load(path)
    if raw is None:
        return {}
    out = {}
    if 'overall' in raw and 'correct' in raw['overall'] and 'total_questions' in raw['overall']:
        c = raw['overall']['correct']
        t = raw['overall']['total_questions']
        if t > 0:
            out['overall'] = (c, t, c / t)
    for cat, per in raw.get('by_clevrer_type', {}).items():
        if 'correct' in per and 'total' in per and per['total'] > 0:
            c, t = per['correct'], per['total']
            out[cat] = (c, t, c / t)
    return out


def summary_freeform(path: Optional[Path]) -> Dict[str, float]:
    """Read a free-form transfer result file and return key metrics in [0, 1]."""
    raw = _safe_load(path)
    if raw is None or 'summary' not in raw:
        return {}
    s = raw['summary']
    out = {}
    if isinstance(s.get('mcq'), dict):
        out['mcq_substring']  = s['mcq'].get('substring_acc_pct',  0.0) / 100.0
        out['mcq_nli']        = s['mcq'].get('nli_acc_pct',        0.0) / 100.0
    if isinstance(s.get('free_form'), dict):
        out['ff_substring']        = s['free_form'].get('substring_acc_pct',        0.0) / 100.0
        out['ff_nli']              = s['free_form'].get('nli_acc_pct',              0.0) / 100.0
        out['ff_template_phrase']  = s['free_form'].get('clevrer_template_phrasing_pct', 0.0) / 100.0
        out['ff_choice_membership']= s['free_form'].get('clevrer_choice_membership_pct', 0.0) / 100.0
    if isinstance(s.get('gap'), dict):
        out['gap_nli']       = s['gap'].get('nli_pp',       0.0) / 100.0
        out['gap_substring'] = s['gap'].get('substring_pp', 0.0) / 100.0

    # Compute the "unknown"-emission rate from the records list (the key
    # diagnostic from the Phase 3 audit -- "99.45% unknown"). Absent in
    # the summary block, present implicitly via prediction text. The
    # canonical field is record['free_form']['predicted'] -- legacy/grounded
    # variants may put it at the record top level.
    unk_total = 0
    seen = 0
    for rec in raw.get('records', []):
        seen += 1
        ff = rec.get('free_form')
        if isinstance(ff, dict):
            pred_text = ff.get('predicted', ff.get('predicted_text', ''))
        else:
            pred_text = rec.get('free_form_pred', rec.get('predicted', ''))
        pred_text = str(pred_text).strip().lower()
        if pred_text in ('unknown', 'unknown unknown', '<unknown>', ''):
            unk_total += 1
    if seen > 0:
        out['ff_unknown_rate'] = unk_total / seen
    return out


def summary_paraphrased(path: Optional[Path]) -> Dict[int, Dict[str, float]]:
    """Read a paraphrased-MCQ result file and return per-tier metrics."""
    raw = _safe_load(path)
    if raw is None or 'summary' not in raw:
        return {}
    s = raw['summary']
    out = {}
    for k, tier_data in s.items():
        try:
            tier = int(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(tier_data, dict):
            continue
        out[tier] = {
            'substring_acc':       tier_data.get('substring_acc_pct',              0.0) / 100.0,
            'fallback_to_template':tier_data.get('original_template_fallback_pct', 0.0) / 100.0,
            'unknown_rate':        tier_data.get('unknown_pct',                    0.0) / 100.0,
            'n':                   tier_data.get('n', 0),
            'label':               tier_data.get('tier_label', f'Tier {tier}'),
        }
    return out


def summary_ood(path: Optional[Path]) -> Dict:
    """Read an encoder OOD probing file and return per-domain headline numbers."""
    raw = _safe_load(path)
    if raw is None or 'results' not in raw:
        return {}
    return raw['results']


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return '   ---'
    return f'{x*100:6.2f}%'


def _fmt_delta(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return '       '
    return f'{(a - b)*100:+6.2f}'


def _fmt_count(d: Optional[Tuple[int, int, float]]) -> str:
    if d is None:
        return '         '
    c, t, _ = d
    return f'{c:>5d}/{t:<4d}'


def print_comparison_table(p3, p4_mix, p4_free, *, label: str) -> None:
    """Multi-column comparison: Phase 3 / Phase 4 mixed / Phase 4 free-form / delta vs P3."""
    line = '=' * 92
    print()
    print(line)
    print(f'  Comparison: Phase 3 (MCQ-only)  vs  Phase 4 mixed-format  vs  {label}')
    print(line)

    # ---------- MCQ rows ----------
    print()
    print(f"  {'MCQ accuracy (heldout, shuffled)':<32}  "
          f"{'Phase 3':>9}  {'P4 mixed':>9}  {'P4 ' + label[:6]:>9}  {'Δ vs P3':>9}")
    print('  ' + '-' * 80)
    for cat, label_str in [('overall', 'Overall'), ('explanatory', 'Explanatory'),
                            ('predictive', 'Predictive'), ('counterfactual', 'Counterfactual')]:
        p3v = p3['mcq'].get(cat)
        p4mv = p4_mix['mcq'].get(cat)
        p4fv = p4_free['mcq'].get(cat)
        a3 = p3v[2] if p3v else None
        a4m = p4mv[2] if p4mv else None
        a4f = p4fv[2] if p4fv else None
        print(f"  {label_str:<32}  {_fmt_pct(a3):>9}  {_fmt_pct(a4m):>9}  "
              f"{_fmt_pct(a4f):>9}  {_fmt_delta(a4f, a3):>9}")

    # ---------- Free-form rows (the key story) ----------
    print()
    print(f"  {'Free-form (no choices in prompt)':<32}  "
          f"{'Phase 3':>9}  {'P4 mixed':>9}  {'P4 ' + label[:6]:>9}  {'Δ vs P3':>9}")
    print('  ' + '-' * 80)
    for key, label_str in [
        ('ff_nli',                'NLI accuracy'),
        ('ff_substring',          'Substring accuracy'),
        ('ff_template_phrase',    'CLEVRER template phrasing %'),
        ('ff_choice_membership',  'Choice-text membership %'),
        ('ff_unknown_rate',       '"unknown" rate'),
    ]:
        a3 = p3['ff'].get(key)
        a4m = p4_mix['ff'].get(key)
        a4f = p4_free['ff'].get(key)
        print(f"  {label_str:<32}  {_fmt_pct(a3):>9}  {_fmt_pct(a4m):>9}  "
              f"{_fmt_pct(a4f):>9}  {_fmt_delta(a4f, a3):>9}")

    # ---------- Paraphrased-MCQ rows ----------
    print()
    print(f"  {'Paraphrased MCQ (substring acc)':<32}  "
          f"{'Phase 3':>9}  {'P4 mixed':>9}  {'P4 ' + label[:6]:>9}  {'Δ vs P3':>9}")
    print('  ' + '-' * 80)
    for tier in (0, 1, 2, 3):
        p3t = p3['paraph'].get(tier, {})
        p4mt = p4_mix['paraph'].get(tier, {})
        p4ft = p4_free['paraph'].get(tier, {})
        a3 = p3t.get('substring_acc')
        a4m = p4mt.get('substring_acc')
        a4f = p4ft.get('substring_acc')
        tier_label = (p4ft.get('label') or p3t.get('label') or f'Tier {tier}')
        print(f"  Tier {tier} {tier_label:<25}  {_fmt_pct(a3):>9}  {_fmt_pct(a4m):>9}  "
              f"{_fmt_pct(a4f):>9}  {_fmt_delta(a4f, a3):>9}")

    # ---------- Verdict ----------
    print()
    print(line)
    p3_nli = p3['ff'].get('ff_nli')
    p4f_nli = p4_free['ff'].get('ff_nli')
    p4f_mcq_overall = p4_free['mcq'].get('overall')
    p3_mcq_overall = p3['mcq'].get('overall')

    if p4f_nli is not None and p3_nli is not None:
        delta_nli_pp = (p4f_nli - p3_nli) * 100
        print(f'  Free-form NLI Δ vs Phase 3:  {delta_nli_pp:+.2f} pp')
        if p4f_nli >= 0.10:
            print(f'  [HYPOTHESIS 2 SUPPORT] Free-form NLI {p4f_nli*100:.1f}% >> 10% --')
            print(f'   prose-target training restored open-ended physics description.')
        elif p4f_nli >= 0.02:
            print(f'  [PARTIAL] Free-form NLI {p4f_nli*100:.1f}% > 2% but < 10% --')
            print(f'   some recovery, but not enough to claim hypothesis 2.')
        else:
            print(f'  [HYPOTHESIS 2 REJECTED] Free-form NLI {p4f_nli*100:.1f}% --')
            print(f'   prose-target training did NOT lift free-form generation.')
            print(f'   Bottleneck is deeper than supervisory-signal mismatch.')

    if p4f_mcq_overall is not None and p3_mcq_overall is not None:
        delta_mcq_pp = (p4f_mcq_overall[2] - p3_mcq_overall[2]) * 100
        if p4f_mcq_overall[2] < p3_mcq_overall[2] - 0.02:
            print(f'  [WARN] MCQ regressed {delta_mcq_pp:+.2f} pp vs Phase 3 -- '
                  f'free-form training cost MCQ ability.')
        else:
            print(f'  [OK] MCQ Δ vs Phase 3:  {delta_mcq_pp:+.2f} pp '
                  f'(within ±2pp gate; free-form training did not erase MCQ).')
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Single-command eval driver for the Phase 4 free-form QA ablation.',
    )
    parser.add_argument('--adapter_checkpoint', type=Path, default=DEFAULT_CHECKPOINT,
                        help='Path to the trained free-form ablation checkpoint '
                             '(adapter_phase4_freeform_qa_best.pt).')
    parser.add_argument('--physics_checkpoint', type=str, default=DEFAULT_PHYSICS_CKPT,
                        help='Path to the Stage-1 physics encoder checkpoint.')
    parser.add_argument('--clevrer_dir', type=str, default=DEFAULT_CLEVRER_DIR,
                        help='Path to the CLEVRER dataset directory.')
    parser.add_argument('--h5', type=str, default=DEFAULT_H5,
                        help='Path to the CLEVRER training H5 (used by --heldout pool).')
    parser.add_argument('--label', type=str, default=DEFAULT_LABEL,
                        help='Output filename prefix (default: freeform_ablation).')
    parser.add_argument('--out_dir', type=Path, default=None,
                        help='Output directory (default: results/{label}).')
    parser.add_argument('--n', type=int, default=1998,
                        help='MCQ / free-form / paraphrased pool size (default 1998 = '
                             'full heldout valid-only paper-primary pool).')
    parser.add_argument('--ood_n', type=int, default=2000,
                        help='Per-domain n for encoder OOD probing (default 2000).')

    parser.add_argument('--skip_mcq',         action='store_true', help='Skip MCQ stage.')
    parser.add_argument('--skip_freeform',    action='store_true', help='Skip free-form stage.')
    parser.add_argument('--skip_paraphrased', action='store_true', help='Skip paraphrased stage.')
    parser.add_argument('--skip_ood',         action='store_true', default=True,
                        help='Skip encoder OOD probe (default on -- the encoder is unchanged '
                             'across Phase 3/4 ablations, so re-running rarely tells you '
                             'anything new). Pass --no_skip_ood to force.')
    parser.add_argument('--no_skip_ood', action='store_false', dest='skip_ood',
                        help='Run the encoder OOD probe even though the encoder is unchanged.')

    parser.add_argument('--no_skip_existing', action='store_false', dest='skip_existing',
                        default=True,
                        help='Force re-run even if output JSONs already exist.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print the subprocess commands but do not execute.')
    args = parser.parse_args()

    # Resolve all user-supplied input paths to absolute paths BEFORE passing
    # them to subprocesses. subprocess.run(cmd, cwd=_SNAPSHOT_ROOT) below would
    # otherwise double-prefix any relative path (e.g. the user passing
    # ``compsac_2026_code/checkpoints/foo.pt`` from the repo root would resolve
    # to ``compsac_2026_code/compsac_2026_code/checkpoints/foo.pt`` inside the
    # subprocess, breaking the MCQ / free-form / paraphrased stages).
    args.adapter_checkpoint = Path(args.adapter_checkpoint).resolve()
    args.physics_checkpoint = str(Path(args.physics_checkpoint).resolve())
    args.clevrer_dir = str(Path(args.clevrer_dir).resolve())
    args.h5 = str(Path(args.h5).resolve())

    out_dir = args.out_dir or (_RESULTS_DIR / args.label)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 92)
    print(f'  eval_phase4_freeform_qa: {args.label}')
    print('=' * 92)
    print(f'  Adapter: {args.adapter_checkpoint}')
    print(f'  Output:  {out_dir}')
    print(f'  Pool n:  {args.n} (paper-primary heldout valid-only)')
    print(f'  OOD n:   {args.ood_n} per domain')
    print()
    print('-' * 92)
    print('  Stages')
    print('-' * 92)

    new_paths: Dict[str, Optional[Path]] = {
        'mcq':      None,
        'ff':       None,
        'paraph':   None,
        'ood':      None,
    }
    if not args.skip_mcq:
        new_paths['mcq'] = run_mcq(args, out_dir)
    if not args.skip_freeform:
        new_paths['ff'] = run_freeform(args, out_dir)
    if not args.skip_paraphrased:
        new_paths['paraph'] = run_paraphrased(args, out_dir)
    if not args.skip_ood:
        new_paths['ood'] = run_ood(args, out_dir)

    if args.dry_run:
        print('\n  [dry-run] Skipping aggregation; no JSONs to read.')
        return

    # Load heldout filter for MCQ (matches paper primary).
    heldout_set = None
    try:
        heldout_set = load_heldout_scenes(Path(args.h5))
    except Exception as e:
        print(f'  [WARN] could not load heldout scene set: {e}')
        print(f'         MCQ deltas will use unfiltered counts.')

    # Build the three baselines.
    p3 = {
        'mcq':    summary_mcq(PHASE3_MCQ_REF, heldout=heldout_set, valid_only=True),
        'ff':     summary_freeform(PHASE3_FREEFORM_REF),
        'paraph': summary_paraphrased(PHASE3_PARAPH_REF),
    }
    p4_mix = {
        'mcq':    summary_mcq(PHASE4_MCQ_REF, heldout=heldout_set, valid_only=True),
        'ff':     summary_freeform(PHASE4_FREEFORM_REF),
        'paraph': summary_paraphrased(PHASE4_PARAPH_REF),
    }
    p4_free = {
        'mcq':    summary_mcq(new_paths['mcq'], heldout=heldout_set, valid_only=True),
        'ff':     summary_freeform(new_paths['ff']),
        'paraph': summary_paraphrased(new_paths['paraph']),
    }

    print_comparison_table(p3, p4_mix, p4_free, label=args.label)

    # Persist the aggregated table as JSON for the paper pipeline.
    table_path = out_dir / f'{args.label}_COMPARISON_TABLE.json'
    table_path.write_text(json.dumps({
        'label': args.label,
        'adapter_checkpoint': str(args.adapter_checkpoint),
        'phase3': p3,
        'phase4_mixed_format': p4_mix,
        f'phase4_{args.label}': p4_free,
    }, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))
    print(f'\n  Aggregated table: {table_path}')


if __name__ == '__main__':
    main()
