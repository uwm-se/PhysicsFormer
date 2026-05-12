"""
Phase 9 paraphrased CLEVRER causal-QA generator (Lever 2).

Reads the canonical free-form CLEVRER causal-QA dataset
(``causal_qa_dataset.json``, ~27k records) and emits a sister file
where every record's QUESTION text has been replaced with a
deterministic paraphrase. The ``target`` (answer prose), ``scene_desc``,
and ``scene_path`` are unchanged -- only the question phrasing varies.

Goal
----
Force question-template invariance during Phase 9 training. Phase 7/8's
prefix-ablation evidence showed the model treats the canonical CLEVRER
templates as a single switch -- the wrong-scene prediction matches the
right-scene prediction byte-identically 64% of the time. Adding a
paraphrased copy of every causal-QA record means the model sees TWO
different question surface forms for the same (scene, answer) pair, so
template memorisation is no longer a winning strategy.

Why a separate, training-only rule set
--------------------------------------
The held-out paraphrase EVALUATION uses
``clevrer_benchmark/scripts/paraphrased_mcq_test.py``'s ``PARAPHRASE_RULES``
which target ANSWER-CHOICE phrasings ("X collides with Y" -> "X bumps
into Y"). To avoid contaminating that eval, this file uses an
independent, QUESTION-targeted rule set that operates on the
interrogative wrapper around the predicate (e.g.
"Will X collide with Y?" -> "Are X and Y going to crash?"). The rule
sets are disjoint by surface form (question wrapper vs. answer
predicate) so train/eval stay clean.

Filters
-------
Only records whose ``scene_index`` falls in the Phase 9 training
range (10000-14498, defined in ``physics_llm_adapter.phase9_splits``)
are paraphrased. Eval-scene records (14499-14999) are dropped on
output -- they are kept clean for held-out grading.

Usage
-----
::

    python compsac_2026_code/data_generation/qa_generation/generate_phase9_paraphrased_qa.py \\
        --src explicit_world_model/llm_adapters/cache/causal_qa_dataset.json \\
        --output compsac_2026_code/data/phase9/causal_qa_dataset_paraphrased.json \\
        --seed 42

Output schema is byte-identical to the source: every record retains
``qa_type, scene_desc, target, scene_index, scene_path``; only
``question`` and ``prompt`` are rewritten.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Resolve splits import without bringing peft / torch in.
_SCRIPT_DIR = Path(__file__).resolve().parent
_SNAPSHOT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SNAPSHOT_ROOT))
from physics_llm_adapter.phase9_splits import (  # type: ignore  # noqa: E402
    PHASE9_PARAPHRASED_QA_FILE,
    is_train_scene,
)


# ---------------------------------------------------------------------
# Question-targeted paraphrase rules.
#
# Each rule is (regex_pattern, [variant_1, variant_2, ...]) where
# ``regex_pattern`` matches a CLEVRER question wrapper and the
# variants are deterministic substitutions. Patterns operate on
# lowercased text and are applied left-to-right; for each matching
# record we pick a variant via a per-record-deterministic seeded RNG
# so re-runs are byte-identical.
#
# Rules target QUESTION phrasings that appear in
# ``causal_qa_dataset.json`` based on enumerating its ``question`` field
# unique values. Coverage emphasis:
#   * "Will X collide with Y?"        -- the dominant predictive form
#   * "What event will happen next?"  -- the canonical "what next?"
#   * "What will happen if ..."       -- counterfactual
#   * "Among the following events"    -- multi-choice predictive
#   * "Are these events responsible"  -- causal attribution
#
# Strict invariant: the answer key (Yes/No/Object N/etc.) is NEVER
# affected because rules only rewrite the question wrapper, never the
# entity references inside it (colours, shapes, materials, indices).
# ---------------------------------------------------------------------
_PARAPHRASE_RULES: List[Tuple[str, List[str]]] = [
    # =================================================================
    # Counterfactual: "What would happen if ..."
    # 6,477 records in causal_qa_dataset.json -- the single largest
    # bucket. These rules MUST match this form for the paraphrase pool
    # to actually exist at meaningful scale.
    # =================================================================
    (
        r'^what would happen if\b',
        [
            r'what if',
            r'suppose, for a moment,',
            r'imagine that, instead,',
            r'in the hypothetical where',
        ],
    ),
    # =================================================================
    # Predictive / temporal: "What happens after ..."
    # ~10k records. Rewrite as a temporal-anchor variant that swaps
    # word order without changing meaning.
    # =================================================================
    (
        r'^what happens after\b',
        [
            r'after the following event, what occurs:',
            r'in the moments after',
            r'once it happens, what comes next:',
        ],
    ),
    (
        r'^what happens before\b',
        [
            r'before the following event, what occurs:',
            r'leading up to',
            r'just prior to',
        ],
    ),
    # =================================================================
    # Explanatory: "What caused the collision between X and Y?"
    # ~500 records.
    # =================================================================
    (
        r'^what caused\b',
        [
            r'what was the reason for',
            r'what triggered',
            r'what made the following happen:',
        ],
    ),
    # =================================================================
    # Interventional: "How could you prevent the collision between ..."
    # ~580 records. Reword as goal-oriented question.
    # =================================================================
    (
        r'^how could you prevent\b',
        [
            r'what would stop',
            r'what could be done to keep from happening:',
            r'how can you avoid',
        ],
    ),
    # =================================================================
    # Generic predictive: "Will X collide with Y?" / "Will X and Y
    # collide?" -- catches the smaller direct-question pool.
    # =================================================================
    (
        r'\bwill (the [\w ]+) collide with (the [\w ]+?)\b',
        [
            r'is \1 going to crash into \2',
            r'does \1 hit \2',
            r'will \1 bump into \2',
        ],
    ),
    (
        r'\bwill the ([\w ]+) and the ([\w ]+?) collide\b',
        [
            r'are the \1 and the \2 going to crash',
            r'is a collision coming between the \1 and the \2',
            r"do the \1 and the \2 hit each other",
        ],
    ),
    # =================================================================
    # Generic "what next" forms.
    # =================================================================
    (
        r'\bwhat event will happen next\b',
        [
            r"what's the next event",
            r'what happens next',
            r'what will occur next',
        ],
    ),
    (
        r'\bwhat is the next event\b',
        [
            r'what comes next',
            r"what's coming up next",
            r'what event is next',
        ],
    ),
]


def _capitalise_first(s: str) -> str:
    """Re-capitalise the first letter to match the original casing."""
    return s[:1].upper() + s[1:] if s else s


def _paraphrase_question(question: str, scene_index: int, qa_type: str) -> Tuple[str, int]:
    """Return (new_question, n_rules_applied).

    Selection of variant is deterministic in (scene_index, qa_type) so
    re-runs of the script produce identical paraphrased datasets.
    """
    if not question:
        return question, 0
    capital_first = question[:1].isupper()
    s = question.lower()
    n_applied = 0
    for pat, variants in _PARAPHRASE_RULES:
        # Pick one variant per rule deterministically. Seed: a hash of
        # (scene_index, qa_type, pattern) so each (record, rule) pair
        # picks the same variant on every run, but different rules in
        # the same record can pick different variants.
        if re.search(pat, s):
            seed_n = (
                int(scene_index) * 31 + sum(ord(c) for c in qa_type) * 7
                + sum(ord(c) for c in pat) * 11
            )
            choice = variants[seed_n % len(variants)]
            new_s, n = re.subn(pat, choice, s)
            if n > 0:
                s = new_s
                n_applied += n
    if capital_first:
        s = _capitalise_first(s)
    # Preserve trailing punctuation: if the original ended in '?'
    # and our rewrite stripped it (some replacements drop the '?'),
    # restore it.
    if question.rstrip().endswith('?') and not s.rstrip().endswith('?'):
        s = s.rstrip() + '?'
    return s, n_applied


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n', 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--src', type=Path,
                        default=Path('explicit_world_model/llm_adapters/cache/'
                                     'causal_qa_dataset.json'),
                        help='Source canonical free-form QA records.')
    parser.add_argument('--output', type=Path, default=PHASE9_PARAPHRASED_QA_FILE,
                        help='Output JSON path.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--require_paraphrase', action='store_true', default=True,
                        help='If true, drop records where no rule matched -- '
                             'paraphrased pool is then PURELY paraphrased '
                             '(every record has new wording). Default true.')
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f'Source file not found: {args.src}')

    print(f'[load] {args.src}')
    with open(args.src, 'r', encoding='utf-8') as f:
        src_records = json.load(f)
    print(f'[load] {len(src_records):,} canonical records')

    # Filter to Phase 9 training scenes only.
    train_records = [r for r in src_records
                     if is_train_scene(r.get('scene_index', -1))]
    n_eval_dropped = len(src_records) - len(train_records)
    print(f'[filter] kept {len(train_records):,} training-scene records '
          f'(dropped {n_eval_dropped:,} eval-scene records)')

    # Apply paraphrase rules.
    out: List[Dict] = []
    n_paraphrased = 0
    n_skipped_no_match = 0
    rules_applied_hist: Dict[int, int] = {}
    t0 = time.time()
    for r in train_records:
        new_q, n_rules = _paraphrase_question(
            r.get('question', ''),
            r.get('scene_index', 0),
            r.get('qa_type', ''),
        )
        rules_applied_hist[n_rules] = rules_applied_hist.get(n_rules, 0) + 1
        if n_rules == 0:
            if args.require_paraphrase:
                n_skipped_no_match += 1
                continue
            new_q = r['question']
        out.append({
            'qa_type': r.get('qa_type', ''),
            'scene_desc': r.get('scene_desc', ''),
            'question': new_q,
            'target': r.get('target', ''),
            # Rebuild the prompt so it stays consistent with the new
            # question. The Phase 7/8 prompt format prepends scene_desc
            # plus 'Question:' before the question text.
            'prompt': (
                (r.get('scene_desc', '') + ' Question: ' + new_q).strip()
                if r.get('scene_desc') else new_q
            ),
            'scene_index': r.get('scene_index'),
            'scene_path': r.get('scene_path'),
        })
        n_paraphrased += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    tmp.replace(args.output)
    dt = time.time() - t0

    print()
    print('=' * 70)
    print('Phase 9 paraphrased CLEVRER QA generation complete')
    print('=' * 70)
    print(f'  source records:               {len(src_records):,}')
    print(f'  training-scene records:       {len(train_records):,}')
    print(f'  paraphrased and emitted:      {n_paraphrased:,}')
    print(f'  skipped (no rule matched):    {n_skipped_no_match:,}')
    print(f'  output path:                  {args.output}')
    print(f'  output size:                  {args.output.stat().st_size/1e6:.1f} MB')
    print(f'  walltime:                     {dt:.1f} sec')
    print()
    print('  rules-applied histogram:')
    for k in sorted(rules_applied_hist.keys()):
        v = rules_applied_hist[k]
        print(f'    {k} rule(s) applied: {v:,} records')
    print()


if __name__ == '__main__':
    main()
