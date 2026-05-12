"""Schema converters: Phase 9 eval-JSON <-> legacy details.jsonl records.

The Phase 3-7 audit pipeline (``paraphrase_audit.py``,
``semantic_equivalence_audit.py``) reads the canonical
``details.jsonl`` format that ``run_adapter_evaluation.py`` writes:

  {
    'scene_id':       str,                     # e.g. 'annotation_14514'
    'question_id':    int|str,
    'question_text':  str,
    'predicted':      str,
    'correct':        bool,                    # strict-substring score
    'choices':        [{'choice': str, 'answer': 'correct'|'wrong'}],
    'correct_choices': [str],
    'wrong_choices':   [str],
    'clevrer_type':   str,                     # e.g. 'counterfactual'
    ...
  }

Phase 9's three CLEVRER eval JSONs use richer per-record nesting:

  - ``eval_paraphrase_n200.json``       : per-question 4 paraphrase tiers
                                          nested under ``tiers.t0..t3``.
  - ``eval_prefix_ablation_n200.json``  : per-question 3 prefix conditions
                                          nested under ``real`` / ``zero``
                                          / ``wrong_scene``.
  - ``eval_free_form_transfer_n200.json`` : per-question 2 prompt formats
                                          nested under ``mcq`` / ``free_form``.

To rescore these with the existing audits, we flatten each nested
condition into its own legacy record (so a 200-question paraphrase eval
becomes 800 legacy records, one per (question, tier)). The flattened
record keeps a ``_phase9_meta`` field with the source condition label
so a downstream aggregator can roll the results back up by condition.

The held-out-type eval (``eval_heldout_type_n200.json``) doesn't ship a
choice list -- its gold answers are categorical. It's graded by
``categorical_match`` directly and does NOT go through this converter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_choices(correct: List[str], wrong: List[str]) -> List[Dict[str, str]]:
    """Build the legacy ``choices`` list-of-dicts from the Phase 9 split
    correct / wrong lists. Order is correct-first, then wrong; both audits
    are order-agnostic.
    """
    out: List[Dict[str, str]] = []
    for c in correct:
        out.append({'choice': c, 'answer': 'correct'})
    for w in wrong:
        out.append({'choice': w, 'answer': 'wrong'})
    return out


def _legacy_record(scene_id: str,
                   question_id,
                   question_text: str,
                   clevrer_type: str,
                   predicted: str,
                   correct_strict: bool,
                   correct_choices: List[str],
                   wrong_choices: List[str],
                   meta: Dict) -> Dict:
    """Assemble a single legacy details.jsonl record."""
    return {
        'scene_id': scene_id,
        'question_id': question_id,
        'question_text': question_text,
        'clevrer_type': clevrer_type,
        'predicted': predicted,
        'correct': bool(correct_strict),
        'choices': _build_choices(correct_choices, wrong_choices),
        'correct_choices': list(correct_choices),
        'wrong_choices': list(wrong_choices),
        '_phase9_meta': meta,
    }


# ---------------------------------------------------------------------------
# Per-eval converters (yield legacy records).
# ---------------------------------------------------------------------------

def iter_legacy_records_paraphrase(records: Iterable[Dict]) -> Iterator[Dict]:
    """Flatten ``eval_paraphrase_n200.json`` records.

    For each Phase 9 record (one CLEVRER question), emit FOUR legacy
    records, one per paraphrase tier (t0..t3). The choice menu changes
    per tier (control = original choices, T1-T3 = increasingly aggressive
    paraphrase rewrites), so each emitted legacy record carries that
    tier's specific choice list as its ground-truth menu.
    """
    for r in records:
        scene_id = r.get('scene_id', '')
        q_idx = r.get('q_idx')
        qtext = r.get('question_text', '')
        qtype = r.get('question_type', '')
        tiers = r.get('tiers') or {}
        for tier_key in ('t0', 't1', 't2', 't3'):
            tier = tiers.get(tier_key) or {}
            if not tier:
                continue
            yield _legacy_record(
                scene_id=scene_id,
                question_id=f'{q_idx}_{tier_key}',
                question_text=qtext,
                clevrer_type=qtype,
                predicted=tier.get('predicted', ''),
                correct_strict=bool(tier.get('substring_correct', False)),
                correct_choices=list(tier.get('paraphrased_correct') or []),
                wrong_choices=list(tier.get('paraphrased_wrong') or []),
                meta={
                    'phase9_eval': 'paraphrase',
                    'tier': tier_key,
                    'tier_label': tier.get('tier_label', tier_key),
                    'original_correct': list(r.get('original_correct') or []),
                    'original_wrong': list(r.get('original_wrong') or []),
                    'original_template_fallback': bool(tier.get('original_template_fallback', False)),
                    'is_unknown': bool(tier.get('is_unknown', False)),
                },
            )


def iter_legacy_records_prefix_ablation(records: Iterable[Dict]) -> Iterator[Dict]:
    """Flatten ``eval_prefix_ablation_n200.json`` records.

    Emit THREE legacy records per question, one per ablation condition:
    ``real`` (real adapter prefix), ``zero`` (zero-vector prefix), and
    ``wrong_scene`` (a different scene's prefix swapped in). Choice menu
    is the same across conditions because only the prefix, not the
    question, is being ablated.
    """
    for r in records:
        scene_id = r.get('scene_id', '')
        q_idx = r.get('q_idx')
        qtext = r.get('question_text', '')
        qtype = r.get('question_type', '')
        correct_choices = list(r.get('correct_choices') or [])
        wrong_choices = list(r.get('wrong_choices') or [])
        for cond in ('real', 'zero', 'wrong_scene'):
            data = r.get(cond) or {}
            if not data:
                continue
            yield _legacy_record(
                scene_id=scene_id,
                question_id=f'{q_idx}_{cond}',
                question_text=qtext,
                clevrer_type=qtype,
                predicted=data.get('predicted', ''),
                correct_strict=bool(data.get('substring_correct', False)),
                correct_choices=correct_choices,
                wrong_choices=wrong_choices,
                meta={
                    'phase9_eval': 'prefix_ablation',
                    'condition': cond,
                    'sub_scene_id': r.get('sub_scene_id'),
                    'prompt': r.get('prompt'),
                    'template_phrasing': bool(data.get('template_phrasing', False)),
                    'nli_correct_legacy': bool(data.get('nli_correct', False)),
                },
            )


def iter_legacy_records_free_form_transfer(records: Iterable[Dict]) -> Iterator[Dict]:
    """Flatten ``eval_free_form_transfer_n200.json`` records.

    Emit TWO legacy records per question, one per prompt format
    (``mcq`` = with-choice-menu, ``free_form`` = no choices). Same
    choice menu across both since the menu is the question's gold
    truth, not part of the test treatment.
    """
    for r in records:
        scene_id = r.get('scene_id', '')
        q_idx = r.get('q_idx')
        qtext = r.get('question_text', '')
        qtype = r.get('question_type', '')
        correct_choices = list(r.get('correct_choices') or [])
        wrong_choices = list(r.get('wrong_choices') or [])
        for fmt in ('mcq', 'free_form'):
            data = r.get(fmt) or {}
            if not data:
                continue
            yield _legacy_record(
                scene_id=scene_id,
                question_id=f'{q_idx}_{fmt}',
                question_text=qtext,
                clevrer_type=qtype,
                predicted=data.get('predicted', ''),
                correct_strict=bool(data.get('substring_correct', False)),
                correct_choices=correct_choices,
                wrong_choices=wrong_choices,
                meta={
                    'phase9_eval': 'free_form_transfer',
                    'prompt_format': fmt,
                    'prompt': data.get('prompt'),
                    'bucket': data.get('bucket'),
                    'looks_clevrer_template': bool(data.get('looks_clevrer_template', False)),
                    'nli_correct_legacy': bool(data.get('nli_correct', False)),
                },
            )


# ---------------------------------------------------------------------------
# Top-level convenience entrypoints.
# ---------------------------------------------------------------------------

# Map from Phase 9 eval source-name -> flattening function. Used by the
# rescore driver to route an unknown JSON to the right converter without
# hard-coding filenames.
PHASE9_FLATTENERS = {
    'paraphrase':         iter_legacy_records_paraphrase,
    'prefix_ablation':    iter_legacy_records_prefix_ablation,
    'free_form_transfer': iter_legacy_records_free_form_transfer,
}


def detect_phase9_eval_kind(payload: Dict) -> Optional[str]:
    """Best-effort detection of which Phase 9 eval a JSON payload came
    from, based on the record schema. Returns one of:
      'paraphrase' | 'prefix_ablation' | 'free_form_transfer' |
      'heldout_type' | None.

    The four Phase 9 eval JSONs are distinguishable by their first
    record's top-level keys -- there's no overlap.
    """
    records = payload.get('records') or []
    if not records:
        return None
    keys = set(records[0].keys())
    if 'tiers' in keys:
        return 'paraphrase'
    if 'real' in keys and 'zero' in keys and 'wrong_scene' in keys:
        return 'prefix_ablation'
    if 'mcq' in keys and 'free_form' in keys:
        return 'free_form_transfer'
    if 'qa_type' in keys and 'gold_answer' in keys:
        return 'heldout_type'
    return None


def write_legacy_jsonl(records: Iterable[Dict], out_path: Path) -> int:
    """Stream ``records`` to ``out_path`` in details.jsonl format.

    Returns the number of records written. Atomic: writes to a .tmp
    sidecar first, then rename, so a kill mid-flush leaves either the
    old file or the new one (never half-written).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    n = 0
    with open(tmp, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + '\n')
            n += 1
    tmp.replace(out_path)
    return n


def convert_phase9_to_legacy_jsonl(in_path: Path, out_path: Path) -> Tuple[str, int]:
    """End-to-end: read a Phase 9 eval JSON, detect its kind, flatten,
    and write a legacy JSONL.

    Returns ``(eval_kind, n_records_written)``. Raises ValueError if the
    eval kind can't be detected or has no flattener (e.g. the held-out-
    type eval, which uses a categorical grader instead).
    """
    with open(in_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    kind = detect_phase9_eval_kind(payload)
    if kind is None:
        raise ValueError(f'Unrecognised Phase 9 eval JSON: {in_path}')
    if kind not in PHASE9_FLATTENERS:
        raise ValueError(
            f"Phase 9 eval '{kind}' is not legacy-JSONL convertible (use "
            f"the categorical_match grader directly): {in_path}"
        )
    records = list(payload.get('records') or [])
    flattened = PHASE9_FLATTENERS[kind](records)
    n = write_legacy_jsonl(flattened, out_path)
    return kind, n
