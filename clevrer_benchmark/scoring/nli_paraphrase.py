"""Bidirectional-entailment NLI scoring + flip-rule evaluator.

Two layers of API live here:

  - Per-prediction online scoring (``nli_setup`` + ``paraphrase_score`` +
    ``nli_correct``) -- used by ``free_form_transfer_test.py`` and
    ``free_form_prefix_ablation.py`` during a generation loop where each
    prediction is graded one at a time.

  - Batched offline scoring (``build_pairs`` + ``run_nli_batched`` +
    ``evaluate_flips``) -- used by ``paraphrase_audit.py`` to grade an
    entire details.jsonl in one pass. Faster when the prediction count
    is in the thousands because batches fill the GPU.

Both layers use the same paraphrase definition: a prediction is a
paraphrase of a choice iff ``min(P(pred entails choice),
P(choice entails pred)) >= threshold``. Bidirectionality is essential
because one-sided entailment lets ``"the cylinder collides with the
sphere"`` count as a paraphrase of any sentence that *implies* a
collision, e.g. ``"the cylinder destroys the sphere"`` -- only the
mutual-entailment min collapses that to a real surface-form match.

Imports of ``transformers`` / ``torch`` are deferred to the call sites
so a script that imports this module just to use ``norm`` doesn't pay
the GPU-init cost. ``--help`` stays snappy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Default model picked to match the existing paper-grade audits exactly.
# 184MB DeBERTa-v3 NLI checkpoint, trained on MNLI+FEVER+ANLI. Strong on
# CLEVRER's short imperative sentences and runs comfortably on a single
# 16GB GPU at batch_size=32.
DEFAULT_NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'


# ---------------------------------------------------------------------------
# Online API: load once, score per prediction.
# ---------------------------------------------------------------------------

def nli_setup(model_id: str = DEFAULT_NLI_MODEL,
              device: str = 'cuda') -> Tuple[Any, Any, int]:
    """Load tokenizer + model and locate the entailment label index.

    Returns a ``(tokenizer, model, ent_idx)`` tuple. ``ent_idx`` is the
    softmax index whose label starts with ``'entail'`` (different MNLI
    checkpoints use different label orderings; we look it up rather
    than hardcode 2).

    Equivalent to ``_nli_setup`` in ``free_form_transfer_test.py``.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f'[nli] loading {model_id} on {device}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    ent_idx = next(i for i, lab in id2label.items() if lab.startswith('entail'))
    return tokenizer, model, ent_idx


def paraphrase_score(tokenizer,
                     model,
                     ent_idx: int,
                     premise: str,
                     hypothesis: str,
                     device: str) -> float:
    """Bidirectional-entailment paraphrase score: ``min(P->H, H->P)``.

    Equivalent to ``_nli_paraphrase`` in ``free_form_transfer_test.py``.
    Returns 0.0 if either string is empty (NLI on empty strings is
    meaningless -- we want a hard zero rather than whatever the model
    happens to predict on an empty sequence).
    """
    import torch
    if not premise.strip() or not hypothesis.strip():
        return 0.0
    pairs = [(premise, hypothesis), (hypothesis, premise)]
    enc = tokenizer([p for p, _ in pairs], [h for _, h in pairs],
                    return_tensors='pt', truncation=True, padding=True,
                    max_length=128).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, ent_idx]
    return float(min(probs[0].item(), probs[1].item()))


def nli_correct(tokenizer,
                model,
                ent_idx: int,
                pred: str,
                correct_choices: Sequence[str],
                device: str,
                threshold: float = 0.7) -> bool:
    """True iff pred bidirectionally entails ANY correct choice above
    ``threshold``. Standard threshold is 0.7 for the paper's primary runs.

    Equivalent to ``_nli_correct`` in ``free_form_transfer_test.py``.
    """
    return any(
        paraphrase_score(tokenizer, model, ent_idx, pred, c, device) >= threshold
        for c in correct_choices
    )


def nli_correct_with_score(tokenizer,
                           model,
                           ent_idx: int,
                           pred: str,
                           correct_choices: Sequence[str],
                           device: str,
                           threshold: float = 0.7) -> Tuple[bool, float]:
    """Like :func:`nli_correct` but also returns the max paraphrase score
    across all correct choices. Useful when the caller wants to threshold-
    sweep the same prediction at multiple cutoffs without re-running the
    LLM (the score is what costs; the threshold check is free).
    """
    if not correct_choices:
        return False, 0.0
    scores = [paraphrase_score(tokenizer, model, ent_idx, pred, c, device)
              for c in correct_choices]
    best = max(scores)
    return (best >= threshold), best


# ---------------------------------------------------------------------------
# Batched offline API: paraphrase_audit pattern.
# ---------------------------------------------------------------------------

def build_pairs(records: List[Dict],
                skip_verbatim_wrong: bool = False) -> Tuple[List[Tuple[str, str]],
                                                             List[Dict]]:
    """Build the flat ``(premise, hypothesis)`` pair list for batched NLI.

    For every record, we emit pairs in BOTH directions for every correct
    and every wrong choice; bidirectional entailment is collapsed later
    in ``evaluate_flips``. The parallel ``meta`` list lets the caller
    map each NLI score back to its (record_index, direction, choice_index,
    correct/wrong) origin.

    The records are mutated in place to add ``_audit_bucket`` and
    ``_audit_skipped`` fields so downstream stages can read what we did.

    Equivalent to ``_build_pairs`` in ``paraphrase_audit.py``. The
    ``norm`` + ``bucket_wrong_pred`` calls are imported from
    ``text_match`` rather than duplicated.
    """
    from .text_match import norm, bucket_wrong_pred
    pairs: List[Tuple[str, str]] = []
    meta: List[Dict] = []
    for ri, r in enumerate(records):
        pred = norm(r.get('predicted', ''))
        correct = [norm(c['choice']) for c in r['choices'] if c['answer'] == 'correct']
        wrong = [norm(c['choice']) for c in r['choices'] if c['answer'] == 'wrong']
        bucket = bucket_wrong_pred(pred, correct, wrong)
        if skip_verbatim_wrong and bucket == 'verbatim_wrong':
            r['_audit_bucket'] = bucket
            r['_audit_skipped'] = True
            continue
        r['_audit_bucket'] = bucket
        r['_audit_skipped'] = False
        for ci, choice in enumerate(correct):
            pairs.append((pred, choice))
            meta.append({'ri': ri, 'dir': 'PtoC', 'ci': ci, 'label': 'correct'})
            pairs.append((choice, pred))
            meta.append({'ri': ri, 'dir': 'CtoP', 'ci': ci, 'label': 'correct'})
        for ci, choice in enumerate(wrong):
            pairs.append((pred, choice))
            meta.append({'ri': ri, 'dir': 'PtoC', 'ci': ci, 'label': 'wrong'})
            pairs.append((choice, pred))
            meta.append({'ri': ri, 'dir': 'CtoP', 'ci': ci, 'label': 'wrong'})
    return pairs, meta


def run_nli_batched(pairs: List[Tuple[str, str]],
                    model_id: str = DEFAULT_NLI_MODEL,
                    device: str = 'cuda',
                    batch_size: int = 32) -> List[float]:
    """Score ``pairs`` in fixed-size GPU batches, returning P(entailment).

    Equivalent to ``_run_nli`` in ``paraphrase_audit.py``. Loads the model
    fresh per call (cheap on a hot HF cache, ~3s on a warm disk).
    """
    if not pairs:
        return []

    import torch
    from tqdm.auto import tqdm

    tokenizer, model, ent_idx = nli_setup(model_id, device)

    scores: List[float] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(pairs), batch_size),
                          desc='[nli] batches', unit='batch'):
            batch = pairs[start:start + batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]
            enc = tokenizer(premises, hypotheses,
                            return_tensors='pt', truncation=True, padding=True,
                            max_length=128).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            scores.extend(probs[:, ent_idx].cpu().tolist())
    return scores


def evaluate_flips(records: List[Dict],
                   meta: List[Dict],
                   scores: List[float],
                   threshold: float = 0.7,
                   margin: float = 0.05) -> Tuple[int, List[Dict]]:
    """Apply the paraphrase-audit flip rule.

    A wrong record flips to correct iff:
      - its max bidirectional paraphrase score against ANY correct choice
        is at least ``threshold``, AND
      - that score exceeds the max-against-any-wrong-choice by at least
        ``margin``.

    Bidirectional entailment is ``min(PtoC, CtoP)`` so a flip requires
    BOTH directions to clear ``threshold`` -- equivalent to mutual
    entailment, not one-sided implication. The margin guards against
    cases where the prediction is equally close to a correct AND a
    wrong choice, which would be ambiguous.

    Returns ``(num_flips, flipped_records)``. The records are mutated
    in place to record ``_audit_max_correct_paraphrase`` /
    ``_audit_max_wrong_paraphrase`` / ``_audit_max_correct_dirs`` for
    downstream inspection.

    Equivalent to ``_evaluate_flips`` in ``paraphrase_audit.py``. The
    signature drops the unused ``pairs`` parameter (the original took
    it but only used ``meta`` + ``scores``).
    """
    by_pair: Dict[Tuple[int, str, int], Dict[str, float]] = {}
    for m, s in zip(meta, scores):
        key = (m['ri'], m['label'], m['ci'])
        by_pair.setdefault(key, {})[m['dir']] = s

    flips: List[Dict] = []
    for ri, r in enumerate(records):
        if r.get('_audit_skipped'):
            continue
        max_correct = -1.0
        max_correct_dirs: Dict[str, float] = {}
        max_wrong = -1.0
        for (rri, lab, ci), dirs in by_pair.items():
            if rri != ri:
                continue
            paraphrase = min(dirs.get('PtoC', 0.0), dirs.get('CtoP', 0.0))
            if lab == 'correct' and paraphrase > max_correct:
                max_correct = paraphrase
                max_correct_dirs = dict(dirs)
            elif lab == 'wrong' and paraphrase > max_wrong:
                max_wrong = paraphrase
        r['_audit_max_correct_paraphrase'] = max_correct
        r['_audit_max_wrong_paraphrase'] = max_wrong
        r['_audit_max_correct_dirs'] = max_correct_dirs

        if max_correct >= threshold and max_correct > max_wrong + margin:
            flips.append(r)

    return len(flips), flips


# Backwards-compat aliases for refactored callers that prefer the
# original underscore-prefixed names.
_nli_setup = nli_setup
_nli_paraphrase = paraphrase_score
_nli_correct = nli_correct
_run_nli = run_nli_batched
_build_pairs = build_pairs
_evaluate_flips = evaluate_flips
