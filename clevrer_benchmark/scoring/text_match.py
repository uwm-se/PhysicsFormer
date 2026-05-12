"""Surface-form scoring helpers: substring match, verbatim-choice bucketing,
CLEVRER-template detection, paraphrase-fallback detection.

These are the lightest-weight grade rules in the suite -- no transformers,
no scene IO, just lowercase + substring. They are the strict-match floor
that every other rubric (NLI paraphrase, referent equivalence, categorical
synonym) must exceed to be worth running.

The functions here are byte-for-byte ports of the originals in
``free_form_transfer_test.py``, ``paraphrased_mcq_test.py``,
``paraphrase_audit.py``, and ``run_adapter_evaluation.evaluate_answer``.
Where the originals had small naming differences (``_norm`` vs
``_substring_correct``) we keep the behaviour identical and standardise
on the public name. The original underscore-prefixed names are still
available as aliases (``_norm = norm`` etc.) so a refactored script can
do ``from clevrer_benchmark.scoring.text_match import _norm`` without
churn.
"""

from __future__ import annotations

from typing import List, Sequence


def norm(s: str) -> str:
    """Lowercase + strip; the canonical text-prep for every scorer here.

    Identical to ``_norm`` in every script that defines its own copy.
    """
    return str(s).strip().lower()


def substring_correct(pred: str,
                      correct_choices: Sequence[str],
                      wrong_choices: Sequence[str]) -> bool:
    """Strict substring rule used by ``run_adapter_evaluation.evaluate_answer``.

    Returns True iff:
      - pred matches at least one correct choice (equality, pred in correct,
        or correct in pred), AND
      - pred does NOT match any wrong choice under the same rule.

    The asymmetry (any-match-correct AND no-match-wrong) is deliberate:
    a prediction that substring-matches both a correct and a wrong choice
    is ambiguous and should not be credited.
    """
    pred_n = norm(pred)
    if not pred_n:
        return False
    correct_n = [norm(c) for c in correct_choices]
    wrong_n = [norm(w) for w in wrong_choices]
    matches_correct = any((pred_n == c) or (pred_n in c) or (c in pred_n) for c in correct_n)
    matches_wrong = any((pred_n == w) or (pred_n in w) or (w in pred_n) for w in wrong_n)
    return matches_correct and not matches_wrong


def verbatim_choice_match(pred: str,
                          correct_choices: Sequence[str],
                          wrong_choices: Sequence[str]) -> str:
    """Bucket a prediction by whether it exact-matches a choice.

    Buckets:
      - ``'verbatim_correct'`` : pred (after norm) equals a labeled correct.
      - ``'verbatim_wrong'``   : pred (after norm) equals a labeled wrong.
      - ``'free_form'``        : neither (the model emitted novel text).

    Same behaviour as ``_verbatim_choice_match`` in
    ``free_form_transfer_test.py``.
    """
    p = norm(pred)
    if any(p == norm(c) for c in correct_choices):
        return 'verbatim_correct'
    if any(p == norm(w) for w in wrong_choices):
        return 'verbatim_wrong'
    return 'free_form'


# Canonical CLEVRER MCQ template stems. Matched as substrings (after
# ``norm``); a hit means the generation parrots CLEVRER phrasing.
_CLEVRER_TEMPLATE_STEMS = (
    'collides with', 'collide', 'and the ', 'the collision between',
    "'s entrance", "'s entering", "the presence of", "exits", "enters",
)


def looks_clevrer_template(text: str) -> bool:
    """Heuristic: does the generation use canonical CLEVRER MCQ phrasing?

    True if any of the canonical template stems appears in the lowercased
    text. The signal is "the model is still emitting choice-style text
    even when no choice menu was shown".

    Same behaviour as ``_looks_clevrer_template`` in
    ``free_form_transfer_test.py``.
    """
    t = norm(text)
    if not t:
        return False
    return any(s in t for s in _CLEVRER_TEMPLATE_STEMS)


def detects_original_template_fallback(pred: str,
                                       original_correct: Sequence[str],
                                       original_wrong: Sequence[str],
                                       paraphrased_correct: Sequence[str],
                                       paraphrased_wrong: Sequence[str]) -> bool:
    """Smoking-gun detector for paraphrase-test template memorisation.

    Returns True iff the prediction substring-matches a member of the
    *original* CLEVRER choice set but does NOT match any member of the
    *paraphrased* choice set that was actually offered. That's direct
    evidence the model is regenerating its memorised training-template
    rather than selecting from the offered paraphrased menu.

    Same behaviour as ``_detects_original_template_fallback`` in
    ``paraphrased_mcq_test.py``.
    """
    p = norm(pred)
    if not p:
        return False
    orig_all = [norm(c) for c in (list(original_correct) + list(original_wrong))]
    para_all = [norm(c) for c in (list(paraphrased_correct) + list(paraphrased_wrong))]
    matches_original = any((p == o) or (p in o) or (o in p) for o in orig_all)
    matches_paraphrased = any((p == pa) or (p in pa) or (pa in p) for pa in para_all)
    return matches_original and not matches_paraphrased


def bucket_wrong_pred(pred: str,
                      correct_choices: Sequence[str],
                      wrong_choices: Sequence[str]) -> str:
    """Classify a wrong prediction by its relation to the choice set.

    Buckets (priority order):
      - ``'BUG_exact_correct_yet_flagged_wrong'`` : pred exact-matches a
        labeled correct but the run was flagged wrong upstream. Indicates
        a bug in the upstream grader, not a real wrong.
      - ``'verbatim_wrong'`` : pred exact-matches a labeled wrong choice.
      - ``'substr_correct_only'`` : pred substring-matches a correct
        choice and not a wrong one (would be flipped by a substring
        rescorer).
      - ``'substr_wrong'`` : pred substring-matches a wrong choice
        (paraphrase audit must distinguish this from free_form).
      - ``'free_form'`` : pred is novel text, no substring overlap.

    Same behaviour as ``_bucket`` in ``paraphrase_audit.py``. Inputs are
    expected to be already ``norm``-ed by the caller; we match the
    original signature exactly.
    """
    if any(pred == c for c in correct_choices):
        return 'BUG_exact_correct_yet_flagged_wrong'
    if any(pred == w for w in wrong_choices):
        return 'verbatim_wrong'
    substr_w = any((pred in w or w in pred) for w in wrong_choices)
    substr_c = any((pred in c or c in pred) for c in correct_choices)
    if substr_c and not substr_w:
        return 'substr_correct_only'
    if substr_w:
        return 'substr_wrong'
    return 'free_form'


# Backwards-compat aliases. Existing scripts can do
# ``from clevrer_benchmark.scoring.text_match import _norm`` to keep the
# original underscore-prefixed name in the calling code.
_norm = norm
_substring_correct = substring_correct
_verbatim_choice_match = verbatim_choice_match
_looks_clevrer_template = looks_clevrer_template
_detects_original_template_fallback = detects_original_template_fallback
_bucket = bucket_wrong_pred
