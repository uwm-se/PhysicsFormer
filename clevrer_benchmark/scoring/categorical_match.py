"""Lenient categorical / ordinal match for the Phase 9 held-out QUESTION TYPES.

The Phase 9 ``eval_phase9_heldout_type.py`` script grades five question
types that the adapter NEVER saw during training:

  - ``kinetic_energy``        : ordinal bucket on the energy scale
                                (``negligible`` < ``very low`` < ``low``
                                < ``moderate`` < ``high`` < ``very high``).
  - ``collision_prediction``  : binary yes/no (will any pair collide soon?).
  - ``mass_comparison``       : 1-based object index of the heaviest object.
  - ``speed_comparison``      : EITHER 1-based object index of the fastest
                                object OR an ordinal max-speed bucket
                                (``stationary`` < ``very slow`` < ``slow``
                                < ``moderate`` < ``fast`` < ``very fast``);
                                the question template determines which.
  - ``time_to_event``         : ordinal time-to-collision bucket
                                (``imminent`` < ``soon`` < ``moderate``
                                < ``long`` < ``very long``) plus a separate
                                ``never`` category.

The substring scorer alone misses paraphrases the LLM emits naturally
without ever seeing the gold label distribution:
  - ``"medium"`` instead of ``"moderate"``.
  - ``"the third object"`` instead of ``"3"``.
  - ``"will collide"`` instead of ``"yes"``.
  - ``"a long time"`` instead of ``"long"``.

This module builds two layers on top of strict substring:

1. **Synonym match** -- a per-question-type table of accepted-equivalent
   phrasings. ``"medium"`` and ``"moderate"`` count as the same bucket;
   ``"object 3"`` and ``"the third object"`` and ``"3"`` count as
   ``mass_comparison=3``.
2. **Ordinal-adjacent match** (optional) -- for ordinal-bucket types,
   credit a prediction that lands one bucket away from the gold (e.g.
   gold=``moderate``, pred=``high`` -> credited as a near-miss).

The strict substring rule is the unconditional floor: anything substring-
correct is also categorical-correct. Everything else falls back to the
synonym table and (for ordinal types) the adjacency rule.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .text_match import norm, substring_correct


# ---------------------------------------------------------------------------
# Per-question-type bucket vocabulary.
# ---------------------------------------------------------------------------
#
# Each entry is:
#   - 'kind'    : 'ordinal' | 'binary' | 'index' | 'ordinal_or_index'
#   - 'buckets' : ordered list of canonical bucket labels (low -> high
#                 for ordinals; for binary, [no, yes])
#   - 'synonyms': dict mapping each canonical bucket to a list of
#                 equivalent surface forms. The canonical bucket itself
#                 is always implicitly included.
#
# These tables are derived from ``data_generation/qa_generation/qa_generator.py``:
# every ``return "label"`` in the answer functions for a given
# QuestionType is captured here, plus the natural-language synonyms
# we want to credit at eval time.

HELDOUT_TYPE_BUCKETS: Dict[str, Dict] = {
    # qa_generator._answer_kinetic_energy_calc returns:
    #   negligible / very low / low / moderate / high / very high
    'kinetic_energy': {
        'kind': 'ordinal',
        'buckets': ['negligible', 'very low', 'low', 'moderate', 'high', 'very high'],
        'synonyms': {
            'negligible':  ['negligible', 'no kinetic energy', 'zero', 'none', 'almost zero', 'essentially zero'],
            'very low':    ['very low', 'extremely low', 'minimal', 'tiny'],
            'low':         ['low', 'small', 'little'],
            'moderate':    ['moderate', 'medium', 'mid', 'average', 'middling', 'middle'],
            'high':        ['high', 'large', 'significant', 'substantial'],
            'very high':   ['very high', 'extremely high', 'huge', 'enormous', 'massive'],
        },
    },
    # qa_generator._answer_collision returns 'Yes' / 'No';
    # _answer_will_collide_soon returns 'yes' / 'no'.
    # We accept any phrasing that semantically resolves to yes/no.
    'collision_prediction': {
        'kind': 'binary',
        'buckets': ['no', 'yes'],
        'synonyms': {
            'yes': [
                'yes', 'will collide', 'collision will occur', 'collision is imminent',
                'a collision', 'they will collide', 'collide', 'colliding', 'impact',
                'will hit', 'will impact', 'will make contact', 'will contact',
                'will crash', 'is imminent', 'imminent collision', 'collision likely',
                'true', 'yep', 'yeah', 'affirmative',
            ],
            'no': [
                'no', 'will not collide', 'no collision', 'will not occur',
                'no impact', 'will not hit', 'will not contact', 'will not make contact',
                'no contact', 'will miss', 'they will miss', 'unlikely',
                'false', 'nope', 'nah', 'negative',
            ],
        },
    },
    # qa_generator._answer_heaviest_object_num returns the 1-based index
    # ('1', '2', '3', '4'). We accept any phrasing that resolves to a
    # specific object index, e.g. "object 3", "the third one", "3rd".
    'mass_comparison': {
        'kind': 'index',
        'buckets': ['1', '2', '3', '4', '5', '6', '7', '8'],
        'synonyms': {},  # filled programmatically in _build_index_synonyms
    },
    # qa_generator has TWO answer functions for SPEED_COMPARISON:
    #   _answer_fastest_object_num -> '1', '2', '3', ...
    #   _answer_max_speed_calc      -> stationary/very slow/slow/moderate/fast/very fast
    # We disambiguate at grading time by checking which the gold matches.
    'speed_comparison': {
        'kind': 'ordinal_or_index',
        'buckets': ['stationary', 'very slow', 'slow', 'moderate', 'fast', 'very fast'],
        'synonyms': {
            'stationary':  ['stationary', 'still', 'not moving', 'at rest', 'motionless'],
            'very slow':   ['very slow', 'extremely slow', 'crawling', 'barely moving'],
            'slow':        ['slow', 'sluggish'],
            'moderate':    ['moderate', 'medium', 'mid', 'average', 'middling'],
            'fast':        ['fast', 'quick', 'rapid', 'speedy'],
            'very fast':   ['very fast', 'extremely fast', 'rapidly', 'quickly'],
        },
        'index_buckets': ['1', '2', '3', '4', '5', '6', '7', '8'],
    },
    # qa_generator._answer_time_to_collision returns:
    #   imminent / soon / moderate / long / very long
    #   plus a separate 'never' category (no approaching pairs).
    'time_to_event': {
        'kind': 'ordinal',
        'buckets': ['imminent', 'soon', 'moderate', 'long', 'very long', 'never'],
        'synonyms': {
            'imminent':    ['imminent', 'right now', 'immediately', 'instantly', 'about to', 'very soon', 'any moment'],
            'soon':        ['soon', 'shortly', 'quickly', 'in a moment', 'in a few', 'in a short time'],
            'moderate':    ['moderate', 'medium', 'mid', 'average'],
            'long':        ['long', 'a long time', 'a while', 'eventually', 'far future', 'distant'],
            'very long':   ['very long', 'extremely long', 'a very long time', 'far in the future'],
            'never':       ['never', 'will not', 'will not collide', 'no collision', 'never collide', 'no impact'],
        },
    },
}


# ---------------------------------------------------------------------------
# Object-index synonym helpers.
# ---------------------------------------------------------------------------

# Map ordinal English phrases to 1-based indices. Used by
# ``extract_object_index`` to credit predictions like "the third object".
_ORDINAL_ENGLISH = {
    '1st': 1, 'first': 1, 'one': 1, 'object 1': 1, '1':  1,
    '2nd': 2, 'second': 2, 'two': 2, 'object 2': 2, '2':  2,
    '3rd': 3, 'third': 3, 'three': 3, 'object 3': 3, '3': 3,
    '4th': 4, 'fourth': 4, 'four': 4, 'object 4': 4, '4': 4,
    '5th': 5, 'fifth': 5, 'five': 5, 'object 5': 5, '5': 5,
    '6th': 6, 'sixth': 6, 'six': 6, 'object 6': 6, '6': 6,
    '7th': 7, 'seventh': 7, 'seven': 7, 'object 7': 7, '7': 7,
    '8th': 8, 'eighth': 8, 'eight': 8, 'object 8': 8, '8': 8,
}

# Compiled regex for "object N" / "the Nth object" / bare "N" / "Object_N".
_INDEX_RE = re.compile(
    r'\b(?:object\s*[_\s]?(\d+)|'
    r'the\s+(\d+)(?:st|nd|rd|th)\s+(?:object|one)|'
    r'(\d+)(?:st|nd|rd|th)\s+(?:object|one)|'
    r'(\d+)(?:st|nd|rd|th)|'
    r'^\s*(\d+)\s*$)\b',
    re.IGNORECASE,
)


def extract_object_index(text: str, max_idx: int = 8) -> Optional[int]:
    """Extract a 1-based object index from a free-form prediction.

    Recognises:
      - Bare numbers: ``"3"``, ``"3."``, ``"3rd"``.
      - "Object N" / "object_N" / ``"object 3"``.
      - English ordinals: ``"the third object"``, ``"third one"``.
      - "the Nth (object|one)".

    Returns the matched index or None if no recognisable form is found
    or the index is out of [1, ``max_idx``].
    """
    t = norm(text)
    if not t:
        return None

    # Cheap path: bare number-only string.
    if t.isdigit():
        idx = int(t)
        return idx if 1 <= idx <= max_idx else None

    # Cheap path: English ordinal in isolation or as the leading clause.
    for phrase, idx in sorted(_ORDINAL_ENGLISH.items(), key=lambda kv: -len(kv[0])):
        # word-boundary check; cheap and avoids "fivefold" -> 5
        if re.search(rf'\b{re.escape(phrase)}\b', t):
            if 1 <= idx <= max_idx:
                return idx

    # Regex path: catch "object 3", "the 3rd one", trailing "3".
    m = _INDEX_RE.search(t)
    if m:
        for g in m.groups():
            if g and g.isdigit():
                idx = int(g)
                if 1 <= idx <= max_idx:
                    return idx
    return None


# ---------------------------------------------------------------------------
# Synonym lookup
# ---------------------------------------------------------------------------

def _bucket_for_text(qtype: str, text: str) -> Optional[str]:
    """Return the canonical bucket label that ``text`` resolves to under
    ``qtype``'s synonym table, or None.

    Match rule: the canonical bucket label or any of its synonyms must
    appear as a substring of the normalised text. Longest-synonym-first
    so ``"very low"`` is preferred over ``"low"`` when both match.
    """
    spec = HELDOUT_TYPE_BUCKETS.get(qtype)
    if spec is None:
        return None
    n = norm(text)
    if not n:
        return None
    syns = spec.get('synonyms', {})
    # Walk longest synonyms first so multi-word labels win against
    # their shorter substrings.
    candidates: List[Tuple[str, str]] = []  # (synonym, canonical)
    for canonical, alts in syns.items():
        for alt in [canonical] + list(alts):
            candidates.append((alt, canonical))
    candidates.sort(key=lambda kv: -len(kv[0]))
    for alt, canonical in candidates:
        if alt in n:
            return canonical
    return None


def _ordinal_distance(qtype: str, gold: str, pred: str) -> Optional[int]:
    """Return the ordinal distance (steps on the bucket scale) between
    ``gold`` and ``pred``, or None if either doesn't resolve.

    ``'never'`` is treated as off-scale -- distance(``never``, anything)
    is None unless they match exactly.
    """
    spec = HELDOUT_TYPE_BUCKETS.get(qtype)
    if spec is None or spec.get('kind') not in ('ordinal', 'ordinal_or_index'):
        return None
    buckets = spec['buckets']
    g = _bucket_for_text(qtype, gold)
    p = _bucket_for_text(qtype, pred)
    if g is None or p is None:
        return None
    if g == 'never' or p == 'never':
        return 0 if g == p else None
    try:
        gi = buckets.index(g)
        pi = buckets.index(p)
    except ValueError:
        return None
    return abs(gi - pi)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def categorical_correct(qtype: str,
                        pred: str,
                        gold_answers: Sequence[str],
                        wrong_answers: Sequence[str] = (),
                        ordinal_tol: int = 0) -> Tuple[bool, str]:
    """Score a held-out-type prediction with a more permissive rule than
    plain substring.

    Match precedence (first hit wins, returned in the second tuple element):

      1. ``'strict'`` -- ``substring_correct`` already credits this prediction.
      2. ``'index'`` -- ``qtype`` is index-style (mass_comparison or
         the index branch of speed_comparison) and the prediction
         resolves to the same 1-based object index as the gold.
      3. ``'synonym'`` -- the prediction's canonical-bucket label
         (via the synonym table) equals the gold's canonical-bucket
         label.
      4. ``'ordinal_adjacent'`` -- only fires when ``ordinal_tol > 0``
         AND ``qtype`` is ordinal AND the bucket distance from gold is
         <= ``ordinal_tol``.
      5. ``'wrong'`` -- none of the above matches.

    Returns ``(is_correct, reason)``. A wrong-choice list is accepted
    for symmetry with substring_correct (so a synonym hit that ALSO
    matches a labeled wrong choice is rejected for ambiguity), but
    held-out-type questions don't ship a wrong-choice menu so the
    default is empty.
    """
    if substring_correct(pred, gold_answers, wrong_answers):
        return True, 'strict'

    spec = HELDOUT_TYPE_BUCKETS.get(qtype)
    if spec is None:
        # Unknown qtype -- fall back to strict (already failed above).
        return False, 'wrong'

    kind = spec['kind']

    # --- Index-style match (mass_comparison; speed_comparison index branch).
    if kind in ('index', 'ordinal_or_index'):
        idx_buckets: Set[str] = set(spec['buckets']) if kind == 'index' else set(spec.get('index_buckets', []))
        gold_indices = {extract_object_index(g) for g in gold_answers}
        gold_indices = {i for i in gold_indices if i is not None}
        if gold_indices:
            pred_idx = extract_object_index(pred)
            wrong_indices = {extract_object_index(w) for w in wrong_answers}
            wrong_indices = {i for i in wrong_indices if i is not None}
            if pred_idx is not None and pred_idx in gold_indices and pred_idx not in wrong_indices:
                return True, 'index'

    # --- Synonym match (ordinal + ordinal_or_index + binary).
    if kind in ('ordinal', 'binary', 'ordinal_or_index'):
        gold_canon: Set[str] = set()
        for g in gold_answers:
            b = _bucket_for_text(qtype, g)
            if b is not None:
                gold_canon.add(b)
        wrong_canon: Set[str] = set()
        for w in wrong_answers:
            b = _bucket_for_text(qtype, w)
            if b is not None:
                wrong_canon.add(b)
        pred_canon = _bucket_for_text(qtype, pred)
        if pred_canon is not None and pred_canon in gold_canon and pred_canon not in wrong_canon:
            return True, 'synonym'

    # --- Ordinal adjacency (optional credit).
    if ordinal_tol > 0 and kind in ('ordinal', 'ordinal_or_index'):
        for g in gold_answers:
            d = _ordinal_distance(qtype, g, pred)
            if d is not None and 0 <= d <= ordinal_tol:
                return True, 'ordinal_adjacent'

    return False, 'wrong'
