"""Scene-aware referent-set equivalence scorer for CLEVRER MCQ predictions.

The substring rule rejects predictions like ``"the blue object's colliding
with the metal cylinder"`` when the labeled correct choice is ``"the blue
cylinder's colliding with the metal cylinder"`` -- but in scenes where
exactly one blue object exists and that object is the blue cylinder, the
two phrases unambiguously reference the same physical event on the same
physical entities.

This module implements the conservative reclaim rule used by
``semantic_equivalence_audit.py``:

    Reclaim a wrong record iff, after loading the actual CLEVRER scene's
    object inventory, the prediction's parsed event template + entity
    descriptors uniquely identify the same physical entities as at least
    one labeled CORRECT choice and do NOT identify the same entities as
    any labeled WRONG choice.

The reclaim is intentionally conservative:
  - Different event kinds (collision vs presence vs entrance/exit) never
    match.
  - A descriptor that resolves to two-or-more scene objects is too
    ambiguous to reclaim (referent-set size > 1 fails).
  - A referent-set match that ALSO matches a wrong choice is "blocked"
    by ambiguity in this scene and is not reclaimed.

The functions here are byte-for-byte ports of the originals in
``semantic_equivalence_audit.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple


# CLEVRER's closed surface vocabulary. These are the only descriptor
# tokens a CLEVRER-trained model is supposed to emit for object reference.
COLORS = {'gray', 'red', 'blue', 'green', 'brown', 'cyan', 'purple', 'yellow'}
MATERIALS = {'rubber', 'metal'}
SHAPES = {'sphere', 'cube', 'cylinder'}
GENERIC_SHAPE = 'object'  # wildcard -- matches any specific shape


# ---------------------------------------------------------------------------
# Event-template parsing
# ---------------------------------------------------------------------------

# Each entry is (regex, event_kind, [group_names_for_object_phrases]).
# event_kind is one of: 'collision', 'presence', 'entrance', 'exit'.
# Order matters: more specific patterns must precede more generic ones.
EVENT_PATTERNS: List[Tuple[re.Pattern, str, List[str]]] = [
    # Collision -- multiple surface forms (symmetric in object pair).
    (re.compile(r"^the ([^']+?)'s colliding with the (.+?)$"), 'collision', ['o1', 'o2']),
    (re.compile(r'^the (.+?) collides with the (.+?)$'),       'collision', ['o1', 'o2']),
    (re.compile(r'^the (.+?) and the (.+?) collide$'),         'collision', ['o1', 'o2']),
    (re.compile(r'^the collision between the (.+?) and the (.+?)$'), 'collision', ['o1', 'o2']),
    # Presence / state.
    (re.compile(r'^the presence of the (.+?)$'),               'presence',  ['o1']),
    # Scene entrance / exit.
    (re.compile(r"^the (.+?)'s entering(?: the scene)?$"),     'entrance',  ['o1']),
    (re.compile(r"^the (.+?)'s entrance$"),                    'entrance',  ['o1']),
    (re.compile(r'^the (.+?) enters the scene$'),              'entrance',  ['o1']),
    (re.compile(r"^the (.+?)'s exiting(?: the scene)?$"),      'exit',      ['o1']),
    (re.compile(r"^the (.+?)'s exit$"),                        'exit',      ['o1']),
    (re.compile(r'^the (.+?) exits the scene$'),               'exit',      ['o1']),
]


def parse_event(text: str) -> Optional[Tuple[str, List[str]]]:
    """Return ``(event_kind, [object_phrase_strings])`` or None if no
    template matches. Inputs are stripped + lowercased before matching.
    """
    t = (text or '').strip().lower()
    if not t:
        return None
    for regex, kind, groups in EVENT_PATTERNS:
        m = regex.match(t)
        if m:
            objs = [m.group(i + 1).strip() for i in range(len(groups))]
            return kind, objs
    return None


def parse_descriptor(phrase: str) -> Dict[str, Optional[str]]:
    """Parse ``'blue rubber object'`` -> ``{'color': 'blue', 'material':
    'rubber', 'shape': None}``.

    ``'object'`` is treated as a wildcard shape (None) so it can match
    any specific shape. Specific shape words (cube/sphere/cylinder)
    become hard shape constraints. Tokens outside the CLEVRER closed
    vocabulary are silently ignored.
    """
    desc: Dict[str, Optional[str]] = {'color': None, 'material': None, 'shape': None}
    for tok in (phrase or '').lower().split():
        tok = tok.strip(",.;:'\"")
        if tok in COLORS:
            desc['color'] = tok
        elif tok in MATERIALS:
            desc['material'] = tok
        elif tok in SHAPES:
            desc['shape'] = tok
        elif tok == GENERIC_SHAPE:
            # Wildcard shape -- leave as None.
            pass
    return desc


# ---------------------------------------------------------------------------
# Scene-object referent resolution
# ---------------------------------------------------------------------------

def matches_scene_object(desc: Dict[str, Optional[str]], obj: Dict[str, str]) -> bool:
    """True iff ``obj`` satisfies every non-None field in ``desc``."""
    if desc['color'] is not None and obj.get('color') != desc['color']:
        return False
    if desc['material'] is not None and obj.get('material') != desc['material']:
        return False
    if desc['shape'] is not None and obj.get('shape') != desc['shape']:
        return False
    return True


def referent_set(desc: Dict[str, Optional[str]],
                 scene_objs: List[Dict[str, str]]) -> FrozenSet[int]:
    """Return the set of scene-object indices the descriptor resolves to."""
    return frozenset(i for i, obj in enumerate(scene_objs) if matches_scene_object(desc, obj))


def is_semantic_match(pred_text: str,
                      choice_text: str,
                      scene_objs: List[Dict[str, str]]) -> bool:
    """True iff pred_text and choice_text refer to the same physical
    event on the same physical entities given the scene_objs inventory.

    Logic:
      - Both must parse to the SAME event kind.
      - Each object phrase on each side must resolve to a NON-EMPTY
        referent set (a descriptor that doesn't match anything in this
        scene can't be reclaimed).
      - For single-entity events the entity referent-set must match.
      - For collision events the object pair is treated as a SET
        (collisions are symmetric: ``X collides with Y`` ==
        ``Y collides with X``).
      - Degenerate ``X collides with X`` parses on either side abort
        the match (a collision needs two distinct entities).

    Same behaviour as ``is_semantic_match`` in
    ``semantic_equivalence_audit.py``.
    """
    pred_p = parse_event(pred_text)
    chc_p = parse_event(choice_text)
    if pred_p is None or chc_p is None:
        return False
    pred_kind, pred_objs = pred_p
    chc_kind, chc_objs = chc_p
    if pred_kind != chc_kind:
        return False

    pred_sets = [referent_set(parse_descriptor(o), scene_objs) for o in pred_objs]
    chc_sets = [referent_set(parse_descriptor(o), scene_objs) for o in chc_objs]
    if any(len(s) == 0 for s in pred_sets) or any(len(s) == 0 for s in chc_sets):
        return False

    if len(pred_objs) == 1:
        return pred_sets[0] == chc_sets[0]

    # Collision -- two-object symmetric.
    p1, p2 = pred_sets
    c1, c2 = chc_sets
    if p1 == p2 or c1 == c2:
        return False
    return (p1 == c1 and p2 == c2) or (p1 == c2 and p2 == c1)


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def scene_id_to_num(scene_id: str) -> Optional[int]:
    """Extract the integer scene number from a CLEVRER scene_id string.

    Handles both ``'annotation_NNNNN'`` and ``'sim_NNNNN'`` prefixes,
    plus the bare numeric suffix.
    """
    try:
        return int(scene_id.replace('annotation_', '').replace('sim_', '').split('_')[-1])
    except (ValueError, IndexError):
        return None


def _candidate_scene_paths(clevrer_dir: Path, scene_num: int) -> List[Path]:
    """Return file paths to try for a given scene number.

    CLEVRER ships scene metadata in several locations:
      - ``annotations/validation/annotation_<lo>-<hi>/annotation_NNNNN.json``
        (canonical CLEVRER annotation format, bucket-nested by 1000s --
        object_property + motion_trajectory + collision)
      - ``scenes/clevrer_scenes/sim_NNNNN.json``  (short format)
      - ``scenes/validation/sim_NNNNN.json``      (short format)
      - ``scenes/sim_NNNNN.json``                 (flat)

    The bucket-nested annotation form is the most common in real CLEVRER
    distributions; we try that first.
    """
    n5 = f'{scene_num:05d}'
    bucket_lo = (scene_num // 1000) * 1000
    bucket_hi = bucket_lo + 1000
    bucket_dir = f'annotation_{bucket_lo}-{bucket_hi}'
    return [
        clevrer_dir / 'annotations' / 'validation' / bucket_dir / f'annotation_{n5}.json',
        clevrer_dir / 'annotations' / bucket_dir / f'annotation_{n5}.json',
        clevrer_dir / 'annotations' / 'validation' / f'annotation_{n5}.json',
        clevrer_dir / 'annotations' / f'annotation_{n5}.json',
        clevrer_dir / 'scenes' / 'validation' / f'sim_{n5}.json',
        clevrer_dir / 'scenes' / 'clevrer_scenes' / f'sim_{n5}.json',
        clevrer_dir / 'scenes' / f'sim_{n5}.json',
    ]


def scene_object_inventory(scene: Dict) -> List[Dict[str, str]]:
    """Extract a list of ``{'color', 'shape', 'material'}`` dicts from
    either CLEVRER scene format (``objects`` array OR ``object_property``
    array). Missing fields default to empty string.
    """
    if 'objects' in scene and isinstance(scene['objects'], list):
        return [
            {
                'color': str(o.get('color', '')).lower(),
                'shape': str(o.get('shape', '')).lower(),
                'material': str(o.get('material', '')).lower(),
            }
            for o in scene['objects']
        ]
    if 'object_property' in scene:
        return [
            {
                'color': str(o.get('color', '')).lower(),
                'shape': str(o.get('shape', '')).lower(),
                'material': str(o.get('material', '')).lower(),
            }
            for o in scene['object_property']
        ]
    return []


# Module-level cache: scene_num -> inventory. Persists for the lifetime
# of a script invocation; keeps re-walks of the same scene cheap.
_SCENE_CACHE: Dict[int, Optional[List[Dict[str, str]]]] = {}


def load_scene_objects(clevrer_dir: Path,
                       scene_num: int) -> Optional[List[Dict[str, str]]]:
    """Resolve and load the scene-object inventory for a scene number.

    Tries every path in ``_candidate_scene_paths`` and returns the first
    inventory that successfully decodes. Caches the result (including a
    None for "not found"). Returns ``None`` if the scene file can't be
    located or fails to decode.
    """
    if scene_num in _SCENE_CACHE:
        return _SCENE_CACHE[scene_num]
    for p in _candidate_scene_paths(clevrer_dir, scene_num):
        if p.exists():
            try:
                with open(p, 'r') as f:
                    scene = json.load(f)
                inventory = scene_object_inventory(scene)
                _SCENE_CACHE[scene_num] = inventory
                return inventory
            except (json.JSONDecodeError, OSError):
                continue
    _SCENE_CACHE[scene_num] = None
    return None


# Backwards-compat aliases for the refactored callers.
_scene_id_to_num = scene_id_to_num
_scene_object_inventory = scene_object_inventory
