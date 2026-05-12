"""Scene-summary text builder used by the per-object prompt-injection path.

The encoder OOD probe (``clevrer_benchmark/scripts/encoder_ood_probing.py``)
established that the frozen PhysicsFormer encoder destroys per-object color
(R^2 1.00 -> -0.004 input -> encoder_per_obj on CLEVRER) and that
shape/material identity is partially washed out by the adapter's mean-pool
into the LLM prefix (prefix_pool AUCs 0.65 / 0.69 vs 1.00 / 1.00 at input).
This means free-form questions about an object's surface attributes
(color, shape, material) cannot be answered from the physics prefix alone:
the information has been thrown away by the time the adapter hands a prefix
to the LLM.

The fix taken by Phase 5 / V4 training is to inject a deterministic textual
summary of the scene's surface attributes directly into the LLM prompt:

    "Scene contains: red metal cube, blue rubber sphere, green rubber
    cylinder. <question> Answer:"

That sentence is built **purely from the per-object state vector** (the
same tensor that already feeds the encoder), so it adds zero new I/O at
training or inference time and never relies on metadata files that might
diverge from the H5 records.

This module provides a single pure function, ``build_scene_summary``, that
takes a state slice + mask and produces the sentence. It is shared by:

  - ``physics_llm_adapter/adapter_v3.py::PhysicsLLMAdapterV3.compute_loss``
  - ``clevrer_benchmark/scripts/eval_phase4_freeform_qa.py``
  - ``clevrer_benchmark/run_adapter_evaluation.py``

so the training-time prompt and the inference-time prompt are guaranteed
byte-identical for the same input states.

Index conventions match
``data_generation/isaac_sim/helpers/state_extraction.py`` and
``clevrer_benchmark/scene_converter.py``:

  state[0:3]   position
  state[3:6]   linear velocity
  state[6:10]  orientation quaternion
  state[10:13] angular velocity
  state[13]    mass               <-  used as material proxy (>=1.5 = metal)
  state[14]    radius
  state[15:18] color RGB           <-  snapped to nearest CLEVRER palette
  state[18]    shape_type          <-  CLEVRER: 0/1/2 = sphere/cube/cylinder
                                       Isaac:   0..6 = sphere/box/cylinder/capsule/mesh/robot/articulation
  state[19]    is_static
  state[20]    friction
  state[21]    is_active
  ...

CLEVRER and Isaac use the same layout, so the summary code is
domain-agnostic.

This module is deliberately torch- and numpy-agnostic: it accepts both
``torch.Tensor`` and ``numpy.ndarray`` and returns plain Python strings.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


# ---- canonical palettes (must mirror scene_converter.COLOR_MAP) -------------

# 8 CLEVRER colors. RGB values match clevrer_benchmark/scene_converter.py
# COLOR_MAP. Probe results (encoder_ood_probing_v2_surface.json) confirmed
# that all 8 colors appear in the CLEVRER training mix.
COLOR_PALETTE = np.array([
    [0.5, 0.5, 0.5],  # 0 gray
    [1.0, 0.0, 0.0],  # 1 red
    [0.0, 0.0, 1.0],  # 2 blue
    [0.0, 1.0, 0.0],  # 3 green
    [0.6, 0.3, 0.1],  # 4 brown
    [0.5, 0.0, 0.5],  # 5 purple
    [0.0, 1.0, 1.0],  # 6 cyan
    [1.0, 1.0, 0.0],  # 7 yellow
], dtype=np.float32)
COLOR_NAMES = (
    'gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow',
)

# Shape ids. CLEVRER uses 0/1/2; Isaac uses 0..6. Both are encoded by the
# same column (state[18]) so we read both with one table; values outside
# the table fall back to ``'object'`` so the summary is never silently wrong.
SHAPE_NAMES_BY_ID = {
    0: 'sphere',
    1: 'cube',     # CLEVRER nomenclature; Isaac calls this 'box'
    2: 'cylinder',
    3: 'capsule',
    4: 'mesh',
    5: 'robot',
    6: 'articulation',
}

# Material proxy: rubber (low mass) vs metal (high mass). CLEVRER uses
# mass = 1.0 for rubber and 2.0 for metal (see scene_converter.MATERIAL_MAP),
# so the 1.5 threshold cleanly splits the two. Isaac counterfactual mass
# values vary more but the same threshold gives a sensible binary signal.
MATERIAL_THRESHOLD = 1.5
MATERIAL_LIGHT = 'rubber'
MATERIAL_HEAVY = 'metal'


# ---- internal helpers -------------------------------------------------------

def _to_numpy(x: Any) -> np.ndarray:
    """Accept torch.Tensor or numpy.ndarray; return a CPU numpy array."""
    if x is None:
        return None
    # torch.Tensor has .detach + .cpu + .numpy. Avoid importing torch here so
    # this module stays import-cheap; duck-typing via attribute presence.
    if hasattr(x, 'detach') and hasattr(x, 'cpu') and hasattr(x, 'numpy'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _classify_color(rgb: np.ndarray) -> int:
    """Snap an RGB triple to the index of the nearest palette entry."""
    dists = np.linalg.norm(COLOR_PALETTE - rgb[None, :], axis=-1)
    return int(dists.argmin())


# ---- public API -------------------------------------------------------------

def describe_object(
    state_vec: np.ndarray,
    *,
    include_color: bool = True,
    include_material: bool = True,
    shape_fallback: str = 'object',
) -> str:
    """Return a 1-3 word description of a single object from its state row.

    Examples:
      "red metal cube"
      "blue rubber sphere"
      "gray cylinder"      (when include_material=False)

    Args:
        state_vec: 1-D state row of length >= 19. Required indices: 13
            (mass), 15:18 (color RGB), 18 (shape id).
        include_color: prepend a color word.
        include_material: include a material word (rubber/metal).
        shape_fallback: word used when shape id is outside the known table
            (e.g. an unmapped Isaac mesh). Use ``'object'`` for a clean
            English fallback rather than emitting an integer.
    """
    if state_vec.shape[0] < 19:
        raise ValueError(
            f"state row too short: got dim {state_vec.shape[0]}, expected >=19"
        )
    parts: List[str] = []
    if include_color:
        rgb = state_vec[15:18].astype(np.float32)
        parts.append(COLOR_NAMES[_classify_color(rgb)])
    if include_material:
        mass = float(state_vec[13])
        parts.append(MATERIAL_HEAVY if mass >= MATERIAL_THRESHOLD
                     else MATERIAL_LIGHT)
    shape_id = int(state_vec[18])
    parts.append(SHAPE_NAMES_BY_ID.get(shape_id, shape_fallback))
    return ' '.join(parts)


def build_scene_summary(
    states: Any,
    object_mask: Optional[Any] = None,
    *,
    frame_idx: Optional[int] = None,
    include_color: bool = True,
    include_material: bool = True,
    prefix: str = 'Scene contains',
    suffix: str = '.',
    deduplicate: bool = False,
    max_objects: Optional[int] = None,
    style: str = 'comma_list',
) -> str:
    """Build a deterministic scene-summary sentence from a state tensor.

    Args:
        states: Either ``[N, 35]`` (single-frame) or ``[T, N, 35]``
            (sequence). Single-batch only -- batched scenes should call
            this once per scene.
        object_mask: Either ``[N]`` or ``[T, N]`` mask aligned with
            ``states``. ``None`` treats all objects as valid.
        frame_idx: When ``states`` is ``[T, N, 35]``, the frame index to
            read color/shape/material from. Defaults to ``T // 2`` (middle
            frame) since color/shape/mass are static across time.
        include_color: passthrough to ``describe_object``.
        include_material: passthrough to ``describe_object``.
        prefix: text to prepend, e.g. ``'Scene contains'``. Ignored for
            ``style='numbered'``, which uses a per-object prefix instead.
        suffix: text to append, typically a sentence-ending punctuation.
        deduplicate: if True, collapse repeated descriptions (e.g. two
            red rubber spheres become a single token). Default False
            because object identity matters for causal QA ("the red sphere
            that hit the cube"). Forced False when ``style='numbered'``
            since the whole point of numbering is to distinguish duplicates.
        max_objects: optional cap on the number of objects mentioned. If
            the scene has more, the rest are silently dropped. Default
            ``None`` = no cap.
        style: one of ``'comma_list'``, ``'numbered'``, or
            ``'numbered_no_descriptors'``:
            - ``'comma_list'`` (default):
              ``"Scene contains: red metal cube, blue rubber sphere."``
            - ``'numbered'``:
              ``"Object 1: red metal cube. Object 2: blue rubber sphere."``
              Numbering makes pronoun/reference questions
              ("the second sphere", "object 2") unambiguous at the token
              level, which matters for causal free-form QA where the
              LLM has to bind surface attributes to specific objects in
              the prefix. See comment in ``adapter_v3.py`` for why this
              helps on top of plain text injection.
            - ``'numbered_no_descriptors'`` (Phase 8):
              ``"Object 1. Object 2. Object 3. Object 4."`` -- slot count
              and ordering preserved, but color/material/shape words
              stripped. Used as the per-sample stripped variant during
              training (drawn with probability ``descriptor_strip_prob``
              in ``adapter_v3.PhysicsLLMAdapterV3._build_scene_texts``)
              so the LoRA cannot answer descriptor-naming questions by
              copying from the prompt and must read the prefix slots
              instead. ``include_color`` / ``include_material`` are
              ignored for this style (it is descriptor-free by
              definition).

    Returns:
        A sentence like
        ``"Scene contains: red metal cube, blue rubber sphere, gray rubber cylinder."``
        or ``"Scene contains: no objects."`` when the mask is all zeros.
    """
    states_np = _to_numpy(states)
    mask_np = _to_numpy(object_mask) if object_mask is not None else None

    if style not in ('comma_list', 'numbered', 'numbered_no_descriptors'):
        raise ValueError(
            f"style must be 'comma_list', 'numbered', or "
            f"'numbered_no_descriptors', got {style!r}"
        )

    if states_np.ndim == 3:
        T = states_np.shape[0]
        f = frame_idx if frame_idx is not None else T // 2
        f = max(0, min(T - 1, f))
        states_t = states_np[f]                      # [N, 35]
        if mask_np is not None and mask_np.ndim == 2:
            mask_t = mask_np[f]
        else:
            mask_t = mask_np
    elif states_np.ndim == 2:
        states_t = states_np
        mask_t = mask_np
    else:
        raise ValueError(
            f"states must be 2D [N, 35] or 3D [T, N, 35], got shape "
            f"{states_np.shape}"
        )

    N = states_t.shape[0]
    if mask_t is None:
        valid = np.ones(N, dtype=bool)
    else:
        valid = mask_t > 0.5
    valid_idxs = np.nonzero(valid)[0]
    if max_objects is not None:
        valid_idxs = valid_idxs[:max_objects]

    if len(valid_idxs) == 0:
        if style == 'numbered' or style == 'numbered_no_descriptors':
            return f"No objects{suffix}"
        return f"{prefix}: no objects{suffix}"

    # Phase 8 fast path: 'numbered_no_descriptors' emits ``Object N.`` per
    # real slot without ever calling ``describe_object``. Slot count and
    # ordering match the regular 'numbered' style so any per-object prefix
    # binding the LoRA may have learned still maps to the same surface
    # ordinals ("Object 1", "Object 2", ...).
    if style == 'numbered_no_descriptors':
        return ' '.join(
            f"Object {i + 1}." for i, _ in enumerate(valid_idxs)
        )

    descriptions: List[str] = []
    seen: set = set()
    for i in valid_idxs:
        d = describe_object(
            states_t[i],
            include_color=include_color,
            include_material=include_material,
        )
        if style != 'numbered' and deduplicate and d in seen:
            continue
        seen.add(d)
        descriptions.append(d)

    if style == 'numbered':
        # 1-indexed because that's how natural language refers to ordinal
        # objects ("the first sphere", "object 1"). Each object becomes
        # its own sentence so a tokenizer can cleanly attach a period.
        return ' '.join(
            f"Object {i + 1}: {d}." for i, d in enumerate(descriptions)
        )
    return f"{prefix}: {', '.join(descriptions)}{suffix}"


def build_scene_summary_batch(
    states_batch: Any,
    object_mask_batch: Optional[Any] = None,
    **kwargs: Any,
) -> List[str]:
    """Build scene summaries for a batch.

    Args:
        states_batch: ``[B, N, 35]`` or ``[B, T, N, 35]``.
        object_mask_batch: ``[B, N]`` or ``[B, T, N]`` aligned with batch.
        kwargs: forwarded to ``build_scene_summary``.
    """
    states_np = _to_numpy(states_batch)
    mask_np = (_to_numpy(object_mask_batch)
               if object_mask_batch is not None else None)

    if states_np.ndim not in (3, 4):
        raise ValueError(
            f"states_batch must be 3D [B, N, 35] or 4D [B, T, N, 35], "
            f"got shape {states_np.shape}"
        )

    summaries: List[str] = []
    B = states_np.shape[0]
    for b in range(B):
        s_b = states_np[b]
        m_b = mask_np[b] if mask_np is not None else None
        summaries.append(build_scene_summary(s_b, m_b, **kwargs))
    return summaries


__all__ = [
    'COLOR_PALETTE',
    'COLOR_NAMES',
    'SHAPE_NAMES_BY_ID',
    'MATERIAL_THRESHOLD',
    'MATERIAL_LIGHT',
    'MATERIAL_HEAVY',
    'describe_object',
    'build_scene_summary',
    'build_scene_summary_batch',
]
