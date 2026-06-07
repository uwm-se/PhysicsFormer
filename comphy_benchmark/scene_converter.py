"""ComPhy annotation -> 35-D state tensor (Phase 3 / FullPhysicsFormer-compatible).

This is the **Phase-3-targeted** converter. ComPhy ships annotations with a
richer schema than CLEVRER (per-frame angular velocity + orientation +
inside_scene flag, plus per-object hidden mass and charge). We project that
schema into the 35-D state tensor the FullPhysicsFormer encoder consumes
during CLEVRER training, so the same Phase 3 checkpoint can be evaluated
zero-shot.

What gets used and what is discarded
------------------------------------
* Position, velocity                    -> state[0:3], state[3:6]    (used)
* Orientation (Euler -> quaternion)     -> state[6:10]               (used)
* Angular velocity                      -> state[10:13]              (used; CLEVRER zeros this slot)
* Hidden mass (1 light / 5 heavy)       -> state[13]                 (used; overrides material default)
* Color RGB                             -> state[15:18]              (used)
* Shape type                            -> state[18]                 (used)
* Lateral friction                      -> state[20]                 (used)
* Inside-scene flag                     -> state[21] + masks         (used)
* Bbox (from shape + scale)             -> state[25:28]              (used)
* Restitution                           -> state[34]                 (used)
* Hidden charge                         -> NOT REPRESENTED           (no slot in 35-D schema)
* linear_damping, scale (raw)           -> partially represented     (folded into radius/bbox)

Note on charge: this is an honest architectural limitation. ComPhy questions
that depend on charge are answered with charge invisible to the model. The
stats script reports a charge-dependence slice via a keyword filter so the
limitation is disclosed in any reported number.

Annotation layout on disk
-------------------------
ComPhy stores annotations as ``target_annotation/annotation_<lo>_<hi>/<id>.json``
where ``<id>`` is the scene index zero-padded to 5 digits and ``<lo>_<hi>`` is
the 1000-scene chunk range. ``find_annotation_for_scene`` handles this.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from clevrer_benchmark.scene_converter import (
    COLOR_MAP,
    DEFAULT_SIZE,
    MATERIAL_MAP,
    SHAPE_MAP,
    get_object_bbox,
)


COMPHY_FPS = 25.0  # 175 simulation steps over ~7s


# ───────────────────────────────────────────────────────────────────────────
# Annotation file discovery
# ───────────────────────────────────────────────────────────────────────────

def find_annotation_for_scene(annotations_dir: str | Path,
                              scene_index: int) -> Optional[Path]:
    """Locate ``annotation_<lo>_<hi>/<scene>.json`` for a given scene_index.

    ComPhy annotations live in 1000-scene chunks:
        target_annotation/annotation_00000_01000/00042.json
    """
    ann_dir = Path(annotations_dir)
    scene_str = f"{scene_index:05d}"

    # The chunk containing scene N is the one whose name matches floor(N/1000).
    chunk_lo = (scene_index // 1000) * 1000
    chunk_hi = chunk_lo + 1000
    direct = ann_dir / f"annotation_{chunk_lo:05d}_{chunk_hi:05d}" / f"{scene_str}.json"
    if direct.exists():
        return direct

    # Fallback: scan all chunk subdirs.
    for subdir in ann_dir.iterdir():
        if not subdir.is_dir():
            continue
        candidate = subdir / f"{scene_str}.json"
        if candidate.exists():
            return candidate

    # Last-resort recursive search.
    matches = list(ann_dir.rglob(f"{scene_str}.json"))
    return matches[0] if matches else None


def load_comphy_annotation(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ───────────────────────────────────────────────────────────────────────────
# 35-D state-tensor construction
# ───────────────────────────────────────────────────────────────────────────

def _euler_to_quat_xyzw(euler_rad: List[float]) -> np.ndarray:
    """Euler angles in **radians** -> quaternion ``[qx, qy, qz, qw]``.

    ComPhy stores ``orientation`` and ``initial_orientation`` in radians (see
    sample annotations: values like ``-1.5708`` = -pi/2).
    """
    rx, ry, rz = euler_rad
    cx, sx = np.cos(rx / 2.0), np.sin(rx / 2.0)
    cy, sy = np.cos(ry / 2.0), np.sin(ry / 2.0)
    cz, sz = np.cos(rz / 2.0), np.sin(rz / 2.0)
    return np.array([
        sx * cy * cz - cx * sy * sz,  # qx
        cx * sy * cz + sx * cy * sz,  # qy
        cx * cy * sz - sx * sy * cz,  # qz
        cx * cy * cz + sx * sy * sz,  # qw
    ], dtype=np.float32)


def _construct_state_vector(
    position: List[float],
    velocity: List[float],
    angular_velocity: List[float],
    orientation_euler: List[float],
    *,
    shape: str,
    color: str,
    material: str,
    mass: float,
    scale: float,
    restitution: float,
    lateral_friction: float,
    inside_scene: bool,
    clevrer_align: bool = False,
) -> np.ndarray:
    """Build one 35-D state vector matching the FullPhysicsFormer input layout.

    See ``clevrer_benchmark.scene_converter.construct_state_vector`` for the
    canonical slot map. ComPhy fills in slots CLEVRER leaves zeroed:
      - state[6:10]  quaternion from Euler angles
      - state[10:13] angular velocity
      - state[13]    hidden mass (1.0 light / 5.0 heavy)
      - state[20]    lateral_friction (overrides material default)
      - state[34]    restitution (overrides material default)

    ``clevrer_align`` puts every channel back onto the distribution the
    CLEVRER-trained encoder actually saw (see
    ``scripts/check_state_distribution.py`` for the OOD evidence that motivates
    this). The raw ComPhy values are ~25x off on velocity (per-second vs
    CLEVRER's per-frame delta), ~20x off on friction, beyond-range on mass
    (5 vs trained max 2), and non-constant on quaternion / angular velocity
    (which CLEVRER held at identity / zero). Aligned mode:
      - quaternion  -> identity, angular velocity -> 0  (CLEVRER constants)
      - mass        -> {light:1.0, heavy:2.0}           (preserve the binary
                       distinction inside the trained range)
      - radius/bbox -> CLEVRER DEFAULT_SIZE
      - friction / restitution -> CLEVRER material defaults
      - velocity    -> KEPT as-is: ComPhy's per-second annotation already
                       matches CLEVRER's training velocity scale (CLEVRER's
                       ``derive_velocity`` = per-frame delta * fps, same order;
                       std 0.87 vs ComPhy 0.72). Only position is left raw too
                       (verified in-range: CLEVRER std 1.64 vs ComPhy 1.84).
    """
    state = np.zeros(35, dtype=np.float32)

    state[0:3] = position
    if clevrer_align:
        # ComPhy's annotated velocity is already per-second, matching CLEVRER's
        # training scale (CLEVRER derive_velocity = per-frame delta * fps ~= the
        # same magnitude; verified against clevrer_training_expanded.h5: vel std
        # 0.87 vs ComPhy 0.72). So keep it as-is. Quaternion -> identity and
        # angular velocity -> 0 to match the CLEVRER constants the encoder saw.
        state[3:6] = velocity
        state[6:10] = [0.0, 0.0, 0.0, 1.0]
        # state[10:13] (angular velocity) stays 0.
        state[13] = 1.0 if float(mass) <= 1.5 else 2.0
        state[14] = DEFAULT_SIZE
        state[20] = MATERIAL_MAP.get(material, MATERIAL_MAP['rubber'])['friction']
        state[34] = MATERIAL_MAP.get(material, MATERIAL_MAP['rubber'])['restitution']
        state[25:28] = get_object_bbox(shape, DEFAULT_SIZE)
    else:
        state[3:6] = velocity
        state[6:10] = _euler_to_quat_xyzw(orientation_euler)
        state[10:13] = angular_velocity
        state[13] = float(mass)
        # state[14] is "radius" in CLEVRER; ComPhy gives raw scale (0.2-0.5
        # range). Use scale directly so the encoder sees ground-truth size
        # rather than the CLEVRER default of 0.3.
        state[14] = float(scale) if scale and scale > 0 else DEFAULT_SIZE
        state[20] = float(lateral_friction) if lateral_friction is not None else (
            MATERIAL_MAP.get(material, MATERIAL_MAP['rubber'])['friction']
        )
        state[25:28] = get_object_bbox(shape, float(scale) if scale and scale > 0 else DEFAULT_SIZE)
        state[34] = float(restitution) if restitution is not None else (
            MATERIAL_MAP.get(material, MATERIAL_MAP['rubber'])['restitution']
        )
    state[15:18] = COLOR_MAP.get(color, [0.5, 0.5, 0.5])
    state[18] = SHAPE_MAP.get(shape, 0)
    state[19] = 0.0  # is_static
    state[21] = 1.0 if inside_scene else 0.0
    state[22] = 0.0  # inertia (no charge slot here; charge is unrepresentable)
    return state


def comphy_scene_to_state_tensor(
    annotation: Dict[str, Any],
    max_objects: Optional[int] = None,
    clevrer_align: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Convert a ComPhy annotation dict to ``(states [T,N,35], masks [T,N], metadata)``.

    ``max_objects`` left as None lets the tensor match the scene's real object
    count (matches the CLEVRER converter's behavior; the adapter handles
    variable N via ``object_mask``). Pass an int to force padding/truncation.

    ``clevrer_align`` (see ``_construct_state_vector``) puts the states back on
    the CLEVRER training distribution: it corrects the channels that are OOD vs
    the encoder's training data (mass, friction, restitution, radius,
    quaternion, angular velocity) while leaving position and velocity as-is
    (both verified in-range against ``clevrer_training_expanded.h5``).
    """
    obj_props = annotation.get("object_property", [])
    trajectory = annotation.get("motion_trajectory", [])
    if not obj_props:
        raise ValueError("ComPhy annotation is missing 'object_property'")
    if not trajectory:
        raise ValueError("ComPhy annotation is missing 'motion_trajectory'")

    real_n = len(obj_props)
    n = max_objects if max_objects is not None else real_n
    t = len(trajectory)
    states = np.zeros((t, n, 35), dtype=np.float32)
    masks = np.zeros((t, n), dtype=np.float32)

    for ti, frame in enumerate(trajectory):
        for oi, obj_state in enumerate(frame.get("objects", [])):
            if oi >= n:
                break
            prop = obj_props[oi]
            inside = bool(obj_state.get("inside_scene", True))
            states[ti, oi] = _construct_state_vector(
                position=obj_state.get("location", [0.0, 0.0, 0.0]),
                velocity=obj_state.get("velocity", [0.0, 0.0, 0.0]),
                angular_velocity=obj_state.get("angular_velocity", [0.0, 0.0, 0.0]),
                orientation_euler=obj_state.get("orientation", [0.0, 0.0, 0.0]),
                shape=prop.get("shape", "sphere"),
                color=prop.get("color", "gray"),
                material=prop.get("material", "rubber"),
                mass=prop.get("mass", 1.0),
                scale=prop.get("scale", 0.0),
                restitution=prop.get("restitution", None),
                lateral_friction=prop.get("lateral_friction", None),
                inside_scene=inside,
                clevrer_align=clevrer_align,
            )
            masks[ti, oi] = 1.0 if inside else 0.5

    metadata = {
        "scene_index": annotation.get("scene_index", -1),
        "num_objects": real_n,
        "num_frames": t,
        "benchmark": "comphy",
        "objects": [
            {
                "shape": p.get("shape"),
                "color": p.get("color"),
                "material": p.get("material"),
                "comphy_mass": p.get("mass"),
                "comphy_charge": p.get("charge"),
                "comphy_scale": p.get("scale"),
            }
            for p in obj_props[:n]
        ],
        "collisions": annotation.get("collision", []),
    }
    return states, masks, metadata
