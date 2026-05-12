"""Encoder-level out-of-distribution physics-signal probing.

Question this audit answers
---------------------------
The CompSAC-2026 paper's central empirical claim is that explicit physics signal
enables reasoning. The Phase 3 adapter has overfit to CLEVRER-specific MCQ
output behaviour (free-form transfer 99.45% ``unknown``; paraphrased-MCQ 88-92%
template fallback), so a full behavioural OOD test on the adapter would be
ambiguous. This script tests whether the **physics signal carried by the
frozen PhysicsFormer encoder** abstracts to a different physics domain, by
running the SAME linear probes on:

  - CLEVRER scenes (the paper's primary eval domain, used by the adapter)
  - Isaac Sim counterfactual scenes (different physics schemas, distinct
    dynamics; the encoder's training distribution but DIFFERENT scenarios)

If linear probes for physics features (object count, spatial centroid, collision
presence) recover those features at COMPARABLE accuracy across both domains,
the encoder's physics signal is domain-general. If R^2 / accuracy collapses on
the OOD domain, the encoder has overfit to CLEVRER and the abstraction claim
is unsupported.

Three feature layers are probed for each scene:

  - L0  ``input_sum``:  raw 35-dim state vector SUM-pooled over valid objects.
                        Sum (not mean) preserves count: a scene with N active
                        objects has Sum-magnitude proportional to N. This is
                        the baseline ``everything is preserved'' representation.
  - L1  ``encoder_pool``: per-object encoder output (after object_encoder MLP +
                          all transformer layers + final norm) mean-pooled by
                          mask. Mean-pool by mask is what the adapter does.
                          This is the layer whose R^2 the paper reports.
  - L2  ``prefix_pool``:  the actual 64 prefix tokens that get prepended to the
                          LLM (output of the adapter MLP applied to L1),
                          mean-pooled across the 64 tokens. Probing this layer
                          tests whether the adapter's MLP further compresses
                          or reorganises physics features.

Three probing targets:
  - object count        -- regression, sklearn LinearRegression, scored by R^2
  - spatial centroid    -- 3D regression, scored by mean per-axis R^2
  - collision presence  -- binary classification, sklearn LogisticRegression,
                           scored by accuracy and ROC AUC. Label is whether
                           the minimum pairwise object distance over the next
                           ``--collision_horizon`` frames crosses below
                           ``--collision_threshold``.

Output
------
A JSON summary table comparing per-(domain, layer, target) probe scores so the
abstraction claim can be evaluated quantitatively. Sample wins when:

  encoder_pool R^2 (CLEVRER) ~ encoder_pool R^2 (Isaac)
  encoder_pool R^2 (object_count) >> adapter_pool R^2 (object_count)

both of which we EXPECT from the paper's mechanistic story.

Usage
-----
::

    python compsac_2026_code/clevrer_benchmark/scripts/encoder_ood_probing.py \\
        [--n_per_domain 2000] [--isaac_schemas N] [--single_frame 64]

Snapshot-portable: paths anchor on ``__file__``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

# Snapshot-portable defaults.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _SCRIPT_DIR.parent
_RESULTS_DIR = _BENCH_DIR / 'results'
_PROJECT_ROOT = _BENCH_DIR.parent

DEFAULT_OUT = _RESULTS_DIR / 'encoder_ood_probing.json'
DEFAULT_CLEVRER_H5 = Path(os.environ.get(
    'CLEVRER_H5',
    os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5'),
))
DEFAULT_ISAAC_DIR = Path('D:/physics_counterfactual_hdf5')
DEFAULT_ADAPTER_CKPT = _PROJECT_ROOT / 'checkpoints' / 'adapter_phase3.pt'
DEFAULT_PHYSICS_CKPT = Path('$CHECKPOINT_DIR/stage1_best.pt')

# Make project modules importable.
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_BENCH_DIR))


# ---------------------------------------------------------------------------
# Encoder loader (reuses the adapter to get encoder + pooling in one shot)
# ---------------------------------------------------------------------------

def load_encoder_adapter(adapter_ckpt: Path, physics_ckpt: Path,
                         device: str = 'cuda'):
    """Load the Phase 3 adapter (which contains the frozen encoder + pooling).

    We use the adapter object because it provides clean APIs:
      ``adapter.extract_physics_features(states, mask)`` -> [B, hidden_dim]
      ``adapter.extract_object_embeddings(states, mask)`` -> [B, N, D], [B, N]

    Both run the encoder in inference mode; we never touch the LLM here.
    """
    from clevrer_benchmark.run_adapter_evaluation import load_adapter_model

    adapter = load_adapter_model(
        adapter_checkpoint=str(adapter_ckpt),
        physics_checkpoint=str(physics_ckpt),
        device=device,
    )
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    return adapter


# ---------------------------------------------------------------------------
# Domain adapters: load (states, masks) + per-scene labels
# ---------------------------------------------------------------------------

def _state_indices() -> Dict[str, slice]:
    """35-dim state vector layout from FullPhysicsFormer."""
    return {
        'position': slice(0, 3),
        'velocity': slice(3, 6),
        'quaternion': slice(6, 10),
        'ang_velocity': slice(10, 13),
        'mass': slice(13, 14),
        'radius': slice(14, 15),
        'color': slice(15, 18),
        'shape': slice(18, 19),
        'is_static': slice(19, 20),
        'friction': slice(20, 21),
        'is_active': slice(21, 22),
        'dimensions': slice(25, 28),
        'restitution': slice(34, 35),
    }


# CLEVRER's 8-color palette (matches scene_converter.COLOR_MAP). Used to
# discretise per-object RGB into a categorical color id for the
# scene-presence multi-label probe. Isaac Sim colors that don't fall on
# this palette are still snapped to the nearest entry; for Isaac probes
# this is a coarse approximation but still answers "does the prefix
# preserve a color signal at all?".
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
COLOR_NAMES = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
SHAPE_NAMES_CLEVRER = ['sphere', 'cube', 'cylinder']
SHAPE_NAMES_ISAAC = ['sphere', 'box', 'cylinder', 'capsule', 'mesh', 'robot', 'articulation']
MATERIAL_NAMES = ['rubber_like', 'metal_like']  # mass<1.5 vs >=1.5


def _compute_collision_labels(states_seq: np.ndarray, masks_seq: np.ndarray,
                              t_start: int, horizon: int,
                              threshold: float) -> np.ndarray:
    """Compute binary collision-presence labels per scene.

    For each scene we look at frames ``[t_start, t_start+horizon)`` and ask:
    does the minimum pairwise object distance ever drop below ``threshold``?
    A near-zero pairwise distance is a collision (or near-miss) signature.
    Self-pairs and masked-out objects are excluded.

    Args:
        states_seq: ``[B, T, N, 35]`` state tensor.
        masks_seq:  ``[B, T, N]`` valid-object mask.
        t_start:    starting frame index.
        horizon:    number of frames to look ahead.
        threshold:  minimum-distance threshold (in scene units).

    Returns:
        ``[B]`` int array of {0, 1}.
    """
    B, T, N, _ = states_seq.shape
    t_end = min(t_start + horizon, T)
    if t_end <= t_start:
        return np.zeros((B,), dtype=np.int32)
    pos = states_seq[:, t_start:t_end, :, 0:3]              # [B, t, N, 3]
    msk = masks_seq[:, t_start:t_end, :].astype(np.float32)  # [B, t, N]
    # Pairwise distances per frame: [B, t, N, N].
    diff = pos[:, :, :, None, :] - pos[:, :, None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1) + 1e-8)
    # Mask invalid pairs (either object missing) and self-pairs.
    pair_mask = msk[:, :, :, None] * msk[:, :, None, :]
    eye = np.eye(N, dtype=np.float32)
    pair_mask = pair_mask * (1.0 - eye)
    # Set masked / self pairs to a large value so they never become the min.
    big = 1e6
    dist = dist * pair_mask + big * (1.0 - pair_mask)
    # Min over (frames, objects, objects): scalar per scene.
    min_dist = dist.reshape(B, -1).min(axis=1)
    return (min_dist < threshold).astype(np.int32)


def _scene_labels(states_t: np.ndarray, masks_t: np.ndarray,
                  collide: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-scene probe targets at a single time t.

    Args:
        states_t: ``[B, N, 35]`` state at the probe frame.
        masks_t:  ``[B, N]`` mask at the probe frame.
        collide:  ``[B]`` collision-presence labels (precomputed).
    """
    msk = masks_t.astype(np.float32)
    counts = msk.sum(axis=-1).astype(np.float32)            # [B]
    pos = states_t[..., 0:3]                                # [B, N, 3]
    # Mean position over valid objects (avoid div-by-zero on empty scenes).
    msk_e = msk[..., None]
    centroid_num = (pos * msk_e).sum(axis=1)                # [B, 3]
    centroid_den = msk.sum(axis=1, keepdims=True).clip(min=1.0)
    centroid = centroid_num / centroid_den                  # [B, 3]
    return {
        'object_count': counts,
        'spatial_centroid': centroid,
        'collision_presence': collide.astype(np.int32),
    }


def _per_object_labels_from_state(
    states_t: np.ndarray,
    masks_t: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Per-object surface-attribute labels for valid objects only.

    Used by Family-A probes that ask: ``does the encoder preserve
    color/shape/material in the per-object embedding?``

    Returns a dict whose array length equals ``masks_t.sum()`` along axis -1
    (i.e. one row per valid object across all scenes):

    - ``color_rgb``     ``[N_total, 3]``  raw RGB from state[15:18]
    - ``color_class``   ``[N_total]``     nearest CLEVRER palette idx (0..7)
    - ``shape_type``    ``[N_total]``     integer shape id from state[18]
    - ``material_proxy````[N_total]``     1 if mass>=1.5 (metal-like) else 0
    - ``scene_index``   ``[N_total]``     batch index of the scene that owns
                                         each row (so probes can stratify)
    """
    valid = masks_t > 0.5                                      # [B, N]
    color_rgb_all = states_t[..., 15:18].astype(np.float32)    # [B, N, 3]
    shape_all = states_t[..., 18].astype(np.float32)           # [B, N]
    mass_all = states_t[..., 13].astype(np.float32)            # [B, N]

    color_rgb = color_rgb_all[valid]                           # [N_total, 3]
    shape_type = shape_all[valid].astype(np.int64)
    material = (mass_all[valid] >= 1.5).astype(np.int64)
    # Snap to CLEVRER palette by nearest-RGB.
    dists = np.linalg.norm(
        color_rgb[:, None, :] - COLOR_PALETTE[None, :, :], axis=-1,
    )                                                          # [N_total, 8]
    color_class = dists.argmin(axis=-1).astype(np.int64)
    # scene_index per row.
    B = masks_t.shape[0]
    row_to_scene = np.repeat(np.arange(B), valid.sum(axis=1))
    return {
        'color_rgb': color_rgb,
        'color_class': color_class,
        'shape_type': shape_type,
        'material_proxy': material,
        'scene_index': row_to_scene,
    }


def _scene_presence_labels_from_state(
    states_t: np.ndarray,
    masks_t: np.ndarray,
    n_shapes: int = 8,
) -> Dict[str, np.ndarray]:
    """Multi-label per-scene presence indicators.

    Used by Family-B probes that ask: ``can a linear classifier on the
    LLM-visible prefix decide which colors / shapes / materials are
    present in the scene?`` The labels are OR-reductions over valid
    objects.

    Returns:
    - ``colors_present``    ``[B, 8]``  binary indicator per palette color
    - ``shapes_present``    ``[B, n_shapes]`` binary indicator per shape id
    - ``materials_present`` ``[B, 2]``  binary indicator (rubber/metal proxy)
    """
    B, N = masks_t.shape
    valid = masks_t > 0.5
    color_rgb_all = states_t[..., 15:18].astype(np.float32)
    shape_all = states_t[..., 18].astype(np.float32)
    mass_all = states_t[..., 13].astype(np.float32)

    # Discretise color to palette index.
    color_dists = np.linalg.norm(
        color_rgb_all[:, :, None, :] - COLOR_PALETTE[None, None, :, :],
        axis=-1,
    )                                                      # [B, N, 8]
    color_idx = color_dists.argmin(axis=-1)                # [B, N]

    n_colors = COLOR_PALETTE.shape[0]
    colors_present = np.zeros((B, n_colors), dtype=np.int32)
    shapes_present = np.zeros((B, n_shapes), dtype=np.int32)
    materials_present = np.zeros((B, 2), dtype=np.int32)

    for b in range(B):
        v_n = np.nonzero(valid[b])[0]
        for n in v_n:
            colors_present[b, int(color_idx[b, n])] = 1
            s = int(shape_all[b, n])
            if 0 <= s < n_shapes:
                shapes_present[b, s] = 1
            m = 1 if mass_all[b, n] >= 1.5 else 0
            materials_present[b, m] = 1

    return {
        'colors_present': colors_present,
        'shapes_present': shapes_present,
        'materials_present': materials_present,
    }


def load_clevrer_subset(h5_path: Path, n: int, single_frame: int,
                        collision_horizon: int, collision_threshold: float,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Sample ``n`` CLEVRER scene-states at frame ``single_frame``.

    Each "row" in CLEVRER's H5 corresponds to a scene/question instance, but
    multiple questions share the same scene; we deduplicate scenes by
    ``scene_index`` (parsed from metadata) so probe samples are independent.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, 'r') as f:
        states_all = f['states']                                # [Q, T, N, 35]
        masks_all = f['masks']                                  # [Q, T, N]
        metadata = f['metadata']                                # [Q] objects
        Q, T, N, _ = states_all.shape
        # Deduplicate by scene_index from metadata.
        scene_to_qidx: Dict[int, int] = {}
        for q in range(Q):
            try:
                meta = json.loads(metadata[q].decode())
                sid = int(meta.get('scene_index', q))
            except Exception:
                sid = q
            if sid not in scene_to_qidx:
                scene_to_qidx[sid] = q
        unique_qidx = np.array(list(scene_to_qidx.values()), dtype=np.int64)
        rng.shuffle(unique_qidx)
        sampled_qidx = sorted(unique_qidx[:n].tolist())
        # Read sampled rows.
        states_seq = states_all[sampled_qidx]                   # [n, T, N, 35]
        masks_seq = masks_all[sampled_qidx]                     # [n, T, N]
    t = min(single_frame, T - 1)
    states_t = states_seq[:, t, :, :]                           # [n, N, 35]
    masks_t = masks_seq[:, t, :]                                # [n, N]
    collide = _compute_collision_labels(
        states_seq, masks_seq, t_start=t, horizon=collision_horizon,
        threshold=collision_threshold)
    labels = _scene_labels(states_t, masks_t, collide)
    return states_t, masks_t, labels


def load_isaac_subset(h5_paths: List[Path], n: int, single_frame: int,
                      collision_horizon: int, collision_threshold: float,
                      seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Sample ``n`` Isaac Sim scene-states at frame ``single_frame``.

    Pulls evenly from each of ``h5_paths`` (one path per Isaac schema).
    """
    rng = np.random.default_rng(seed)
    per_file = max(1, n // max(1, len(h5_paths)))
    states_chunks = []
    masks_chunks = []
    label_chunks: Dict[str, List[np.ndarray]] = {
        'object_count': [],
        'spatial_centroid': [],
        'collision_presence': [],
    }
    for p in h5_paths:
        with h5py.File(p, 'r') as f:
            states_all = f['states']                            # [E, T, N, 35]
            masks_all = f['masks']                              # [E, T, N]
            E, T, N_max, _ = states_all.shape
            t = min(single_frame, T - 1)
            sample_n = min(per_file, E)
            idx = sorted(rng.choice(E, size=sample_n, replace=False).tolist())
            states_seq = states_all[idx]                        # [b, T, N, 35]
            masks_seq = masks_all[idx]                          # [b, T, N]
        states_t = states_seq[:, t, :, :]
        masks_t = masks_seq[:, t, :]
        collide = _compute_collision_labels(
            states_seq, masks_seq, t_start=t, horizon=collision_horizon,
            threshold=collision_threshold)
        labels = _scene_labels(states_t, masks_t, collide)
        states_chunks.append(states_t)
        masks_chunks.append(masks_t)
        for key in label_chunks:
            label_chunks[key].append(labels[key])

    # Concatenate. Different schemas may have different N_max -- pad to the
    # max width and merge.
    max_N = max(s.shape[1] for s in states_chunks)
    padded_states = []
    padded_masks = []
    for s, m in zip(states_chunks, masks_chunks):
        pad_N = max_N - s.shape[1]
        if pad_N > 0:
            s_pad = np.pad(s, ((0, 0), (0, pad_N), (0, 0)))
            m_pad = np.pad(m, ((0, 0), (0, pad_N)))
        else:
            s_pad, m_pad = s, m
        padded_states.append(s_pad)
        padded_masks.append(m_pad)
    states_t = np.concatenate(padded_states, axis=0)
    masks_t = np.concatenate(padded_masks, axis=0)
    labels = {k: np.concatenate(v, axis=0) for k, v in label_chunks.items()}
    return states_t, masks_t, labels


# ---------------------------------------------------------------------------
# Feature extraction at three layers
# ---------------------------------------------------------------------------

def _ensure_max_objects(states_np: np.ndarray, masks_np: np.ndarray,
                        target_N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pad or truncate the object dimension to ``target_N``.

    The encoder was built with a fixed ``max_objects`` size (inferred from the
    adapter checkpoint). Isaac Sim has up to 20 objects per scene; CLEVRER has
    up to 6. We need to bring both into the encoder's expected shape.
    """
    N = states_np.shape[1]
    if N == target_N:
        return states_np, masks_np
    if N < target_N:
        pad_N = target_N - N
        s = np.pad(states_np, ((0, 0), (0, pad_N), (0, 0)))
        m = np.pad(masks_np, ((0, 0), (0, pad_N)))
        return s, m
    # N > target_N: truncate to first target_N objects (preserve mask).
    return states_np[:, :target_N, :], masks_np[:, :target_N]


@torch.no_grad()
def _extract_layer_features(adapter, states_np: np.ndarray, masks_np: np.ndarray,
                            batch_size: int = 64,
                            device: str = 'cuda') -> Dict[str, np.ndarray]:
    """Run encoder forward and return mean-pooled features at each layer.

    Scene-level layers (returned with batch dim B):
      - ``input_sum``    : raw 35-dim state vector SUM-pooled over valid
                           objects. Sum preserves count (empty slots = 0).
      - ``encoder_pool`` : per-object encoder output, mean-pooled by mask.
      - ``prefix_pool``  : the 64 LLM prefix tokens, mean-pooled across
                           tokens (this is the LLM-visible representation).

    Per-object layers (returned with batch dim N_total = sum of valid
    objects across the batch -- one row per object, lined up with the
    rows produced by ``_per_object_labels_from_state``):
      - ``input_per_obj``   : raw 35-dim per-object state.
      - ``encoder_per_obj`` : per-object encoder output (no pooling).

    Returns
    -------
    dict[str, np.ndarray]
    """
    target_N = adapter.physics_model.max_objects
    states_np, masks_np = _ensure_max_objects(states_np, masks_np, target_N)

    B = states_np.shape[0]
    input_pools = []
    encoder_pools = []
    adapter_pools = []
    input_per_obj_chunks = []
    encoder_per_obj_chunks = []
    for i in range(0, B, batch_size):
        states = torch.from_numpy(states_np[i:i + batch_size]).float().to(device)
        masks = torch.from_numpy(masks_np[i:i + batch_size]).float().to(device)

        # L0 input_sum: raw 35-dim state, SUM-pooled by mask.
        # Sum preserves count: empty slots contribute 0. The probe should
        # trivially recover ``object_count`` from this.
        msk_e = masks.unsqueeze(-1)
        input_sum = (states * msk_e).sum(dim=1)              # [B, 35]
        input_pools.append(input_sum.cpu().numpy())

        # L1 encoder_pool: per-object encoder output, mean-pool by mask.
        # Mean-pool is exactly what extract_physics_features does.
        obj_emb, flat_mask = adapter.extract_object_embeddings(states, masks)
        if flat_mask is None:
            flat_mask = masks
            enc_pool = obj_emb.mean(dim=1)
        else:
            m_e = flat_mask.unsqueeze(-1)
            denom2 = flat_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            enc_pool = (obj_emb * m_e).sum(dim=1) / denom2
        encoder_pools.append(enc_pool.cpu().numpy())

        # Per-object features for surface-attribute probes.
        # Use the SAME mask the encoder used to compute valid objects so
        # the row order matches ``_per_object_labels_from_state``.
        valid_b = (flat_mask.cpu().numpy() > 0.5)            # [b, N]
        states_b = states.cpu().numpy()                       # [b, N, 35]
        obj_emb_b = obj_emb.cpu().numpy()                     # [b, N, D]
        input_per_obj_chunks.append(states_b[valid_b])        # [n_valid, 35]
        encoder_per_obj_chunks.append(obj_emb_b[valid_b])     # [n_valid, D]

        # L2 prefix_pool: actual 64 prefix tokens, mean-pooled across tokens.
        # This is the layer the LLM sees -- one MLP transform past L1.
        physics_features = adapter.extract_physics_features(states, masks)
        prefix_tokens = adapter.create_prefix_tokens(physics_features)  # [B, 64, 768]
        prefix_pool = prefix_tokens.mean(dim=1)                          # [B, 768]
        adapter_pools.append(prefix_pool.cpu().numpy())

    return {
        'input_sum': np.concatenate(input_pools, axis=0),
        'encoder_pool': np.concatenate(encoder_pools, axis=0),
        'prefix_pool': np.concatenate(adapter_pools, axis=0),
        'input_per_obj': np.concatenate(input_per_obj_chunks, axis=0),
        'encoder_per_obj': np.concatenate(encoder_per_obj_chunks, axis=0),
    }


# ---------------------------------------------------------------------------
# Linear probes
# ---------------------------------------------------------------------------

def _fit_regression_probe(X_train, y_train, X_test, y_test) -> Dict[str, float]:
    """Linear regression with closed-form fit; report R^2 (and per-axis if 2D)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    if y_train.ndim == 1:
        clf = Ridge(alpha=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return {
            'r2': float(r2_score(y_test, y_pred)),
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
        }
    # Multi-target regression (e.g., 3D centroid).
    clf = Ridge(alpha=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    per_axis = [float(r2_score(y_test[:, k], y_pred[:, k]))
                for k in range(y_train.shape[1])]
    return {
        'r2': float(np.mean(per_axis)),
        'r2_per_axis': per_axis,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
    }


def _fit_classification_probe(X_train, y_train, X_test, y_test) -> Dict[str, float]:
    """L2-regularized logistic regression; report acc and ROC AUC.

    Falls back gracefully when only one class is present in train or test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    n_classes = len(np.unique(y_train))
    if n_classes < 2:
        # Degenerate: only one class. Predict the majority class trivially.
        pred = np.full_like(y_test, fill_value=int(y_train[0]))
        return {
            'acc': float(accuracy_score(y_test, pred)),
            'auc': float('nan'),
            'class_balance_train': float(y_train.mean()),
            'class_balance_test': float(y_test.mean()),
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'note': 'single-class train -- AUC undefined',
        }
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    try:
        proba = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = float('nan')
    return {
        'acc': float(accuracy_score(y_test, y_pred)),
        'auc': auc,
        'class_balance_train': float(y_train.mean()),
        'class_balance_test': float(y_test.mean()),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
    }


def _split_train_test(X: np.ndarray, y: np.ndarray, test_frac: float = 0.25,
                      seed: int = 0) -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _fit_multiclass_probe(X_train, y_train, X_test, y_test,
                          class_names: Optional[List[str]] = None) -> Dict:
    """Multi-class logistic-regression probe; reports overall + per-class acc.

    Used by the per-object Family-A probes for ``shape_type`` (3 / 7 way)
    and ``color_class`` (8 way). Falls back to a degenerate single-class
    score when only one class is present in train.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    classes = np.unique(y_train)
    n_classes = len(classes)
    if n_classes < 2:
        majority = int(classes[0]) if n_classes == 1 else 0
        pred = np.full_like(y_test, fill_value=majority)
        return {
            'acc': float(accuracy_score(y_test, pred)),
            'n_classes': int(n_classes),
            'class_balance_train': {},
            'class_balance_test': {},
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'note': 'degenerate single-class train',
        }
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    overall_acc = float(accuracy_score(y_test, y_pred))
    # Per-class accuracy: P(pred=c | y=c) for each c that appears in test.
    per_class = {}
    for c in np.unique(y_test):
        mask_c = (y_test == c)
        if mask_c.sum() == 0:
            continue
        per_class_name = (class_names[int(c)] if class_names is not None
                          and 0 <= int(c) < len(class_names) else str(int(c)))
        per_class[per_class_name] = float((y_pred[mask_c] == c).mean())
    cb_train = {(class_names[int(c)] if class_names is not None
                 and 0 <= int(c) < len(class_names) else str(int(c))):
                float(np.mean(y_train == c)) for c in classes}
    cb_test = {(class_names[int(c)] if class_names is not None
                and 0 <= int(c) < len(class_names) else str(int(c))):
               float(np.mean(y_test == c)) for c in np.unique(y_test)}
    return {
        'acc': overall_acc,
        'n_classes': int(n_classes),
        'per_class_acc': per_class,
        'class_balance_train': cb_train,
        'class_balance_test': cb_test,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
    }


def _fit_multilabel_probe(X_train, Y_train, X_test, Y_test,
                          label_names: List[str]) -> Dict:
    """One-vs-rest binary classifier per label; report per-label + means.

    Used by Family-B probes ``colors_present``, ``shapes_present``,
    ``materials_present``. Each column of Y is one binary indicator.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    K = Y_train.shape[1]
    per_label: List[Dict] = []
    for k in range(K):
        y_tr = Y_train[:, k]
        y_te = Y_test[:, k]
        name = label_names[k] if k < len(label_names) else str(k)
        if len(np.unique(y_tr)) < 2:
            # Degenerate: predict the only class seen.
            pred_const = int(y_tr[0]) if len(y_tr) > 0 else 0
            per_label.append({
                'label': name,
                'acc': float(np.mean(y_te == pred_const)),
                'auc': float('nan'),
                'pos_rate_train': float(y_tr.mean()) if len(y_tr) > 0 else 0.0,
                'pos_rate_test': float(y_te.mean()) if len(y_te) > 0 else 0.0,
                'note': 'single-class train',
            })
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_train, y_tr)
        pred = clf.predict(X_test)
        try:
            proba = clf.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_te, proba))
        except Exception:
            auc = float('nan')
        per_label.append({
            'label': name,
            'acc': float(accuracy_score(y_te, pred)),
            'auc': auc,
            'pos_rate_train': float(y_tr.mean()),
            'pos_rate_test': float(y_te.mean()),
        })
    accs = [p['acc'] for p in per_label]
    aucs = [p['auc'] for p in per_label
            if isinstance(p['auc'], float) and not math.isnan(p['auc'])]
    return {
        'mean_acc': float(np.mean(accs)) if accs else float('nan'),
        'mean_auc': float(np.mean(aucs)) if aucs else float('nan'),
        'per_label': per_label,
        'n_labels': K,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Encoder-level OOD physics-signal probing.')
    parser.add_argument('--n_per_domain', type=int, default=2000)
    parser.add_argument('--single_frame', type=int, default=64)
    parser.add_argument('--collision_horizon', type=int, default=24,
                        help='Frames to look ahead for a collision proxy '
                             '(default: 24 = 1 second @ 24fps).')
    parser.add_argument('--collision_threshold', type=float, default=0.7,
                        help='Min pairwise distance threshold for the '
                             'collision-presence label (default: 0.7 -- '
                             'CLEVRER objects have radius ~0.4-0.5).')
    parser.add_argument('--clevrer_h5', type=Path, default=DEFAULT_CLEVRER_H5)
    parser.add_argument('--isaac_dir', type=Path, default=DEFAULT_ISAAC_DIR)
    parser.add_argument('--isaac_schemas', type=int, default=8,
                        help='Number of Isaac H5 schemas to sample from.')
    parser.add_argument('--adapter_ckpt', type=Path, default=DEFAULT_ADAPTER_CKPT)
    parser.add_argument('--physics_ckpt', type=Path, default=DEFAULT_PHYSICS_CKPT)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'[setup] device={device}')

    # ---------- 1. load encoder ----------
    print(f'[load] adapter: {args.adapter_ckpt}')
    print(f'[load] physics: {args.physics_ckpt}')
    adapter = load_encoder_adapter(args.adapter_ckpt, args.physics_ckpt, device)
    print(f'[load] encoder max_objects={adapter.physics_model.max_objects}, '
          f'hidden_dim={adapter.physics_model.hidden_dim}')

    # ---------- 2. select Isaac schemas ----------
    isaac_h5s = sorted(args.isaac_dir.glob('cf_isaac_*.h5'))
    if not isaac_h5s:
        raise FileNotFoundError(f'No cf_isaac_*.h5 files in {args.isaac_dir}')
    isaac_h5s = isaac_h5s[:args.isaac_schemas]
    print(f'[isaac] using {len(isaac_h5s)} schemas:')
    for p in isaac_h5s:
        print(f'        {p.name}')

    # ---------- 3. load + label per-domain data ----------
    print(f'[load] CLEVRER: n={args.n_per_domain}, frame={args.single_frame}')
    t0 = time.time()
    cl_states, cl_masks, cl_labels = load_clevrer_subset(
        args.clevrer_h5, args.n_per_domain, args.single_frame,
        args.collision_horizon, args.collision_threshold, seed=args.seed)
    print(f'        states {cl_states.shape}, masks {cl_masks.shape}, '
          f'count avg={cl_labels["object_count"].mean():.2f}, '
          f'collide rate={cl_labels["collision_presence"].mean():.3f}, '
          f'{time.time() - t0:.1f}s')

    print(f'[load] Isaac:   n={args.n_per_domain}, frame={args.single_frame}')
    t0 = time.time()
    is_states, is_masks, is_labels = load_isaac_subset(
        isaac_h5s, args.n_per_domain, args.single_frame,
        args.collision_horizon, args.collision_threshold, seed=args.seed)
    print(f'        states {is_states.shape}, masks {is_masks.shape}, '
          f'count avg={is_labels["object_count"].mean():.2f}, '
          f'collide rate={is_labels["collision_presence"].mean():.3f}, '
          f'{time.time() - t0:.1f}s')

    # ---------- 4. extract features at 3 layers per domain ----------
    print(f'[extract] CLEVRER features...')
    t0 = time.time()
    cl_features = _extract_layer_features(
        adapter, cl_states, cl_masks, batch_size=args.batch_size, device=device)
    for k, v in cl_features.items():
        print(f'        {k}: {v.shape}')
    print(f'        {time.time() - t0:.1f}s')

    print(f'[extract] Isaac features...')
    t0 = time.time()
    is_features = _extract_layer_features(
        adapter, is_states, is_masks, batch_size=args.batch_size, device=device)
    for k, v in is_features.items():
        print(f'        {k}: {v.shape}')
    print(f'        {time.time() - t0:.1f}s')

    # ---------- 4b. compute per-object + scene-presence labels ------------
    # Pad to the encoder's expected width so the row order matches the
    # per-object feature extraction (which also pads via _ensure_max_objects).
    target_N = adapter.physics_model.max_objects
    cl_states_p, cl_masks_p = _ensure_max_objects(cl_states, cl_masks, target_N)
    is_states_p, is_masks_p = _ensure_max_objects(is_states, is_masks, target_N)
    cl_obj_labels = _per_object_labels_from_state(cl_states_p, cl_masks_p)
    is_obj_labels = _per_object_labels_from_state(is_states_p, is_masks_p)
    cl_presence = _scene_presence_labels_from_state(cl_states_p, cl_masks_p)
    is_presence = _scene_presence_labels_from_state(is_states_p, is_masks_p)
    print(f'[labels] CLEVRER per-object rows: {len(cl_obj_labels["color_rgb"])}, '
          f'Isaac: {len(is_obj_labels["color_rgb"])}')

    # Sanity: feature row count must match label row count.
    assert cl_features['encoder_per_obj'].shape[0] == cl_obj_labels['color_rgb'].shape[0], (
        'CLEVRER per-object row mismatch: '
        f'{cl_features["encoder_per_obj"].shape[0]} feats vs '
        f'{cl_obj_labels["color_rgb"].shape[0]} labels')
    assert is_features['encoder_per_obj'].shape[0] == is_obj_labels['color_rgb'].shape[0], (
        'Isaac per-object row mismatch: '
        f'{is_features["encoder_per_obj"].shape[0]} feats vs '
        f'{is_obj_labels["color_rgb"].shape[0]} labels')

    # ---------- 5. fit linear probes per (domain, layer, target) ----------
    domains = {'CLEVRER': (cl_features, cl_labels),
               'Isaac':   (is_features, is_labels)}
    layers = ['input_sum', 'encoder_pool', 'prefix_pool']
    targets_reg = ['object_count', 'spatial_centroid']
    target_clf = 'collision_presence'

    print()
    print('=' * 88)
    print('Encoder OOD probing: per-(domain, layer, target) probe scores')
    print('=' * 88)
    results: Dict[str, Dict[str, Dict[str, Dict]]] = {}
    for domain, (feats, lbls) in domains.items():
        print(f'\n--- {domain} (n={args.n_per_domain}) ---')
        results[domain] = {}
        for layer in layers:
            X = feats[layer]
            results[domain][layer] = {}
            print(f'  [{layer}: dim={X.shape[1]}]')
            # Regression targets.
            for tgt in targets_reg:
                y = lbls[tgt]
                X_tr, X_te, y_tr, y_te = _split_train_test(X, y, seed=args.seed)
                res = _fit_regression_probe(X_tr, y_tr, X_te, y_te)
                results[domain][layer][tgt] = res
                if 'r2_per_axis' in res:
                    print(f'    {tgt:<22}  R^2 = {res["r2"]:.4f}'
                          f'  (per axis: {res["r2_per_axis"]})')
                else:
                    print(f'    {tgt:<22}  R^2 = {res["r2"]:.4f}')
            # Classification target.
            y = lbls[target_clf]
            X_tr, X_te, y_tr, y_te = _split_train_test(X, y, seed=args.seed)
            res = _fit_classification_probe(X_tr, y_tr, X_te, y_te)
            results[domain][layer][target_clf] = res
            print(f'    {target_clf:<22}  acc = {res["acc"]:.4f}'
                  f'  auc = {res["auc"]:.4f}'
                  f'  (pos rate train={res["class_balance_train"]:.3f}'
                  f', test={res["class_balance_test"]:.3f})')

    # ---------- 5b. Family-A: per-object surface-attribute probes ----------
    # Tests whether the encoder's per-object embedding preserves color,
    # shape, and material. If encoder_per_obj fails (esp. while
    # input_per_obj succeeds), the encoder is the bottleneck for any
    # free-form question that asks about an object's surface attributes.
    print()
    print('=' * 88)
    print('Family A: per-object surface-attribute probes')
    print('=' * 88)
    per_obj_layers = ['input_per_obj', 'encoder_per_obj']
    per_obj_label_packs = {'CLEVRER': cl_obj_labels, 'Isaac': is_obj_labels}
    shape_classnames = {'CLEVRER': SHAPE_NAMES_CLEVRER,
                        'Isaac':   SHAPE_NAMES_ISAAC}
    for domain, lpack in per_obj_label_packs.items():
        feats = cl_features if domain == 'CLEVRER' else is_features
        n_rows = lpack['color_rgb'].shape[0]
        print(f'\n--- {domain} per-object (n_objects={n_rows}) ---')
        for layer in per_obj_layers:
            X = feats[layer]
            if layer not in results[domain]:
                results[domain][layer] = {}
            print(f'  [{layer}: dim={X.shape[1]}]')
            # color_rgb: 3-D regression
            X_tr, X_te, y_tr, y_te = _split_train_test(
                X, lpack['color_rgb'], seed=args.seed)
            res = _fit_regression_probe(X_tr, y_tr, X_te, y_te)
            results[domain][layer]['color_rgb'] = res
            print(f'    color_rgb (3D)         R^2 = {res["r2"]:.4f}'
                  f'  (per channel: {res.get("r2_per_axis")})')
            # color_class: 8-way multi-class
            X_tr, X_te, y_tr, y_te = _split_train_test(
                X, lpack['color_class'], seed=args.seed)
            res = _fit_multiclass_probe(X_tr, y_tr, X_te, y_te,
                                        class_names=COLOR_NAMES)
            results[domain][layer]['color_class'] = res
            print(f'    color_class (8-way)    acc = {res["acc"]:.4f}'
                  f'  (n_classes seen={res["n_classes"]})')
            # shape_type: domain-specific class space
            X_tr, X_te, y_tr, y_te = _split_train_test(
                X, lpack['shape_type'], seed=args.seed)
            res = _fit_multiclass_probe(X_tr, y_tr, X_te, y_te,
                                        class_names=shape_classnames[domain])
            results[domain][layer]['shape_type'] = res
            print(f'    shape_type             acc = {res["acc"]:.4f}'
                  f'  (n_classes seen={res["n_classes"]})')
            # material_proxy: binary
            X_tr, X_te, y_tr, y_te = _split_train_test(
                X, lpack['material_proxy'], seed=args.seed)
            res = _fit_classification_probe(X_tr, y_tr, X_te, y_te)
            results[domain][layer]['material_proxy'] = res
            print(f'    material_proxy         acc = {res["acc"]:.4f}'
                  f'  auc = {res["auc"]:.4f}')

    # ---------- 5c. Family-B: scene-presence multi-label probes -----------
    # Tests whether the LLM-visible prefix preserves "what colors / shapes
    # / materials are in this scene". Multi-label: independent binary
    # classifier per palette entry, mean acc/AUC reported. Run on every
    # scene-level layer so we can see info-loss progressing through
    # input_sum -> encoder_pool -> prefix_pool.
    print()
    print('=' * 88)
    print('Family B: scene-presence multi-label probes')
    print('=' * 88)
    presence_packs = {'CLEVRER': cl_presence, 'Isaac': is_presence}
    presence_specs = [
        ('colors_present',    COLOR_NAMES,         8),
        ('shapes_present',    SHAPE_NAMES_ISAAC,   8),  # shape array width=8
        ('materials_present', MATERIAL_NAMES,      2),
    ]
    for domain, ppack in presence_packs.items():
        feats = cl_features if domain == 'CLEVRER' else is_features
        print(f'\n--- {domain} scene-presence (n={args.n_per_domain}) ---')
        for layer in layers:
            X = feats[layer]
            print(f'  [{layer}: dim={X.shape[1]}]')
            for tgt, names, _width in presence_specs:
                Y = ppack[tgt]
                X_tr, X_te, Y_tr, Y_te = _split_train_test(
                    X, Y, seed=args.seed)
                res = _fit_multilabel_probe(X_tr, Y_tr, X_te, Y_te,
                                            label_names=names)
                results[domain][layer][tgt] = res
                print(f'    {tgt:<22}  mean_acc = {res["mean_acc"]:.4f}'
                      f'  mean_auc = {res["mean_auc"]:.4f}'
                      f'  (K={res["n_labels"]})')

    # ---------- 6. cross-domain abstraction summary ----------
    print()
    print('=' * 88)
    print('Cross-domain abstraction summary (CLEVRER vs Isaac)')
    print('=' * 88)
    for layer in layers:
        print(f'\n[{layer}]')
        for tgt in targets_reg:
            cl = results['CLEVRER'][layer][tgt]['r2']
            iss = results['Isaac'][layer][tgt]['r2']
            delta = iss - cl
            print(f'  {tgt:<22}  CLEVRER R^2 = {cl:.4f}'
                  f'  | Isaac R^2 = {iss:.4f}'
                  f'  | delta = {delta:+.4f}')
        cl = results['CLEVRER'][layer][target_clf]['acc']
        iss = results['Isaac'][layer][target_clf]['acc']
        print(f'  {target_clf:<22}  CLEVRER acc = {cl:.4f}'
              f'  | Isaac acc = {iss:.4f}'
              f'  | delta = {iss - cl:+.4f}')

    # Surface-attribute summary (Family A on encoder_per_obj).
    print()
    print('-' * 88)
    print('Surface-attribute extraction (encoder_per_obj):')
    print('-' * 88)
    for tgt, score_key, label in [
        ('color_rgb',      'r2',  'R^2'),
        ('color_class',    'acc', 'acc'),
        ('shape_type',     'acc', 'acc'),
        ('material_proxy', 'acc', 'acc'),
    ]:
        cl = results['CLEVRER']['encoder_per_obj'][tgt][score_key]
        iss = results['Isaac']['encoder_per_obj'][tgt][score_key]
        print(f'  {tgt:<18} {label} | CLEVRER={cl:.4f}'
              f' | Isaac={iss:.4f} | delta={iss - cl:+.4f}')

    # Scene-presence summary (Family B on prefix_pool -- the LLM-visible layer).
    print()
    print('-' * 88)
    print('Scene-presence at the LLM prefix (prefix_pool):')
    print('-' * 88)
    for tgt in ['colors_present', 'shapes_present', 'materials_present']:
        cl = results['CLEVRER']['prefix_pool'][tgt]['mean_auc']
        iss = results['Isaac']['prefix_pool'][tgt]['mean_auc']
        print(f'  {tgt:<22}  mean_auc | CLEVRER={cl:.4f}'
              f' | Isaac={iss:.4f} | delta={iss - cl:+.4f}')

    # ---------- 7. write JSON ----------
    summary = {
        'config': {
            'n_per_domain': args.n_per_domain,
            'single_frame': args.single_frame,
            'collision_horizon': args.collision_horizon,
            'collision_threshold': args.collision_threshold,
            'clevrer_h5': str(args.clevrer_h5),
            'isaac_dir': str(args.isaac_dir),
            'isaac_schemas': [p.name for p in isaac_h5s],
            'adapter_ckpt': str(args.adapter_ckpt),
            'physics_ckpt': str(args.physics_ckpt),
            'seed': args.seed,
            'device': device,
        },
        'features': {
            'scene_layers': layers,
            'per_object_layers': per_obj_layers,
            'feature_dims': {layer: int(cl_features[layer].shape[1])
                             for layer in (layers + per_obj_layers)},
        },
        'probe_targets': {
            'scene_regression': targets_reg,
            'scene_classification': [target_clf],
            'per_object': ['color_rgb', 'color_class', 'shape_type',
                           'material_proxy'],
            'scene_presence_multilabel': [t for t, _, _ in presence_specs],
        },
        'results': results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\n  full summary written to: {args.out}')


if __name__ == '__main__':
    main()
