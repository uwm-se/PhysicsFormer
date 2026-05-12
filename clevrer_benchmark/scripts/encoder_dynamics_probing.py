"""Per-object DYNAMICS probing of the encoder + adapter prefix.

Why this script exists
----------------------
The Phase 3 surface-attribute probe (``encoder_ood_probing.py`` -> ``encoder_ood_probing_v2_surface.json``)
established that the frozen PhysicsFormer encoder destroys per-object color
(R^2 1.000 -> -0.004) but preserves shape/material (acc 1.000 -> 0.922 / 0.842).
That probe answered "what surface attributes survive?". Phase 6 mitigates the
color leakage with `inject_scene_text=True`, so the LLM gets colors as text.

The remaining open question for whether a stronger LLM can lift Phase 6's
free-form accuracy is: **does the prefix preserve enough DYNAMICS signal
(motion, future-collision, exits-frame) for the LLM to reason about
counterfactual / explanatory CLEVRER questions?**

If dynamics signal is at random (AUC 0.5) at the prefix layer, no LLM, no
matter how big, can answer "if X is removed, will Y collide with Z?" — the
information is simply not in the input it sees. If dynamics signal is
meaningfully above random, then the bottleneck is the LLM and a Qwen-1.5B
upgrade is justified.

This script is the per-object analogue of Family A in the surface probe,
but with **dynamics** targets instead of static surface attributes:

  - velocity_xyz (regression, R^2): per-object velocity at the probe frame.
                                    Direct readout from state[3:6].
  - speed (regression, R^2)        : ||velocity||_2.
  - is_moving (binary, AUC)        : speed > 0.1 (object is in motion).
  - future_collision (binary, AUC) : in [t, t+horizon), does this object's
                                     min-distance to any other valid
                                     object drop below collision_threshold?
                                     This is the per-object analogue of the
                                     scene-level collision_presence target.
  - exits_frame (binary, AUC)      : does the object's mask transition
                                     from 1 to 0 at any future frame
                                     (object leaves the camera view)?

Probed at three layers:
  - input_per_obj    : raw 35-dim per-object state. Sanity baseline (R^2=1
                       expected for velocity since v lives at state[3:6]).
  - encoder_per_obj  : per-object encoder output (same as surface probe).
  - prefix_per_obj   : per-object PREFIX VECTOR -- the K=4 tokens emitted
                       per object by adapter_per_object, flattened to
                       [N, K*llm_dim]. This is what the LLM actually sees
                       under Phase 6 (per-object prefix, tokens_per_object > 0).

Output
------
A JSON summary mirroring the surface probe layout:
  results['CLEVRER'][layer][target] = {r2, auc, n_train, n_test, ...}

Plus a console table that highlights the **input -> encoder -> prefix delta**
for each dynamics target so the bottleneck story is unambiguous.

Usage
-----
::

    python compsac_2026_code/clevrer_benchmark/scripts/encoder_dynamics_probing.py \\
        --adapter_ckpt compsac_2026_code/checkpoints/adapter_phase6_per_object_best.pt \\
        --n 1500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _SCRIPT_DIR.parent
_RESULTS_DIR = _BENCH_DIR / 'results'
_PROJECT_ROOT = _BENCH_DIR.parent

DEFAULT_OUT = _RESULTS_DIR / 'encoder_dynamics_probing.json'
DEFAULT_CLEVRER_H5 = Path(os.environ.get(
    'CLEVRER_H5',
    os.environ.get('CLEVRER_H5', 'data/clevrer_training_expanded.h5'),
))
DEFAULT_ADAPTER_CKPT = (
    _PROJECT_ROOT / 'checkpoints' / 'adapter_phase6_per_object_best.pt'
)
DEFAULT_PHYSICS_CKPT = Path('$CHECKPOINT_DIR/stage1_best.pt')

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_BENCH_DIR))

# Reuse the existing surface-probe infrastructure: encoder loader, train/test
# split, and probe fitters. We only override the label generators and add the
# prefix_per_obj feature extractor; everything else is shared so the two
# probes stay in sync.
from clevrer_benchmark.scripts.encoder_ood_probing import (  # noqa: E402
    load_encoder_adapter,
    _ensure_max_objects,
    _split_train_test,
    _fit_regression_probe,
    _fit_classification_probe,
)


# ---------------------------------------------------------------------------
# Data loader: full sequence (not single-frame) so we can compute future-
# collision and exits-frame labels.
# ---------------------------------------------------------------------------

def load_clevrer_sequence_subset(
    h5_path: Path,
    n: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Sample ``n`` CLEVRER scenes; return full state + mask sequences.

    Mirrors ``encoder_ood_probing.load_clevrer_subset`` but keeps the time
    axis intact so future-collision and exits-frame labels can be computed.

    Returns:
        states_seq: [n, T, N, 35]
        masks_seq:  [n, T, N]
        scene_ids:  list[int] of len n (parsed from H5 metadata; for logging)
    """
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, 'r') as f:
        states_all = f['states']
        masks_all = f['masks']
        metadata = f['metadata']
        Q = states_all.shape[0]
        # Deduplicate scenes by metadata['scene_index'] (multiple Q entries
        # share a scene because each Q is a question instance).
        scene_to_qidx: Dict[int, int] = {}
        scene_ids_list: List[int] = []
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
        scene_ids = [int(k) for k, v in scene_to_qidx.items() if v in sampled_qidx]
        states_seq = states_all[sampled_qidx]
        masks_seq = masks_all[sampled_qidx]
    return states_seq, masks_seq, scene_ids


# ---------------------------------------------------------------------------
# Per-object dynamics labels (one row per valid object across the batch).
# Row order matches feature extraction (which iterates valid_b in batch loop).
# ---------------------------------------------------------------------------

def per_object_dynamics_labels(
    states_seq: np.ndarray,    # [B, T, N, 35]
    masks_seq: np.ndarray,     # [B, T, N]
    t_probe: int,
    horizon: int,
    collision_threshold: float,
    speed_threshold: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Build per-object dynamics labels at probe frame ``t_probe``.

    Labels are aligned with valid objects at frame ``t_probe`` (i.e.,
    ``masks_seq[:, t_probe, :] > 0.5``), matching the row ordering produced
    by the feature extractor.

    Args:
        states_seq: full state sequence
        masks_seq:  full mask sequence
        t_probe:    frame index at which features are extracted
        horizon:    look-ahead window for future_collision / exits_frame
        collision_threshold: distance threshold for collision presence
        speed_threshold: |v| threshold for the binary is_moving label

    Returns:
        dict with keys:
          velocity_xyz   : [n_obj, 3] -- per-object velocity at t_probe
          speed          : [n_obj]    -- ||velocity||_2 at t_probe
          is_moving      : [n_obj]    -- binary speed > threshold
          future_collision : [n_obj]  -- binary, will collide in [t,t+H)
          exits_frame    : [n_obj]    -- binary, mask 1->0 in [t,t+H)
          scene_index    : [n_obj]    -- which scene this row belongs to
    """
    B, T, N, _ = states_seq.shape
    assert masks_seq.shape == (B, T, N), (
        f'mask shape mismatch: {masks_seq.shape} vs expected {(B, T, N)}')
    t = int(min(max(t_probe, 0), T - 1))
    t_end = int(min(t + horizon, T))

    valid_t = masks_seq[:, t, :] > 0.5                       # [B, N]
    states_t = states_seq[:, t, :, :]                        # [B, N, 35]

    # Velocity from the canonical 35-dim state layout (slice 3:6).
    velocity_all = states_t[..., 3:6].astype(np.float32)     # [B, N, 3]
    speed_all = np.linalg.norm(velocity_all, axis=-1)        # [B, N]

    # Future-collision per object: for each (b, n), the minimum distance from
    # object n to any OTHER valid object across [t, t_end). Vectorised.
    pos = states_seq[:, t:t_end, :, 0:3].astype(np.float32)  # [B, t_h, N, 3]
    msk_h = masks_seq[:, t:t_end, :].astype(np.float32)      # [B, t_h, N]
    # diff[b, k, i, j, :] = pos[b, k, i] - pos[b, k, j]
    diff = pos[:, :, :, None, :] - pos[:, :, None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1) + 1e-8)        # [B, t_h, N, N]
    pair_mask = msk_h[:, :, :, None] * msk_h[:, :, None, :]  # [B, t_h, N, N]
    eye = np.eye(N, dtype=np.float32)
    pair_mask = pair_mask * (1.0 - eye)
    big = 1e6
    dist = dist * pair_mask + big * (1.0 - pair_mask)
    # min over (time, other_object) -> [B, N]
    min_dist_per_obj = dist.min(axis=1).min(axis=-1)
    future_collision_all = (min_dist_per_obj < collision_threshold).astype(np.int32)

    # Exits-frame per object: was visible at t (we restrict to those rows
    # below), and at any frame in [t+1, t_end) is not visible.
    if t_end > t + 1:
        future_mask = msk_h[:, 1:, :] > 0.5                  # [B, t_h-1, N]
        # Object exits if any future frame in window has mask 0.
        exits_all = ((future_mask == 0).any(axis=1)).astype(np.int32)
    else:
        exits_all = np.zeros((B, N), dtype=np.int32)

    # Filter to valid-at-t rows. Order: row-major over (B, N), keeping b in
    # the outer loop and n in the inner loop. This MUST match the feature
    # extractor's iteration order.
    velocity_xyz = velocity_all[valid_t]                     # [n_obj, 3]
    speed = speed_all[valid_t]                               # [n_obj]
    is_moving = (speed > speed_threshold).astype(np.int32)
    future_collision = future_collision_all[valid_t]
    exits_frame = exits_all[valid_t]
    scene_index = np.repeat(np.arange(B), valid_t.sum(axis=1)).astype(np.int64)

    return {
        'velocity_xyz': velocity_xyz,
        'speed': speed.astype(np.float32),
        'is_moving': is_moving,
        'future_collision': future_collision,
        'exits_frame': exits_frame,
        'scene_index': scene_index,
    }


# ---------------------------------------------------------------------------
# Feature extraction at three layers (input_per_obj, encoder_per_obj,
# prefix_per_obj). Mirrors encoder_ood_probing._extract_layer_features but
# adds prefix_per_obj for the V3 / Phase-6 per-object prefix path.
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_per_object_features(
    adapter,
    states_np: np.ndarray,    # [B, N, 35] (single frame slice)
    masks_np: np.ndarray,     # [B, N]
    batch_size: int = 64,
    device: str = 'cuda',
) -> Dict[str, np.ndarray]:
    """Run encoder + adapter and return per-object features at 3 layers.

    Returns dict with arrays shaped [n_obj, dim] where n_obj is the total
    number of valid objects across the batch (one row per valid object).
    The row order is row-major (b, n), matching ``per_object_dynamics_labels``.

    Layers:
      input_per_obj   : raw 35-dim state slice for valid objects.
      encoder_per_obj : per-object encoder output (D = adapter hidden dim).
      prefix_per_obj  : per-object prefix vector (K * llm_dim) -- the
                        actual tokens that go to the LLM in Phase 6.
                        Only present when adapter.tokens_per_object > 0.
                        For V2 / Phase-3 ckpts (scene-level prefix) we
                        skip this layer (returns no key).
    """
    target_N = adapter.physics_model.max_objects
    states_np, masks_np = _ensure_max_objects(states_np, masks_np, target_N)
    K = int(getattr(adapter, 'tokens_per_object', 0) or 0)

    B = states_np.shape[0]
    in_chunks: List[np.ndarray] = []
    enc_chunks: List[np.ndarray] = []
    prefix_chunks: List[np.ndarray] = []
    for i in range(0, B, batch_size):
        states = torch.from_numpy(states_np[i:i + batch_size]).float().to(device)
        masks = torch.from_numpy(masks_np[i:i + batch_size]).float().to(device)

        obj_emb, flat_mask = adapter.extract_object_embeddings(states, masks)
        if flat_mask is None:
            flat_mask = masks
        valid_b = (flat_mask.cpu().numpy() > 0.5)            # [b, N]
        states_b = states.cpu().numpy()
        obj_emb_b = obj_emb.cpu().numpy()
        in_chunks.append(states_b[valid_b])
        enc_chunks.append(obj_emb_b[valid_b])

        if K > 0:
            # prefix_tokens: [b, N*K, llm_dim]. Reshape to [b, N, K, llm_dim]
            # so we can pull per-object slots, then flatten to [n_obj, K*llm_dim].
            prefix_tokens, _ = adapter.create_prefix_tokens_per_object(states, masks)
            b, NK, llm_dim = prefix_tokens.shape
            assert NK == target_N * K, (
                f'prefix shape mismatch: got {NK} expected {target_N * K}')
            per_obj_prefix = prefix_tokens.view(b, target_N, K, llm_dim).reshape(
                b, target_N, K * llm_dim)
            per_obj_prefix_np = per_obj_prefix.cpu().numpy()
            prefix_chunks.append(per_obj_prefix_np[valid_b])

    out: Dict[str, np.ndarray] = {
        'input_per_obj':   np.concatenate(in_chunks, axis=0),
        'encoder_per_obj': np.concatenate(enc_chunks, axis=0),
    }
    if prefix_chunks:
        out['prefix_per_obj'] = np.concatenate(prefix_chunks, axis=0)
    return out


# ---------------------------------------------------------------------------
# Main probing loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Per-object DYNAMICS probe of the encoder + Phase-6 prefix.')
    parser.add_argument('--n', type=int, default=1500,
                        help='Number of CLEVRER scenes to sample.')
    parser.add_argument('--single_frame', type=int, default=64,
                        help='Frame at which to extract features (canonical 64).')
    parser.add_argument('--horizon', type=int, default=24,
                        help='Future-window length for future_collision and '
                             'exits_frame labels (default 24 = 1 sec @ 24 fps).')
    parser.add_argument('--collision_threshold', type=float, default=0.7,
                        help='Pairwise-distance threshold (scene units) used '
                             'for the future_collision label.')
    parser.add_argument('--speed_threshold', type=float, default=0.1,
                        help='|v| threshold for the is_moving binary label.')
    parser.add_argument('--clevrer_h5', type=Path, default=DEFAULT_CLEVRER_H5)
    parser.add_argument('--adapter_ckpt', type=Path, default=DEFAULT_ADAPTER_CKPT)
    parser.add_argument('--physics_ckpt', type=Path, default=DEFAULT_PHYSICS_CKPT)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'[setup] device={device}')

    # 1. Load adapter (V2 or V3) + frozen encoder.
    print(f'[load] adapter: {args.adapter_ckpt}')
    print(f'[load] physics: {args.physics_ckpt}')
    adapter = load_encoder_adapter(args.adapter_ckpt, args.physics_ckpt, device)
    K = int(getattr(adapter, 'tokens_per_object', 0) or 0)
    print(f'[load] tokens_per_object={K}, '
          f'max_objects={adapter.physics_model.max_objects}, '
          f'hidden_dim={adapter.physics_model.hidden_dim}')
    if K == 0:
        print('[warn] adapter has no per-object prefix path '
              '(tokens_per_object=0). prefix_per_obj will be omitted; '
              'use a Phase-6 / V3 checkpoint to probe the per-object prefix.')

    # 2. Sample CLEVRER scenes (full sequences for label computation).
    print(f'[load] CLEVRER: n={args.n} (full sequence)')
    t0 = time.time()
    states_seq, masks_seq, scene_ids = load_clevrer_sequence_subset(
        args.clevrer_h5, args.n, seed=args.seed)
    print(f'        states_seq {states_seq.shape}, '
          f'masks_seq {masks_seq.shape}, '
          f'{time.time() - t0:.1f}s')

    # 3. Build labels.
    t0 = time.time()
    labels = per_object_dynamics_labels(
        states_seq, masks_seq,
        t_probe=args.single_frame,
        horizon=args.horizon,
        collision_threshold=args.collision_threshold,
        speed_threshold=args.speed_threshold,
    )
    n_obj = labels['speed'].shape[0]
    print(f'[labels] n_objects={n_obj}, '
          f'speed mean={labels["speed"].mean():.3f}, '
          f'is_moving rate={labels["is_moving"].mean():.3f}, '
          f'future_collision rate={labels["future_collision"].mean():.3f}, '
          f'exits_frame rate={labels["exits_frame"].mean():.3f}, '
          f'{time.time() - t0:.1f}s')

    # 4. Extract features at the probe frame.
    t0 = time.time()
    states_t = states_seq[:, min(args.single_frame, states_seq.shape[1] - 1), :, :]
    masks_t = masks_seq[:, min(args.single_frame, masks_seq.shape[1] - 1), :]
    feats = extract_per_object_features(
        adapter, states_t, masks_t,
        batch_size=args.batch_size, device=device,
    )
    for k, v in feats.items():
        print(f'        {k}: {v.shape}')
    # Sanity: row counts must agree with labels.
    for k, v in feats.items():
        assert v.shape[0] == n_obj, (
            f'row count mismatch at {k}: {v.shape[0]} vs labels {n_obj}')
    print(f'        {time.time() - t0:.1f}s')

    # 5. Probe each layer x dynamics target.
    print()
    print('=' * 88)
    print('Per-object DYNAMICS probes (CLEVRER)')
    print('=' * 88)

    layers = [layer for layer in
              ('input_per_obj', 'encoder_per_obj', 'prefix_per_obj')
              if layer in feats]

    # Targets and probe types.
    targets = [
        ('velocity_xyz',       'reg', 'r2'),
        ('speed',              'reg', 'r2'),
        ('is_moving',          'clf', 'auc'),
        ('future_collision',   'clf', 'auc'),
        ('exits_frame',        'clf', 'auc'),
    ]

    results: Dict[str, Dict[str, Dict]] = {}
    for layer in layers:
        X = feats[layer]
        results[layer] = {}
        print(f'\n--- {layer} (dim={X.shape[1]}, n_obj={X.shape[0]}) ---')
        for target, kind, key in targets:
            y = labels[target]
            X_tr, X_te, y_tr, y_te = _split_train_test(X, y, seed=args.seed)
            if kind == 'reg':
                res = _fit_regression_probe(X_tr, y_tr, X_te, y_te)
                primary = res['r2']
                msg = f'R^2 = {primary:.4f}'
                if 'r2_per_axis' in res:
                    msg += f'  (per-axis: {[round(x, 3) for x in res["r2_per_axis"]]})'
            else:
                res = _fit_classification_probe(X_tr, y_tr, X_te, y_te)
                primary = res.get('auc', float('nan'))
                msg = (f'AUC = {primary:.4f}'
                       f'  acc = {res["acc"]:.4f}'
                       f'  (pos rate train={res["class_balance_train"]:.3f}'
                       f', test={res["class_balance_test"]:.3f})')
            results[layer][target] = res
            print(f'  {target:<22}  {msg}')

    # 6. Bottleneck summary table: input -> encoder -> prefix delta.
    print()
    print('=' * 88)
    print('Bottleneck table: where does dynamics signal degrade?')
    print('=' * 88)
    header = f'{"target":<22} {"input":>9} {"encoder":>9} {"prefix":>9} {"enc-in":>9} {"pre-enc":>9}'
    print(header)
    print('-' * len(header))
    for target, kind, key in targets:
        score_in = results.get('input_per_obj', {}).get(target, {}).get(key)
        score_en = results.get('encoder_per_obj', {}).get(target, {}).get(key)
        score_pr = results.get('prefix_per_obj', {}).get(target, {}).get(key)

        def fmt(v):
            return f'{v:9.4f}' if isinstance(v, (int, float)) and v == v else f'{"----":>9}'

        if (isinstance(score_in, float) and score_in == score_in
                and isinstance(score_en, float) and score_en == score_en):
            d_en_in = score_en - score_in
        else:
            d_en_in = None
        if (isinstance(score_en, float) and score_en == score_en
                and isinstance(score_pr, float) and score_pr == score_pr):
            d_pr_en = score_pr - score_en
        else:
            d_pr_en = None

        row = (f'{target:<22} {fmt(score_in)} {fmt(score_en)} {fmt(score_pr)} '
               f'{fmt(d_en_in)} {fmt(d_pr_en)}')
        print(row)

    print()
    print('Reading guide: enc-in < 0 means the encoder DESTROYS that signal '
          '(input had it, encoder loses it). pre-enc < 0 means the adapter '
          'MLP further degrades it. Closer to 0 = better preservation.')

    # 7. Write JSON.
    summary = {
        'config': {
            'n_scenes': args.n,
            'single_frame': args.single_frame,
            'horizon': args.horizon,
            'collision_threshold': args.collision_threshold,
            'speed_threshold': args.speed_threshold,
            'clevrer_h5': str(args.clevrer_h5),
            'adapter_ckpt': str(args.adapter_ckpt),
            'physics_ckpt': str(args.physics_ckpt),
            'tokens_per_object': K,
            'seed': args.seed,
            'device': device,
        },
        'features': {
            'layers': layers,
            'feature_dims': {layer: int(feats[layer].shape[1]) for layer in layers},
            'n_objects': int(n_obj),
        },
        'targets': [t[0] for t in targets],
        'results': results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\n  full summary written to: {args.out}')


if __name__ == '__main__':
    main()
