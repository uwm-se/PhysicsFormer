"""Linear-probe diagnostic: how much physics info survives into the prefix
the LLM actually sees?

For each held-out scene we extract three per-object representations:

  (a) raw  -- the 35-dim state vector at frame 0 (sanity floor; should
              recover everything trivially via a linear map).
  (b) enc  -- ``adapter.extract_object_embeddings(states, mask)`` --
              the physics-encoder output the adapter consumes.
  (c) mlp  -- ``adapter.create_prefix_tokens_per_object(states, mask)``
              averaged over the K tokens emitted per object. This is
              what the LLM-facing prefix looks like AFTER the adapter
              MLP projection into llm_dim space.

For each representation we fit a linear probe (sklearn) against four
labels that the encoder either trivially passes through (the input
already encodes them) or has to integrate temporally to recover:

  speed     -- ||v_xy|| at frame 0; linear regression, scored by R^2.
  shape     -- 3-way categorical {sphere,cube,cylinder}; logistic, acc.
  color     -- 8-way categorical (CLEVRER palette); logistic, acc.
  material  -- 2-way {rubber,metal} (mass proxy); logistic, acc.

The probe is a sanity check on the architecture, not on the LoRA
adapter that sits on top. The interesting comparison is enc -> mlp:
a large drop means the per-object adapter MLP is the information
bottleneck, and no amount of training-pool diversity will fix the
LLM's grounding because the LLM never sees that information.

Usage::

    python compsac_2026_code/clevrer_benchmark/scripts/probe_prefix_grounding.py \\
        --adapter_checkpoint compsac_2026_code/checkpoints/adapter_phase10_format_diverse.pt \\
        --physics_checkpoint compsac_2026_code/checkpoints/physics_former_best.pt \\
        --clevrer_dir $CLEVRER_DIR \\
        --n_scenes 300 \\
        --out compsac_2026_code/clevrer_benchmark/results/phase10_format_diverse/probe_b/prefix_probe.json
"""
from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _SCRIPT_DIR.parent
_SNAPSHOT_ROOT = _BENCH_DIR.parent
sys.path.insert(0, str(_SNAPSHOT_ROOT))
sys.path.insert(0, str(_BENCH_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

from run_adapter_evaluation import load_adapter_model  # noqa: E402
from clevrer_benchmark.scene_converter import (  # noqa: E402
    SHAPE_MAP, COLOR_MAP, MATERIAL_MAP,
    clevrer_scene_to_state_tensor, load_clevrer_scene,
)
from physics_llm_adapter.phase9_splits import load_heldout_scene_set  # noqa: E402

COLOR_NAMES = list(COLOR_MAP.keys())
COLOR_TO_IDX = {c: i for i, c in enumerate(COLOR_NAMES)}
MAT_TO_IDX = {'rubber': 0, 'metal': 1}


def _resolve_scene_dir(clevrer_dir: Path) -> Path:
    for c in (clevrer_dir / 'scenes' / 'clevrer_scenes',
              clevrer_dir / 'scenes' / 'validation',
              clevrer_dir / 'scenes'):
        if c.exists() and any(c.glob('annotation_*.json')):
            return c
    raise FileNotFoundError(f'No CLEVRER scenes under {clevrer_dir}')


def _collect_features_and_labels(
    adapter,
    clevrer_dir: Path,
    heldout: set,
    n_scenes: int,
    device: str,
) -> Dict[str, np.ndarray]:
    """Walk N held-out scenes; emit one row per real object slot."""
    scene_dir = _resolve_scene_dir(clevrer_dir)
    paths = []
    for p in sorted(scene_dir.glob('annotation_*.json')):
        try:
            num = int(p.stem.split('_')[-1])
        except ValueError:
            continue
        if num in heldout:
            paths.append(p)
        if len(paths) >= n_scenes:
            break
    if not paths:
        raise RuntimeError('no held-out scenes resolved')

    raw_rows: List[np.ndarray] = []
    enc_rows: List[np.ndarray] = []
    mlp_rows: List[np.ndarray] = []
    speed_labels: List[float] = []
    shape_labels: List[int] = []
    color_labels: List[int] = []
    mat_labels: List[int] = []

    adapter.eval()
    for p in tqdm(paths, desc='probe', unit='scene'):
        try:
            scene = load_clevrer_scene(str(p))
            states_np, masks_np, meta = clevrer_scene_to_state_tensor(scene)
        except Exception:
            continue

        if states_np.shape[0] < 2:
            continue

        # Single-frame state slice (frame 0) matches what the adapter
        # sees for non-temporal eval paths; we still derive speed from
        # frames 0 -> 1 because frame-0 velocity is zero by convention.
        f0 = states_np[0]                                       # [N, 35]
        f1 = states_np[1] if states_np.shape[0] > 1 else f0     # [N, 35]
        mask_union = (masks_np.max(axis=0) > 0.5).astype('float32')

        states_t = torch.from_numpy(f0).float().unsqueeze(0).to(device)   # [1, N, 35]
        mask_t = torch.from_numpy(mask_union).float().unsqueeze(0).to(device)  # [1, N]

        with torch.no_grad():
            obj_emb, flat_mask = adapter.extract_object_embeddings(
                states_t, mask_t,
            )  # [1, N, D_enc], [1, N]
            prefix_tok, _ = adapter.create_prefix_tokens_per_object(
                states_t, mask_t,
            )  # [1, N*K, llm_dim]
        K = adapter.tokens_per_object
        N = obj_emb.shape[1]
        per_obj_prefix = prefix_tok.view(1, N, K, -1).mean(dim=2)  # avg K tokens
        # Per-object features at frame 0
        obj_emb_np = obj_emb[0].cpu().float().numpy()             # [N, D_enc]
        per_obj_prefix_np = per_obj_prefix[0].cpu().float().numpy()  # [N, llm_dim]

        objects_meta = meta.get('objects', [])
        for i in range(N):
            if mask_union[i] < 0.5:
                continue
            if i >= len(objects_meta):
                continue
            obj = objects_meta[i]
            # speed from frame 0 -> 1 displacement (m/s; scene_converter
            # multiplies by FPS in derive_velocity, but we recompute here
            # in unit-frames to keep the probe scale-invariant).
            dpos = f1[i, 0:3] - f0[i, 0:3]
            speed = float(np.linalg.norm(dpos[:2]))  # xy plane only
            shape_idx = SHAPE_MAP.get(obj.get('shape', ''), -1)
            color_idx = COLOR_TO_IDX.get(obj.get('color', ''), -1)
            mat_idx = MAT_TO_IDX.get(obj.get('material', ''), -1)
            if shape_idx < 0 or color_idx < 0 or mat_idx < 0:
                continue

            raw_rows.append(f0[i].astype(np.float32))
            enc_rows.append(obj_emb_np[i].astype(np.float32))
            mlp_rows.append(per_obj_prefix_np[i].astype(np.float32))
            speed_labels.append(speed)
            shape_labels.append(shape_idx)
            color_labels.append(color_idx)
            mat_labels.append(mat_idx)

    return {
        'raw': np.stack(raw_rows),
        'enc': np.stack(enc_rows),
        'mlp': np.stack(mlp_rows),
        'speed': np.asarray(speed_labels, dtype=np.float32),
        'shape': np.asarray(shape_labels, dtype=np.int64),
        'color': np.asarray(color_labels, dtype=np.int64),
        'material': np.asarray(mat_labels, dtype=np.int64),
    }


def _split(n: int, seed: int, train_frac: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * train_frac)
    return idx[:cut], idx[cut:]


def _probe_continuous(X_tr, y_tr, X_te, y_te) -> Dict[str, float]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        'r2': float(r2_score(y_te, pred)),
        'mae': float(np.mean(np.abs(y_te - pred))),
        'y_var': float(np.var(y_te)),
    }


def _probe_categorical(X_tr, y_tr, X_te, y_te, n_classes: int) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    model = LogisticRegression(
        max_iter=2000, multi_class='auto', solver='lbfgs', C=1.0,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        'acc': float(accuracy_score(y_te, pred)),
        'chance': 1.0 / n_classes,
        'support_test': int(len(y_te)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    ap.add_argument('--adapter_checkpoint', type=Path, required=True)
    ap.add_argument('--physics_checkpoint', type=str,
                    default=str(_SNAPSHOT_ROOT / 'checkpoints' / 'physics_former_best.pt'))
    ap.add_argument('--clevrer_dir', type=Path, default=Path(os.environ.get('CLEVRER_DIR', 'clevrer')))
    ap.add_argument('--n_scenes', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    device = 'cuda' if (args.device != 'cpu' and torch.cuda.is_available()) else 'cpu'
    print(f'[probe] loading adapter from {args.adapter_checkpoint} (device={device})')
    adapter = load_adapter_model(
        args.adapter_checkpoint, args.physics_checkpoint, device=device,
    )

    heldout = load_heldout_scene_set()
    print(f'[probe] {len(heldout)} held-out scene ids; sampling up to {args.n_scenes}')

    feats = _collect_features_and_labels(
        adapter, args.clevrer_dir, heldout, args.n_scenes, device,
    )
    n = len(feats['speed'])
    print(f'[probe] collected {n} per-object rows; '
          f'raw_dim={feats["raw"].shape[1]} '
          f'enc_dim={feats["enc"].shape[1]} '
          f'mlp_dim={feats["mlp"].shape[1]}')

    tr_idx, te_idx = _split(n, seed=args.seed)
    results: Dict[str, Dict] = {}
    for rep in ('raw', 'enc', 'mlp'):
        X = feats[rep]
        X_tr, X_te = X[tr_idx], X[te_idx]
        results[rep] = {
            'speed': _probe_continuous(X_tr, feats['speed'][tr_idx],
                                       X_te, feats['speed'][te_idx]),
            'shape': _probe_categorical(X_tr, feats['shape'][tr_idx],
                                        X_te, feats['shape'][te_idx],
                                        n_classes=3),
            'color': _probe_categorical(X_tr, feats['color'][tr_idx],
                                        X_te, feats['color'][te_idx],
                                        n_classes=len(COLOR_NAMES)),
            'material': _probe_categorical(X_tr, feats['material'][tr_idx],
                                           X_te, feats['material'][te_idx],
                                           n_classes=2),
        }

    summary = {
        'config': {
            'adapter_checkpoint': str(args.adapter_checkpoint),
            'physics_checkpoint': args.physics_checkpoint,
            'n_scenes_target': args.n_scenes,
            'n_objects': n,
            'n_train': int(len(tr_idx)),
            'n_test': int(len(te_idx)),
            'seed': args.seed,
            'feature_dims': {
                'raw': int(feats['raw'].shape[1]),
                'enc': int(feats['enc'].shape[1]),
                'mlp': int(feats['mlp'].shape[1]),
            },
        },
        'results': results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'[probe] saved {args.out}')

    # One-line readouts.
    print('\n' + '=' * 70)
    print('Linear-probe summary (test split)')
    print('=' * 70)
    print(f"  {'task':10s} {'raw':>10s} {'enc':>10s} {'mlp':>10s}")
    for task in ('speed', 'shape', 'color', 'material'):
        row = []
        for rep in ('raw', 'enc', 'mlp'):
            r = results[rep][task]
            if 'r2' in r:
                row.append(f"R2={r['r2']:+.3f}")
            else:
                row.append(f"acc={r['acc']:.3f}")
        print(f"  {task:10s} {row[0]:>10s} {row[1]:>10s} {row[2]:>10s}")
    print()


if __name__ == '__main__':
    main()
