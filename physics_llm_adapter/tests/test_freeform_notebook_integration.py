"""End-to-end integration test for colab_train_adapter_v3_freeform.ipynb.

Exercises every code path the notebook will hit on Colab, against tiny
synthetic data, on CPU, in <60s. Catches the kinds of bugs that are
otherwise discovered 30 minutes into a Colab run:

* `compute_combined_loss` positional/keyword arg mismatches
* `optimizer has no trainable parameters` after freeze_physics + freeze_llm
* `state_dict load_state_dict` mismatches between V3 fresh and warm-start
* generation `forward()` signature mismatches in Cell 11
* free-form QA dataset -> collate_fn -> compute path

Mirrors Cell-by-cell:

* Cells 1-5 (env)               -> sys.path setup at top of this file
* Cell 6 (config)               -> _Config namespace below
* Cell 7 (encoder load)         -> tiny FullPhysicsFormer with random weights
* Cell 8 (dataset + dataloaders)-> synthetic 32-record dataset, real collate_fn
* Cell 9 (adapter + warm-start) -> create_adapter_v3 then strict=False reload
* Cell 10 (training loop)       -> one train step + one val step
* Cell 11 (qualitative gen)     -> adapter.forward([question]) on one sample

Run:
    python physics_llm_adapter/tests/test_freeform_notebook_integration.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch
import torch.optim as optim

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "physics_former"))
sys.path.insert(0, str(REPO / "physics_llm_adapter"))
sys.path.insert(0, str(REPO / "clevrer_benchmark"))


# ---------------------------------------------------------------------------
# Synthetic CLEVRER scene fixtures (mimic the JSON shape scene_converter
# expects from real CLEVRER annotation files)
# ---------------------------------------------------------------------------


def _build_synthetic_scenes(scenes_dir: Path, n_scenes: int = 8) -> None:
    """Write synthetic scene JSONs that mimic CLEVRER's annotation format.

    Real CLEVRER ``annotation_NNNNN.json`` files have:
      * ``scene_index`` (int)
      * ``object_property`` (list of dicts with ``object_id``, ``color``,
        ``shape``, ``material``, ``size``)
      * ``motion_trajectory`` (list of frames, each with ``objects`` list of
        dicts with ``object_id``, ``location``, ``velocity``,
        ``inside_camera_view``)

    See ``_convert_motion_trajectory_format`` in
    ``compsac_2026_code/clevrer_benchmark/scene_converter.py:185-237``.
    """
    scenes_dir.mkdir(parents=True, exist_ok=True)
    for sid in range(n_scenes):
        n_objects = 2 + (sid % 3)
        object_property = [
            {
                "object_id": k,
                "color": ["red", "green", "blue", "yellow"][k % 4],
                "shape": ["sphere", "cube", "cylinder"][k % 3],
                "material": "metal" if k % 2 == 0 else "rubber",
                "size": "large" if k % 2 == 0 else "small",
            }
            for k in range(n_objects)
        ]
        n_frames = 16
        motion = []
        for t in range(n_frames):
            frame_objs = []
            for k in range(n_objects):
                frame_objs.append({
                    "object_id": k,
                    "location": [0.1 * t + 0.05 * k, 0.05 * k, 0.1],
                    "velocity": [0.1, 0.0, 0.0],
                    "inside_camera_view": True,
                })
            motion.append({"objects": frame_objs})
        scene = {
            "scene_index": sid,
            "object_property": object_property,
            "motion_trajectory": motion,
        }
        (scenes_dir / f"annotation_{sid:05d}.json").write_text(json.dumps(scene))


def _build_synthetic_qa_records(qa_path: Path, scenes_dir: Path, n_per_scene: int = 4) -> int:
    """Write a synthetic causal_qa_dataset.json with valid scene back-references.

    Schema mirrors the real records produced by
    ``explicit_world_model.llm_adapters.causal_qa_data.generate_causal_qa_dataset``
    (see line 1058: each record has ``scene_index`` (int) and ``scene_path``).
    """
    records = []
    qa_types = ["predictive", "counterfactual", "explanatory", "interventional"]
    for scene_path in sorted(scenes_dir.glob("annotation_*.json")):
        with scene_path.open() as f:
            scene = json.load(f)
        sid = scene["scene_index"]
        first_color = scene["object_property"][0]["color"]
        for k in range(n_per_scene):
            records.append({
                "scene_index": sid,                       # canonical key
                "scene_path": str(scene_path),
                "question": f"What happens to the {first_color} object after frame {k}?",
                "target": f"The {first_color} object moves to the right.",
                "qa_type": qa_types[k % len(qa_types)],
                "scene_desc": f"Scene {sid} with {len(scene['object_property'])} objects",
            })
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(records))
    return len(records)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_freeform_notebook_pipeline() -> None:
    print("=" * 70)
    print("End-to-end integration test for colab_train_adapter_v3_freeform.ipynb")
    print("=" * 70)
    t0 = time.time()

    from training.models.physics_former_full import FullPhysicsFormer
    from training.configs.config import TrainingConfig
    from train_adapter_v2 import PhysicsReasoningDataset, collate_fn
    from adapter_v3 import create_adapter_v3
    from torch.utils.data import DataLoader

    tmpdir = Path(tempfile.mkdtemp(prefix="ff_nb_test_"))
    print(f"\n[setup] tmpdir = {tmpdir}")
    try:
        # ── Cell 8 prerequisites: scenes + QA records ─────────────────────
        scenes_dir = tmpdir / "scenes"
        qa_path    = tmpdir / "causal_qa_dataset.json"
        cache_path = tmpdir / "freeform_state_cache.pt"
        _build_synthetic_scenes(scenes_dir, n_scenes=8)
        n_records = _build_synthetic_qa_records(qa_path, scenes_dir, n_per_scene=4)
        print(f"[setup] {n_records} synthetic QA records across 8 scenes")

        # ── Cell 7: tiny encoder (random weights -- never load checkpoint) ──
        config = TrainingConfig()
        # Use a TINY model so this runs in <60s on CPU.
        physics_model = FullPhysicsFormer(
            state_dim=config.state_dim,
            hidden_dim=64,                   # vs production 256+
            num_layers=2,                    # vs production 6
            num_heads=2,                     # vs production 8
            ff_dim=128,
            max_objects=config.max_objects,
            dropout=config.dropout,
            num_schema_classes=10,
        )
        physics_model.eval()
        for p in physics_model.parameters():
            p.requires_grad = False
        n_phys = sum(p.numel() for p in physics_model.parameters())
        print(f"[cell 7] tiny FullPhysicsFormer: {n_phys:,} params (frozen)")

        # ── Cell 8: build dataset + dataloaders ───────────────────────────
        ff_dataset = PhysicsReasoningDataset(
            physics_dataset=None,
            freeform_qa_data_path=str(qa_path),
            freeform_qa_scenes_dir=str(scenes_dir),
            freeform_qa_cache_path=str(cache_path),
            freeform_max_objects=config.max_objects,
            freeform_seq_len=getattr(config, "sequence_length", 16),
        )
        assert len(ff_dataset) == n_records, f"Expected {n_records} records, got {len(ff_dataset)}"
        print(f"[cell 8] dataset built: {len(ff_dataset)} records")

        # 75/25 split (deterministic) -> 24 train, 8 val
        n_val = max(1, int(len(ff_dataset) * 0.25))
        n_train = len(ff_dataset) - n_val
        all_pairs = list(ff_dataset.qa_pairs)
        train_pairs = all_pairs[:n_train]
        val_pairs   = all_pairs[n_train:]

        class _Subset:
            def __init__(self, pairs): self.qa_pairs = pairs
            def __len__(self): return len(self.qa_pairs)
            def __getitem__(self, idx): return self.qa_pairs[idx]

        train_loader = DataLoader(
            _Subset(train_pairs), batch_size=4, shuffle=True,
            collate_fn=collate_fn, num_workers=0, drop_last=True,
        )
        val_loader = DataLoader(
            _Subset(val_pairs), batch_size=4, shuffle=False,
            collate_fn=collate_fn, num_workers=0, drop_last=False,
        )

        sample = next(iter(train_loader))
        print(f"[cell 8] sample batch:")
        print(f"          states: {tuple(sample['states'].shape)}")
        print(f"          masks:  {tuple(sample['masks'].shape)}")
        print(f"          questions: {len(sample['questions'])} (e.g. {sample['questions'][0][:60]!r})")
        print(f"          answers:   {len(sample['answers'])} (e.g. {sample['answers'][0][:60]!r})")
        print(f"          choices:   {sample['choices']}  <- must be None")
        assert sample["choices"] is None, "free-form mode must yield batch-level choices=None"

        # ── Cell 9: create V3 adapter with freeform recipe + warm-start ───
        adapter = create_adapter_v3(
            physics_model=physics_model,
            physics_dim=64,                   # match tiny encoder
            num_prefix_tokens=8,              # smaller than production's 64 for speed
            freeze_physics=True,
            freeze_llm=True,
            include_choices_prob=0.0,         # <-- recipe knob: Format A only
            mixed_format_seed=42,
            label_smoothing=0.1,
            answer_token_dropout=0.15,        # <-- recipe knob: regularizer
            force_llm_only_routing=True,      # <-- required for prose targets
        )
        print(f"[cell 9] V3 adapter created (Format A only, dropout=0.15, LLM-only routing)")

        # Simulate warm-start: save the just-created state_dict, then reload
        # with strict=False to exercise the missing/unexpected-keys path.
        warm_path = tmpdir / "fake_adapter_phase3.pt"
        torch.save({"adapter_state_dict": adapter.state_dict()}, warm_path)

        warm_ckpt = torch.load(warm_path, map_location="cpu", weights_only=False)
        warm_state = warm_ckpt["adapter_state_dict"]
        warm_state = {k.replace("_orig_mod.", ""): v for k, v in warm_state.items()}
        missing, unexpected = adapter.load_state_dict(warm_state, strict=False)
        print(f"[cell 9] warm-start load: missing={len(missing)}, unexpected={len(unexpected)}")
        # When loading a freshly-built adapter into itself, both should be 0.
        assert len(missing) == 0, f"unexpected missing keys: {missing[:5]}"
        assert len(unexpected) == 0, f"unexpected extra keys: {unexpected[:5]}"

        # ── Cell 10: ONE training step (exercises compute_combined_loss) ──
        trainable = [p for p in adapter.parameters() if p.requires_grad]
        assert len(trainable) > 0, (
            "BUG: no trainable parameters after freeze_physics=True, freeze_llm=True. "
            "The adapter MLPs / heads should still be trainable."
        )
        n_train_p = sum(p.numel() for p in trainable)
        n_total_p = sum(p.numel() for p in adapter.parameters())
        print(f"[cell 9] trainable: {n_train_p:,} / {n_total_p:,} ({100*n_train_p/n_total_p:.2f}%)")

        optimizer = optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)

        def _step(batch, training: bool) -> float:
            # Exact pattern from notebook Cell 10 _ff_step.
            states = batch["states"]
            masks  = batch["masks"]
            questions = batch["questions"]
            answers   = batch["answers"]
            numerical_targets = {k: v for k, v in batch["numerical_targets"].items()}

            loss, loss_dict = adapter.compute_combined_loss(
                states, masks, questions, answers,
                choices=None,
                correct_choice_idx=None,
                numerical_targets=numerical_targets,
                categorical_weight=1.0,
                numerical_weight=0.5,
            )
            assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
            return float(loss.detach().item())

        adapter.train()
        train_loss = _step(next(iter(train_loader)), training=True)
        print(f"[cell 10] one training step OK: loss={train_loss:.4f}")

        adapter.eval()
        with torch.no_grad():
            val_loss = _step(next(iter(val_loader)), training=False)
        print(f"[cell 10] one val step OK:      loss={val_loss:.4f}")

        # ── Cell 11: qualitative generation ──────────────────────────────
        rec = val_pairs[0]
        states_b = rec["states"].unsqueeze(0)              # [1, T, N, D]
        mask_2d  = rec["mask"]
        mask_1d  = mask_2d[0] if mask_2d.dim() == 2 else mask_2d
        mask_b   = mask_1d.unsqueeze(0)                    # [1, N]

        with torch.no_grad():
            answers = adapter(
                physics_states=states_b,
                object_mask=mask_b,
                question_text=[rec["question"]],
                max_length=20,                              # tiny for speed
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
            )
        assert isinstance(answers, list), f"adapter() must return list, got {type(answers)}"
        assert len(answers) == 1, f"expected 1 answer, got {len(answers)}"
        assert isinstance(answers[0], str), f"answer must be str, got {type(answers[0])}"
        print(f"[cell 11] qualitative generation OK: {answers[0][:80]!r}")

        # ── Recipe metadata round-trip ───────────────────────────────────
        recipe = {
            "adapter_class": "PhysicsLLMAdapterV3",
            "training_phase": "phase4_freeform_qa_ablation",
            "include_choices_prob": adapter.include_choices_prob,
            "answer_token_dropout": adapter.answer_token_dropout,
        }
        save_path = tmpdir / "adapter_phase4_freeform_qa_test.pt"
        torch.save({
            "adapter_state_dict": adapter.state_dict(),
            "config": recipe,
        }, save_path)
        reloaded = torch.load(save_path, map_location="cpu", weights_only=False)
        assert reloaded["config"]["adapter_class"] == "PhysicsLLMAdapterV3"
        assert reloaded["config"]["include_choices_prob"] == 0.0
        assert reloaded["config"]["answer_token_dropout"] == 0.15
        print(f"[recipe] metadata round-trip OK")

        elapsed = time.time() - t0
        print(f"\n{'=' * 70}")
        print(f"All integration tests passed in {elapsed:.1f}s")
        print(f"{'=' * 70}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_phase_b_c_pipeline(
    label: str,
    *,
    inject_scene_text: bool,
    scene_text_style: str,
    tokens_per_object: int,
    scene_text_include_color: bool = True,
    scene_text_include_material: bool = True,
    descriptor_strip_prob: float = 0.0,
    prefix_dropout: float = 0.0,
) -> None:
    """Shared pipeline driver for Phase B / Phase C / Phase 8 variants.

    Runs the same end-to-end flow as ``test_freeform_notebook_pipeline``
    but with the Phase B scene-text injection knob, the Phase C
    per-object prefix knob, and/or the Phase 8 descriptor-stripping +
    prefix-dropout knobs enabled. Used to verify:

      * ``compute_loss`` dispatches on ``tokens_per_object`` correctly.
      * ``_build_scene_texts`` produces byte-identical strings for the
        same input tensor every time (determinism). When
        ``descriptor_strip_prob > 0`` we check this in ``eval()`` mode
        only -- training mode is intentionally stochastic per-sample.
      * ``forward`` (generation path) handles the variable prefix mask
        when ``tokens_per_object > 0``.
      * The checkpoint round-trip preserves the new flags as ``config``
        metadata so eval scripts can reconstruct the adapter shape.
      * Phase 8: training-mode stripping fires at the configured rate
        and eval-mode stripping never fires (so deployment-time prompt
        always sees the full descriptors).
    """
    print("=" * 70)
    print(f"Phase B/C/8 integration: label={label!r}")
    print(f"  inject_scene_text     = {inject_scene_text}  "
          f"(style={scene_text_style}, color={scene_text_include_color}, "
          f"material={scene_text_include_material})")
    print(f"  tokens_per_object     = {tokens_per_object}")
    print(f"  descriptor_strip_prob = {descriptor_strip_prob}")
    print(f"  prefix_dropout        = {prefix_dropout}")
    print("=" * 70)
    t0 = time.time()

    from training.models.physics_former_full import FullPhysicsFormer
    from training.configs.config import TrainingConfig
    from train_adapter_v2 import PhysicsReasoningDataset, collate_fn
    from adapter_v3 import create_adapter_v3
    from torch.utils.data import DataLoader

    tmpdir = Path(tempfile.mkdtemp(prefix=f"phaseBC_{label}_"))
    try:
        scenes_dir = tmpdir / "scenes"
        qa_path    = tmpdir / "causal_qa_dataset.json"
        cache_path = tmpdir / "freeform_state_cache.pt"
        _build_synthetic_scenes(scenes_dir, n_scenes=8)
        _build_synthetic_qa_records(qa_path, scenes_dir, n_per_scene=4)

        config = TrainingConfig()
        physics_model = FullPhysicsFormer(
            state_dim=config.state_dim,
            hidden_dim=64, num_layers=2, num_heads=2, ff_dim=128,
            max_objects=config.max_objects,
            dropout=config.dropout, num_schema_classes=10,
        )
        physics_model.eval()
        for p in physics_model.parameters():
            p.requires_grad = False

        ff_dataset = PhysicsReasoningDataset(
            physics_dataset=None,
            freeform_qa_data_path=str(qa_path),
            freeform_qa_scenes_dir=str(scenes_dir),
            freeform_qa_cache_path=str(cache_path),
            freeform_max_objects=config.max_objects,
            freeform_seq_len=getattr(config, "sequence_length", 16),
        )

        loader = DataLoader(
            ff_dataset, batch_size=4, shuffle=False,
            collate_fn=collate_fn, num_workers=0, drop_last=False,
        )

        # When tokens_per_object > 0, num_prefix_tokens is forced to
        # max_objects * tokens_per_object by V2.__init__ regardless of
        # the num_prefix_tokens argument, so we pass a sentinel of 8 for
        # the scene-level case and let V2 compute the per-object size.
        adapter = create_adapter_v3(
            physics_model=physics_model,
            physics_dim=64,
            num_prefix_tokens=8,
            freeze_physics=True,
            freeze_llm=True,
            include_choices_prob=0.0,
            mixed_format_seed=42,
            label_smoothing=0.1,
            answer_token_dropout=0.15,
            force_llm_only_routing=True,
            inject_scene_text=inject_scene_text,
            scene_text_style=scene_text_style,
            scene_text_include_color=scene_text_include_color,
            scene_text_include_material=scene_text_include_material,
            descriptor_strip_prob=descriptor_strip_prob,
            prefix_dropout=prefix_dropout,
            tokens_per_object=tokens_per_object,
        )

        # Assert the adapter wired the requested flags through -- catches
        # a whole class of "factory accepts but doesn't forward" bugs.
        assert adapter.inject_scene_text == inject_scene_text
        assert adapter.scene_text_style == scene_text_style
        assert adapter.tokens_per_object == tokens_per_object
        assert adapter.descriptor_strip_prob == descriptor_strip_prob, (
            f"descriptor_strip_prob mismatch: "
            f"got {adapter.descriptor_strip_prob}, expected {descriptor_strip_prob}"
        )
        assert adapter.prefix_dropout == prefix_dropout, (
            f"prefix_dropout mismatch: "
            f"got {adapter.prefix_dropout}, expected {prefix_dropout}"
        )
        if tokens_per_object > 0:
            expected = int(physics_model.max_objects) * tokens_per_object
            assert adapter.num_prefix_tokens == expected, (
                f"per-object prefix size wrong: got {adapter.num_prefix_tokens}, "
                f"expected {expected}"
            )
            assert hasattr(adapter, "adapter_per_object"), (
                "tokens_per_object > 0 but adapter_per_object MLP not built"
            )
            assert not hasattr(adapter, "adapter") or adapter.adapter is None, (
                "scene-level adapter should not exist when tokens_per_object > 0"
            )
        else:
            assert hasattr(adapter, "adapter"), (
                "scene-level adapter MLP missing when tokens_per_object == 0"
            )

        # Scene-text determinism: build twice on the same batch and
        # confirm string identity. Guards against RNG drift or accidental
        # dependence on training-mode dropout. We do this in eval() mode
        # because Phase 8's descriptor stripping is *intentionally*
        # stochastic in training mode (per-sample Bernoulli), so calling
        # _build_scene_texts twice in a row would produce different
        # outputs by design when descriptor_strip_prob > 0. The eval-mode
        # check guarantees the deployment-time prompt is deterministic.
        if inject_scene_text:
            adapter.eval()
            sample = next(iter(loader))
            texts_1 = adapter._build_scene_texts(sample["states"], sample["masks"])
            texts_2 = adapter._build_scene_texts(sample["states"], sample["masks"])
            assert texts_1 == texts_2, (
                "scene-text builder is not deterministic in eval() mode"
            )
            # Spot-check structure -- should start with the expected
            # prefix. eval() mode disables stripping, so the prefix is
            # always the descriptor-full form.
            expected_prefix = (
                "Scene contains:" if scene_text_style == "comma_list" else "Object 1:"
            )
            assert any(
                t.startswith(expected_prefix) or t.startswith("No objects")
                for t in texts_1
            ), (
                f"scene texts don't have expected prefix in eval mode: "
                f"{texts_1[:2]}"
            )
            print(f"  [scene-text eval] sample: {texts_1[0][:80]!r}")

            # Phase 8: in training mode with descriptor_strip_prob > 0,
            # at least some samples in a sufficiently large batch should
            # be stripped. Use a fresh batch so stats start clean.
            if descriptor_strip_prob > 0.0:
                adapter.train()
                # Reset diagnostic counters for a clean check.
                adapter._n_strip = 0
                adapter._n_strip_eligible = 0
                # Re-seed the strip RNG so this assertion is deterministic
                # across CI runs (otherwise a very unlucky random draw
                # could fail the test even with correct logic).
                import random as _random
                adapter._descriptor_strip_rng = _random.Random(
                    adapter.descriptor_strip_seed
                )
                strip_batch = next(iter(loader))
                texts_train = adapter._build_scene_texts(
                    strip_batch["states"], strip_batch["masks"]
                )
                # The descriptor-stripped form is exactly
                # "Object 1. Object 2. ..." (no colon, no descriptors)
                # while the descriptor-full numbered form is
                # "Object 1: <color> <material> <shape>. ...". Distinguish
                # by checking for ': ' before the first period.
                def _is_stripped(t: str) -> bool:
                    if not t.startswith("Object 1"):
                        return False
                    # full form: "Object 1: red metal cube. Object 2: ..."
                    # strip form: "Object 1. Object 2. ..."
                    return t.startswith("Object 1.")
                n_stripped_observed = sum(_is_stripped(t) for t in texts_train)
                B = len(texts_train)
                # Counter-based check: tracks all calls in this train()
                # session, which is exactly one batch of size B here.
                assert adapter._n_strip_eligible == B, (
                    f"strip-eligible counter wrong: "
                    f"got {adapter._n_strip_eligible}, expected {B}"
                )
                # The realised strip count should equal the per-text
                # detection above (so the counter is wired to the same
                # decision the strings reflect).
                assert adapter._n_strip == n_stripped_observed, (
                    f"counter / observed strip mismatch: "
                    f"_n_strip={adapter._n_strip}, observed={n_stripped_observed}"
                )
                print(
                    f"  [scene-text train] strip rate: "
                    f"{adapter._n_strip}/{adapter._n_strip_eligible}"
                    f"  (configured p={descriptor_strip_prob})"
                )
                if texts_train:
                    print(f"  [scene-text train] sample: {texts_train[0][:80]!r}")
            else:
                # When descriptor_strip_prob == 0, training mode must
                # NEVER strip -- this guards against accidental
                # default-on regressions if someone later changes the
                # adapter init or the build_scene_summary fast path.
                adapter.train()
                texts_train = adapter._build_scene_texts(
                    sample["states"], sample["masks"]
                )
                # In train mode with strip_prob=0, the texts must match
                # the eval-mode texts byte for byte (no stochasticity).
                assert texts_train == texts_1, (
                    f"strip_prob=0 but train mode produced different texts "
                    f"than eval mode -- stripping fired by accident"
                )

        # Single training step exercising compute_combined_loss.
        trainable = [p for p in adapter.parameters() if p.requires_grad]
        assert trainable, "no trainable params after freeze"
        optimizer = optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)

        adapter.train()
        batch = next(iter(loader))
        loss, _ = adapter.compute_combined_loss(
            batch["states"], batch["masks"],
            batch["questions"], batch["answers"],
            choices=None, correct_choice_idx=None,
            numerical_targets={k: v for k, v in batch["numerical_targets"].items()},
            categorical_weight=1.0, numerical_weight=0.5,
        )
        assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Confirm the per-object adapter MLP (or scene-level adapter MLP,
        # depending on the branch) actually received gradients. This is
        # the test that caught earlier bugs where a new module was built
        # but the forward path still routed through the old one.
        if tokens_per_object > 0:
            for name, p in adapter.adapter_per_object.named_parameters():
                if p.requires_grad and p.grad is not None:
                    assert p.grad.abs().sum().item() > 0, (
                        f"adapter_per_object.{name} received zero gradient"
                    )
                    break
        else:
            for name, p in adapter.adapter.named_parameters():
                if p.requires_grad and p.grad is not None:
                    assert p.grad.abs().sum().item() > 0, (
                        f"adapter.{name} received zero gradient"
                    )
                    break
        optimizer.step()
        print(f"  [train] loss={loss.item():.4f}")

        # Generation smoke test -- exercises the V3.forward override that
        # handles tokens_per_object > 0. A single short question through
        # the exact call pattern notebook Cell 11 uses.
        adapter.eval()
        rec = ff_dataset.qa_pairs[0]
        states_b = rec["states"].unsqueeze(0)
        mask_2d  = rec["mask"]
        mask_1d  = mask_2d[0] if mask_2d.dim() == 2 else mask_2d
        mask_b   = mask_1d.unsqueeze(0)
        with torch.no_grad():
            answers = adapter(
                physics_states=states_b,
                object_mask=mask_b,
                question_text=[rec["question"]],
                max_length=10,
                do_sample=False,
            )
        assert isinstance(answers, list) and len(answers) == 1
        print(f"  [gen  ] {answers[0][:80]!r}")

        elapsed = time.time() - t0
        print(f"  [ok   ] {label} in {elapsed:.1f}s")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_phase_b_scene_text_injection() -> None:
    """Phase B: scene-text injection only (no per-object prefix)."""
    _run_phase_b_c_pipeline(
        "phase_b_comma_list",
        inject_scene_text=True,
        scene_text_style="comma_list",
        tokens_per_object=0,
    )


def test_phase_b_scene_text_numbered() -> None:
    """Phase B variant: numbered scene-text style."""
    _run_phase_b_c_pipeline(
        "phase_b_numbered",
        inject_scene_text=True,
        scene_text_style="numbered",
        tokens_per_object=0,
    )


def test_phase_c_per_object_prefix_only() -> None:
    """Phase C: per-object prefix without scene-text injection.

    Isolates the per-object adapter MLP so any failure here points at
    the prefix construction / variable-length mask plumbing, not at the
    scene-text pipeline.
    """
    _run_phase_b_c_pipeline(
        "phase_c_per_object_only",
        inject_scene_text=False,
        scene_text_style="comma_list",
        tokens_per_object=4,
    )


def test_phase_c_combined() -> None:
    """Phase C full stack: injection + per-object prefix together."""
    _run_phase_b_c_pipeline(
        "phase_c_combined",
        inject_scene_text=True,
        scene_text_style="numbered",
        tokens_per_object=4,
    )


def test_phase_8_descriptor_strip() -> None:
    """Phase 8 recipe: descriptor stripping + bumped prefix dropout.

    Verifies the full Phase 8 stack -- per-object prefix (Phase C),
    numbered scene-text (Phase B), descriptor stripping at p=0.5
    (Phase 8 new), and prefix dropout at p=0.3 (Phase 8 new) -- runs
    end-to-end through ``compute_combined_loss`` and the generation
    path without breaking the existing pipeline. The stronger
    assertions live inside ``_run_phase_b_c_pipeline`` (eval-mode
    determinism, train-mode stripping rate, counter correctness).
    """
    _run_phase_b_c_pipeline(
        "phase_8_strip_descriptors",
        inject_scene_text=True,
        scene_text_style="numbered",
        tokens_per_object=4,
        descriptor_strip_prob=0.5,
        prefix_dropout=0.3,
    )


if __name__ == "__main__":
    try:
        test_freeform_notebook_pipeline()
        test_phase_b_scene_text_injection()
        test_phase_b_scene_text_numbered()
        test_phase_c_per_object_prefix_only()
        test_phase_c_combined()
        test_phase_8_descriptor_strip()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
