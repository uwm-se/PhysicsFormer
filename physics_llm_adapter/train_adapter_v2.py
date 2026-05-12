"""
Training Script for Physics-LLM Adapter V2

Features:
- GPT-2 Medium (355M params)
- Agent-centric physical reasoning questions
- Increased training data (50k+ samples)
- Combined categorical + numerical training
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from collections import defaultdict
from tqdm import tqdm
import json
import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "physics_former"))

from adapter_v2 import PhysicsLLMAdapterV2, create_adapter_v2
from adapter_v3 import PhysicsLLMAdapterV3, create_adapter_v3
try:
    from qa_generator import PhysicsQAGenerator, QuestionType
except ImportError:
    PhysicsQAGenerator = None
    QuestionType = None
from training.models.physics_former_full import FullPhysicsFormer
from training.models.apply_modern_improvements import apply_modern_improvements
from training.configs.config import TrainingConfig
from training.datasets.hdf5_physics_dataset import HDF5PhysicsDataset


# =============================================================================
# PHYSICS REASONING QUESTION TYPES (for training)
# These are the grounded physics questions - NO metaphors during training
# Only available when qa_generator is installed (not on Colab)
# =============================================================================
if QuestionType is not None:
    PHYSICS_QUESTION_TYPES = [
        # Category 1: Basic Physical Properties
        QuestionType.OBJECT_COUNT,
        QuestionType.OBJECT_POSITION,
        QuestionType.OBJECT_VELOCITY,
        QuestionType.OBJECT_MASS,
        QuestionType.MOTION_DIRECTION,
        # Category 2: Physical Quantities
        QuestionType.KINETIC_ENERGY,
        QuestionType.TOTAL_MOMENTUM,
        QuestionType.RELATIVE_VELOCITY,
        QuestionType.SPATIAL_DISTANCE,
        QuestionType.SPEED_COMPARISON,
        QuestionType.MASS_COMPARISON,
        # Category 3: Predictive Reasoning
        QuestionType.COLLISION_PREDICTION,
        QuestionType.TRAJECTORY_EXTRAPOLATION,
        QuestionType.TIME_TO_EVENT,
        QuestionType.REACHABILITY,
        QuestionType.PATH_OBSTRUCTION,
        # Category 4: Relational Reasoning
        QuestionType.PROXIMITY,
        QuestionType.SPATIAL_CONTAINMENT,
        QuestionType.RELATIVE_POSITION,
        QuestionType.CONTACT_STATE,
        # Category 5: Evaluative Concepts (Safety & Normative Judgments)
        QuestionType.SAFETY_ASSESSMENT,
        QuestionType.DANGER_LEVEL,
        QuestionType.THREAT_TO_SELF,
        QuestionType.THREAT_TO_OTHERS,
        QuestionType.STABILITY_ASSESSMENT,
        QuestionType.COLLISION_RISK,
        QuestionType.ESCAPE_ROUTES,
        QuestionType.PROTECTIVE_ACTION,
        QuestionType.VULNERABILITY,
        QuestionType.CAUSAL_RESPONSIBILITY,
        QuestionType.FORCE_ASSESSMENT,
        QuestionType.STRUCTURAL_LOAD,
        QuestionType.URGENCY_ASSESSMENT,
        # Category 6: Intentional Concepts (Agency & Goal Attribution)
        QuestionType.AGENT_IDENTIFICATION,
        QuestionType.GOAL_INFERENCE,
        QuestionType.HELPING_HINDERING,
        QuestionType.CHASING_FLEEING,
        QuestionType.COOPERATION_COMPETITION,
        # Category 7: CLEVRER-Style Reasoning (Benchmark Alignment)
        QuestionType.CAUSAL_CHAIN,
        QuestionType.FUTURE_PREDICTION,
        QuestionType.COUNTERFACTUAL_REASONING,
    ]

    METAPHOR_QUESTION_TYPES = [
        # Original physics metaphors
        QuestionType.METAPHOR_COLLISION,
        QuestionType.METAPHOR_MOMENTUM,
        QuestionType.METAPHOR_EQUILIBRIUM,
        QuestionType.METAPHOR_TRAJECTORY,
        QuestionType.METAPHOR_FORCE,
        # Lakoff image schemas (from "Metaphors We Live By")
        QuestionType.METAPHOR_CONTAINER,
        QuestionType.METAPHOR_SOURCE_PATH_GOAL,
        QuestionType.METAPHOR_BALANCE,
        QuestionType.METAPHOR_LINK,
        QuestionType.METAPHOR_CENTER_PERIPHERY,
        QuestionType.METAPHOR_RESISTANCE,
        # Mathematical metaphors (from "Where Mathematics Comes From" - Lakoff & Núñez)
        QuestionType.METAPHOR_ARITHMETIC_MOTION,
        QuestionType.METAPHOR_ARITHMETIC_COLLECTION,
        QuestionType.METAPHOR_ARITHMETIC_CONSTRUCTION,
        QuestionType.METAPHOR_MEASURING_STICK,
        QuestionType.METAPHOR_SETS_CONTAINERS,
        QuestionType.METAPHOR_CONTINUITY_GAPLESS,
        QuestionType.METAPHOR_CHANGE_MOTION,
        QuestionType.METAPHOR_NUMBERS_POINTS,
        QuestionType.METAPHOR_RECURRENCE_CIRCULAR,
        QuestionType.METAPHOR_INFINITY,
    ]
else:
    PHYSICS_QUESTION_TYPES = []
    METAPHOR_QUESTION_TYPES = []


GENERALIZATION_SCHEMAS = [
    "barrier_breakthrough",
    "bridge_stability", 
    "cause_effect_chain",
    "chaos_driven_oscillator",
    "collision_elastic",
    "critical_point",
    "equilibrium_dynamic",
    "hierarchy_cascade",
    "multi_scale_interaction",
    "orbit_elliptical",
    "pendulum_double",
    "saturation_limit",
    "stack_balance",
    "symmetry_breaking",
]


class PhysicsReasoningDataset(Dataset):
    """Dataset for physics reasoning questions with balanced schema sampling.
    
    Supports four categories of physics questions:
    1. Basic Physical Properties (perceptual grounding)
    2. Physical Quantities (numerical reasoning)
    3. Predictive Reasoning (temporal inference)
    4. Relational Reasoning (spatial relations)
    5. CLEVRER-style reasoning (causal, predictive, counterfactual)
    
    Metaphor questions are held out for zero-shot evaluation.
    """
    
    CACHE_VERSION = "v2"
    
    def __init__(
        self,
        physics_dataset=None,
        num_samples: int = 50000,
        seed: int = 42,
        oversample_generalization_schemas: bool = True,
        objects_in_range_boost: int = 5,
        include_metaphor_questions: bool = False,
        metaphor_ratio: float = 0.2,
        clevrer_data_path: str = None,
        clevrer_ratio: float = 0.2,
        cache_dir: str = None,
        force_regenerate: bool = False,
        # Free-form QA mode (recipe ablation off the Phase 3 SOTA): replaces
        # the synthetic+CLEVRER MCQ mix entirely with free-form prose targets
        # like the 27k records in causal_qa_dataset.json. Parallel to the
        # mixed-format MCQ ablation in colab_train_adapter_v3.ipynb -- both
        # warm-start from adapter_phase3.pt, but exercise different
        # supervisory signals. Set freeform_qa_data_path to enable.
        freeform_qa_data_path: str = None,
        freeform_qa_scenes_dir: str = None,
        freeform_qa_cache_path: str = None,
        freeform_max_objects: int = 4,
        freeform_seq_len: int = 128,
        # Balanced-mix mode (Tier A.1): when ``freeform_qa_data_path`` is
        # provided AND ``physics_dataset`` is provided AND ``freeform_ratio``
        # is in (0, 1), build BOTH the synthetic + CLEVRER MCQ pool (existing
        # default flow) AND the free-form prose pool, then concatenate them
        # so the realised free-form fraction equals ``freeform_ratio``. The
        # synthetic vs CLEVRER MCQ split inside the base pool is still
        # controlled by the existing ``clevrer_ratio`` knob; the descriptive
        # head is supervised on the synthetic-descriptive subset because the
        # router (``classify_clevrer_question``) sends ``what color/shape/
        # how many ...`` items there. Leave ``freeform_ratio=None`` to
        # preserve the existing exclusive-mode behaviour byte-identically.
        freeform_ratio: float = None,
    ):
        self.physics_dataset = physics_dataset
        self.num_samples = num_samples
        self.seed = seed
        self.oversample_generalization_schemas = oversample_generalization_schemas
        self.objects_in_range_boost = objects_in_range_boost
        self.include_metaphor_questions = include_metaphor_questions
        self.metaphor_ratio = metaphor_ratio
        self.clevrer_data_path = clevrer_data_path
        self.clevrer_ratio = clevrer_ratio
        self.clevrer_samples = []
        self.freeform_qa_samples = []
        self.freeform_ratio = freeform_ratio
        self._freeform_qa_kwargs = (
            dict(
                qa_path=freeform_qa_data_path,
                scenes_dir=freeform_qa_scenes_dir,
                cache_path=freeform_qa_cache_path,
                max_objects=freeform_max_objects,
                seq_len=freeform_seq_len,
            )
            if freeform_qa_data_path is not None
            else None
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # ------------------------------------------------------------------
        # Free-form QA mode short-circuit (recipe ablation off Phase 3 SOTA).
        #
        # When ``freeform_qa_data_path`` is provided AND ``physics_dataset``
        # is None (or ``freeform_ratio`` is None / >= 1), replace the entire
        # training distribution with free-form QA records. Skip the synthetic
        # generator (no PhysicsQAGenerator dependency), skip the MCQ-style
        # CLEVRER mix, skip schema oversampling. ``physics_dataset`` is
        # unused in this mode.
        #
        # Each emitted record sets ``choices=None`` so V3's
        # ``compute_combined_loss`` routes every sample through Format A
        # (``{question} Answer:`` -> free-form prose target), which is the
        # supervisory signal the Phase 4 mixed-format ablation never trained
        # on. The two exclusive ablations are otherwise identical (same
        # architecture, same encoder, same warm-start checkpoint).
        # ------------------------------------------------------------------
        if freeform_qa_data_path is not None and (
            physics_dataset is None or freeform_ratio is None or freeform_ratio >= 1.0
        ):
            self.qa_pairs = self._load_freeform_qa_data(**self._freeform_qa_kwargs)
            self._print_distribution()
            return

        if physics_dataset is None:
            raise ValueError(
                "PhysicsReasoningDataset: physics_dataset is required unless "
                "freeform_qa_data_path is set (free-form QA ablation mode)."
            )

        # Validate balanced-mode ratio early so misconfig fails fast.
        if freeform_qa_data_path is not None:
            if not 0.0 < freeform_ratio < 1.0:
                raise ValueError(
                    f"freeform_ratio must be in (0, 1) for balanced mode (Tier A.1), "
                    f"got {freeform_ratio}. Use None or >=1.0 for 100% free-form, or "
                    f"omit freeform_qa_data_path entirely for 100% synthetic+MCQ."
                )

        cache_path = self._get_cache_path() if self.cache_dir else None

        if cache_path and cache_path.exists() and not force_regenerate:
            print(f"Loading cached QA pairs from {cache_path}...")
            self.qa_pairs = self._load_cache(cache_path)
            print(f"Loaded {len(self.qa_pairs)} cached QA pairs")
            self._print_distribution()
            self._maybe_mix_freeform()
            return

        if clevrer_data_path and Path(clevrer_data_path).exists():
            self._load_clevrer_data(clevrer_data_path)
        
        all_question_types = PHYSICS_QUESTION_TYPES.copy()
        if include_metaphor_questions:
            all_question_types.extend(METAPHOR_QUESTION_TYPES)
        
        self.generator = PhysicsQAGenerator(
            question_types=all_question_types,
            seed=seed
        )
        
        self.metaphor_generator = PhysicsQAGenerator(
            question_types=METAPHOR_QUESTION_TYPES,
            seed=seed + 1
        ) if include_metaphor_questions else None
        
        print(f"Generating {num_samples} embodied reasoning QA pairs...")
        print(f"Question types: {[qt.value for qt in all_question_types]}")
        if include_metaphor_questions:
            print(f"Including metaphor questions at {metaphor_ratio*100:.0f}% ratio")
        if self.clevrer_samples:
            print(f"Including {len(self.clevrer_samples)} CLEVRER samples at {clevrer_ratio*100:.0f}% ratio")
        if oversample_generalization_schemas:
            print(f"Oversampling generalization schemas: {len(GENERALIZATION_SCHEMAS)} schemas")
            print(f"objects_in_range boost factor: {objects_in_range_boost}x")
        self.qa_pairs = self._generate_qa_pairs()
        self._print_distribution()
        
        if cache_path:
            self._save_cache(cache_path)

        # Balanced-mix mode runs AFTER the synthetic + CLEVRER cache save so
        # the cache continues to represent only the deterministic base pool;
        # mixing in free-form is a fast post-process that does not need to
        # be cached separately.
        self._maybe_mix_freeform()
    
    def _get_cache_path(self) -> Path:
        """Generate cache filename based on dataset parameters."""
        import hashlib
        params = f"{self.num_samples}_{self.oversample_generalization_schemas}_{self.include_metaphor_questions}_{self.metaphor_ratio}_{self.clevrer_ratio}_{self.CACHE_VERSION}"
        param_hash = hashlib.md5(params.encode()).hexdigest()[:8]
        return self.cache_dir / f"adapter_qa_cache_{param_hash}.pt"
    
    def _save_cache(self, cache_path: Path):
        """Save QA pairs to cache file."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'qa_pairs': self.qa_pairs,
            'num_samples': self.num_samples,
            'version': self.CACHE_VERSION
        }, cache_path)
        print(f"Saved {len(self.qa_pairs)} QA pairs to cache: {cache_path}")
    
    def _load_cache(self, cache_path: Path) -> list:
        """Load QA pairs from cache file."""
        data = torch.load(cache_path, weights_only=False)
        if data.get('version') != self.CACHE_VERSION:
            print(f"Cache version mismatch (got {data.get('version')}, expected {self.CACHE_VERSION}), regenerating...")
            return None
        return data['qa_pairs']
    
    def _load_clevrer_data(self, clevrer_data_path: str):
        """Load pre-converted CLEVRER training data.
        
        Only keeps non-descriptive MCQ questions (counterfactual, explanatory,
        predictive) that have valid choices and a correct answer index.
        """
        import json
        from adapter_heads import classify_clevrer_question, CLEVRERQuestionCategory
        
        print(f"Loading CLEVRER training data from {clevrer_data_path}...")
        with open(clevrer_data_path, 'r') as f:
            data = json.load(f)
        
        skipped_no_mcq = 0
        skipped_descriptive = 0
        skipped_invalid = 0
        
        for sample in data:
            # Require valid question and answer text
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            if not isinstance(question, str) or not question.strip():
                skipped_invalid += 1
                continue
            if not isinstance(answer, str) or not answer.strip():
                skipped_invalid += 1
                continue
            
            # Require valid MCQ choices
            choices = sample.get('choices')
            correct_choice_idx = None
            if isinstance(choices, list) and choices:
                if answer in choices:
                    correct_choice_idx = choices.index(answer)
            
            if correct_choice_idx is None:
                skipped_no_mcq += 1
                continue
            
            # Reject descriptive questions — only train on causal reasoning MCQ
            cat = classify_clevrer_question(sample['question'])
            if cat == CLEVRERQuestionCategory.DESCRIPTIVE:
                skipped_descriptive += 1
                continue
            
            self.clevrer_samples.append({
                'states': torch.tensor(sample['states'], dtype=torch.float32),
                'mask': torch.tensor(sample['mask'], dtype=torch.float32),
                'question': sample['question'],
                'answer': sample['answer'],
                'choices': choices,
                'correct_choice_idx': correct_choice_idx,
                'question_type': sample['question_type'],
                'metadata': sample.get('metadata', {}),
                'numerical_targets': sample.get('numerical_targets', {
                    'distance': 0.0, 'speed': 0.0, 'time_to_collision': 0.0,
                    'kinetic_energy': 0.0, 'momentum': 0.0, 'object_count': 0.0
                })
            })
        print(f"Loaded {len(self.clevrer_samples)} non-descriptive MCQ CLEVRER samples")
        if skipped_invalid or skipped_no_mcq or skipped_descriptive:
            print(f"  Skipped: {skipped_invalid} invalid Q/A, {skipped_no_mcq} without MCQ, {skipped_descriptive} descriptive")

    # ------------------------------------------------------------------
    # Free-form QA loaders (recipe ablation off Phase 3 SOTA)
    # ------------------------------------------------------------------

    def _load_freeform_qa_data(
        self,
        qa_path: str,
        scenes_dir: str = None,
        cache_path: str = None,
        max_objects: int = 4,
        seq_len: int = 128,
    ):
        """Load a free-form QA JSON and join with a per-scene state cache.

        Mirrors ``_load_clevrer_data`` in spirit but consumes the prose-target
        QA records produced by ``explicit_world_model.llm_adapters.causal_qa_data``
        (e.g. ``causal_qa_dataset.json``). Each record's ``target`` becomes
        the answer_text fed to V3's loss path; ``choices=None`` forces the
        Format-A (no Options:) prompt branch in
        ``PhysicsLLMAdapterV3.compute_combined_loss``.

        Args
        ----
        qa_path : path to a JSON list of records with at minimum
            ``scene_index``, ``scene_path``, ``question``, ``target``.
        scenes_dir : optional override; if set, scene state is loaded from
            ``{scenes_dir}/annotation_{scene_index}.json`` (with recursive
            fallback) rather than from each record's ``scene_path``. Useful
            when training in a different environment than the one that
            produced the QA records (e.g. Drive-mounted Colab vs. local).
        cache_path : path to a ``.pt`` file holding the per-scene state
            cache. Built on-the-fly the first time this is called and
            re-used on subsequent invocations -- saves ~30 min on the
            27k-record CLEVRER causal QA dataset.
        max_objects, seq_len : padding/truncation budget; must match the
            encoder's training-time configuration (4/128 for the compsac
            Phase 1 physics_former).
        """
        from scene_converter import clevrer_scene_to_state_tensor  # local import

        print(f"Loading free-form QA records from {qa_path}...")
        with open(qa_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(
                f"Expected a JSON list of QA records at {qa_path}, "
                f"got {type(records).__name__}"
            )
        print(f"  loaded {len(records):,} QA records")

        cache_path_p = Path(cache_path) if cache_path else None
        if cache_path_p is not None and cache_path_p.exists():
            print(f"Loading state cache from {cache_path_p} ...")
            state_cache = torch.load(str(cache_path_p), map_location='cpu',
                                     weights_only=False)
            print(f"  cached scenes: {len(state_cache):,}")
        else:
            print(f"Building state cache (target={cache_path_p}) ...")
            state_cache = self._build_freeform_state_cache(
                records, scenes_dir, max_objects, seq_len,
            )
            if cache_path_p is not None:
                cache_path_p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(state_cache, str(cache_path_p))
                size_mb = cache_path_p.stat().st_size / 1e6
                print(f"  wrote {len(state_cache):,} scenes -> {cache_path_p}"
                      f" ({size_mb:.1f} MB)")

        qa_pairs = []
        skipped_no_state = 0
        skipped_invalid = 0
        type_counts = defaultdict(int)
        for rec in tqdm(records, desc='Building free-form QA pairs'):
            sid = rec.get('scene_index')
            if sid not in state_cache:
                skipped_no_state += 1
                continue
            question = rec.get('question', '')
            answer = rec.get('target', '')
            if not isinstance(question, str) or not question.strip():
                skipped_invalid += 1
                continue
            if not isinstance(answer, str) or not answer.strip():
                skipped_invalid += 1
                continue

            states_t, mask_t = state_cache[sid]
            qa_type = rec.get('qa_type', 'free_form')
            type_counts[qa_type] += 1

            qa_pairs.append({
                'states': states_t,
                'mask': mask_t,
                'question': question,
                'answer': answer,
                # CRITICAL: choices=None forces V3 into Format A on every
                # sample (no ``Options:`` clause in the prompt). This is the
                # supervisory-signal change that distinguishes the free-form
                # QA ablation from the mixed-format MCQ ablation -- both run
                # off the same Phase 3 SOTA warm-start.
                'choices': None,
                'correct_choice_idx': None,
                'question_type': qa_type,
                'metadata': {
                    'qa_source': 'free_form_causal_qa',
                    'scene_desc': rec.get('scene_desc', ''),
                    'scene_path': rec.get('scene_path', ''),
                },
                # No numerical targets for free-form prose; the numerical
                # head receives a zero-filled dict so the regression loss
                # contribution is constant (and the head can be frozen via
                # set_training_phase if desired).
                'numerical_targets': {
                    'distance': 0.0, 'speed': 0.0, 'time_to_collision': 0.0,
                    'kinetic_energy': 0.0, 'momentum': 0.0, 'object_count': 0.0,
                },
            })

        self.freeform_qa_samples = qa_pairs
        print(
            f"Built {len(qa_pairs):,} free-form QA pairs "
            f"(skipped {skipped_no_state} for missing state, "
            f"{skipped_invalid} invalid Q/A)"
        )
        if type_counts:
            print(f"  qa_type distribution:")
            for qt, n in sorted(type_counts.items(), key=lambda x: -x[1]):
                pct = 100 * n / len(qa_pairs)
                print(f"    {qt}: {n:,} ({pct:.1f}%)")
        return qa_pairs

    def _build_freeform_state_cache(
        self,
        records,
        scenes_dir: str,
        max_objects: int,
        seq_len: int,
    ):
        """Extract per-scene physics states once and pad to (seq_len, max_objects).

        Deduplicates ``scene_index`` across the QA record list so each
        scene JSON is parsed once (the 27k causal QA records cover ~5k
        unique CLEVRER scenes, so this is a ~5x speedup over per-record
        loading). Each cache entry is a tuple ``(states, mask)`` of
        ``(torch.float32 [seq_len, max_objects, state_dim])`` and
        ``(torch.float32 [seq_len, max_objects])`` respectively, matching
        the contract that ``collate_fn`` (above) expects from the
        per-record dicts.
        """
        from scene_converter import clevrer_scene_to_state_tensor  # local import

        # Dedup scene_indices, preferring per-record scene_path when
        # scenes_dir is not supplied.
        scene_to_path = {}
        for rec in records:
            sid = rec.get('scene_index')
            sp = rec.get('scene_path', '')
            if sid is None:
                continue
            scene_to_path.setdefault(sid, sp)

        print(f"  unique scenes to extract: {len(scene_to_path):,}")
        scenes_dir_p = Path(scenes_dir) if scenes_dir else None
        cache = {}
        failed = 0
        for sid, rec_path in tqdm(scene_to_path.items(),
                                   desc='Extracting scene states'):
            scene_path = None
            if scenes_dir_p is not None:
                # Try common naming first; fall back to recursive search.
                candidate = scenes_dir_p / f'annotation_{sid:05d}.json'
                if candidate.exists():
                    scene_path = candidate
                else:
                    for sub in scenes_dir_p.rglob(f'annotation_{sid:05d}.json'):
                        scene_path = sub
                        break
                    if scene_path is None:
                        for sub in scenes_dir_p.rglob(f'annotation_{sid}.json'):
                            scene_path = sub
                            break
            else:
                if rec_path:
                    scene_path = Path(rec_path)

            if scene_path is None or not scene_path.exists():
                failed += 1
                continue

            try:
                with open(scene_path, 'r') as f:
                    scene = json.load(f)
                states, masks, _ = clevrer_scene_to_state_tensor(scene)
            except Exception:
                failed += 1
                continue

            states = np.asarray(states, dtype=np.float32)
            masks = np.asarray(masks, dtype=np.float32)
            T, N, D = states.shape

            # Temporal dim: pad/truncate to seq_len.
            if T > seq_len:
                states = states[:seq_len]
                masks = masks[:seq_len]
            elif T < seq_len:
                pad = seq_len - T
                states = np.concatenate(
                    [states, np.zeros((pad, N, D), dtype=np.float32)], axis=0,
                )
                masks = np.concatenate(
                    [masks, np.zeros((pad, N), dtype=np.float32)], axis=0,
                )

            # Object dim: pad/truncate to max_objects.
            if N > max_objects:
                states = states[:, :max_objects]
                masks = masks[:, :max_objects]
            elif N < max_objects:
                pad = max_objects - N
                states = np.concatenate(
                    [states, np.zeros((seq_len, pad, D), dtype=np.float32)],
                    axis=1,
                )
                masks = np.concatenate(
                    [masks, np.zeros((seq_len, pad), dtype=np.float32)],
                    axis=1,
                )

            cache[sid] = (
                torch.from_numpy(states),
                torch.from_numpy(masks),
            )

        if failed:
            print(f"  ** {failed} scenes failed to load (missing JSON or parse error) **")
        return cache

    # ------------------------------------------------------------------
    # Balanced-mix helper (Tier A.1)
    # ------------------------------------------------------------------

    def _maybe_mix_freeform(self) -> None:
        """Concatenate free-form prose records into the synthetic + CLEVRER pool.

        Idempotent no-op unless ``__init__`` was called with both
        ``freeform_qa_data_path`` and a valid ``freeform_ratio in (0, 1)``.

        The realised free-form fraction matches the requested ratio:

            freeform_ratio = n_freeform / (n_base + n_freeform)
            =>  n_freeform = freeform_ratio * n_base / (1 - freeform_ratio)

        We sample without replacement when the free-form pool is large
        enough, with replacement otherwise. The merged list is shuffled
        with a dedicated RNG (``seed + 7``) so the realised order is
        deterministic and orthogonal to the synthetic generator's RNG.
        """
        if self._freeform_qa_kwargs is None:
            return
        if self.freeform_ratio is None or not 0.0 < self.freeform_ratio < 1.0:
            return

        print(
            f"\n[Balanced mode] Loading free-form QA pool to mix in "
            f"at ratio {self.freeform_ratio:.3f} ..."
        )
        freeform_pairs = self._load_freeform_qa_data(**self._freeform_qa_kwargs)

        n_base = len(self.qa_pairs)
        target_n_ff = int(round(
            self.freeform_ratio * n_base / (1.0 - self.freeform_ratio)
        ))

        import random as _random
        rng = _random.Random(self.seed + 7)

        if target_n_ff <= len(freeform_pairs):
            sampled_ff = rng.sample(freeform_pairs, target_n_ff)
            sampling_mode = "without replacement"
        else:
            sampled_ff = [rng.choice(freeform_pairs) for _ in range(target_n_ff)]
            sampling_mode = (
                f"with replacement (pool {len(freeform_pairs):,} < target {target_n_ff:,})"
            )

        print(
            f"[Balanced mode] base={n_base:,}, freeform={target_n_ff:,} "
            f"({sampling_mode})"
        )

        self.qa_pairs = list(self.qa_pairs) + sampled_ff
        rng.shuffle(self.qa_pairs)

        realised = target_n_ff / max(1, n_base + target_n_ff)
        print(
            f"[Balanced mode] realised freeform_ratio: {realised:.3f} "
            f"(target {self.freeform_ratio:.3f})"
        )
        print(f"[Balanced mode] total mixed samples: {len(self.qa_pairs):,}")

    def _get_schema_index_cache_path(self) -> Path:
        """Get path for schema index cache."""
        if not self.cache_dir:
            return None
        return self.cache_dir / "schema_index_cache.pt"
    
    def _build_schema_index(self):
        """Build index mapping schema names to dataset indices by sampling."""
        cache_path = self._get_schema_index_cache_path()
        
        if cache_path and cache_path.exists():
            print(f"Loading cached schema index from {cache_path}...")
            cached = torch.load(cache_path, weights_only=False)
            print(f"Loaded {len(cached)} schemas from cache")
            return cached
        
        schema_to_indices = {}
        print("Building schema index (sampling 10% of dataset)...")
        dataset_size = len(self.physics_dataset)
        sample_size = min(dataset_size // 10, 50000)
        
        for _ in tqdm(range(sample_size), desc="Indexing schemas"):
            idx = torch.randint(0, dataset_size, (1,)).item()
            sample = self.physics_dataset[idx]
            schema_name = sample.get('schema_name', 'unknown')
            if schema_name not in schema_to_indices:
                schema_to_indices[schema_name] = []
            schema_to_indices[schema_name].append(idx)
        
        print(f"Found {len(schema_to_indices)} unique schemas with {sum(len(v) for v in schema_to_indices.values())} indexed episodes")
        for schema, indices in sorted(schema_to_indices.items()):
            print(f"  {schema}: {len(indices)} samples")
        
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(schema_to_indices, cache_path)
            print(f"Saved schema index to cache: {cache_path}")
        
        return schema_to_indices
    
    def _generate_qa_pairs(self):
        qa_pairs = []
        dataset_size = len(self.physics_dataset)
        
        schema_to_indices = None
        if self.oversample_generalization_schemas:
            schema_to_indices = self._build_schema_index()
        
        base_samples = self.num_samples
        if self.oversample_generalization_schemas:
            base_samples = int(self.num_samples * 0.7)
        
        metaphor_samples = int(base_samples * self.metaphor_ratio) if self.include_metaphor_questions else 0
        regular_samples = base_samples - metaphor_samples
        
        print(f"Generating {regular_samples} regular samples + {metaphor_samples} metaphor samples...")
        
        for i in tqdm(range(regular_samples), desc="Regular samples"):
            idx = torch.randint(0, dataset_size, (1,)).item()
            sample = self.physics_dataset[idx]
            
            states = sample['object_states']
            mask = sample['object_mask']
            
            qa = self.generator.generate_qa_pair(states, mask)
            
            numerical_targets = self._extract_numerical_targets(qa.metadata, qa.question_type)
            
            qa_pairs.append({
                'states': qa.states,
                'mask': qa.mask,
                'question': qa.question,
                'answer': qa.answer,
                'question_type': qa.question_type.value,
                'metadata': qa.metadata,
                'numerical_targets': numerical_targets
            })
        
        if self.include_metaphor_questions and self.metaphor_generator:
            print(f"Generating {metaphor_samples} metaphor samples for LLM generalization...")
            for i in tqdm(range(metaphor_samples), desc="Metaphor samples"):
                idx = torch.randint(0, dataset_size, (1,)).item()
                sample = self.physics_dataset[idx]
                
                states = sample['object_states']
                mask = sample['object_mask']
                
                qa = self.metaphor_generator.generate_qa_pair(states, mask)
                
                numerical_targets = self._extract_numerical_targets(qa.metadata, qa.question_type)
                
                qa_pairs.append({
                    'states': qa.states,
                    'mask': qa.mask,
                    'question': qa.question,
                    'answer': qa.answer,
                    'question_type': qa.question_type.value,
                    'metadata': qa.metadata,
                    'numerical_targets': numerical_targets
                })
        
        if self.oversample_generalization_schemas and schema_to_indices:
            extra_samples = self.num_samples - base_samples
            samples_per_schema = extra_samples // len(GENERALIZATION_SCHEMAS)
            
            print(f"\nGenerating {extra_samples} extra samples from generalization schemas...")
            print(f"  {samples_per_schema} samples per schema")
            
            for schema_name in GENERALIZATION_SCHEMAS:
                if schema_name not in schema_to_indices:
                    print(f"  WARNING: Schema '{schema_name}' not found in dataset")
                    continue
                
                indices = schema_to_indices[schema_name]
                objects_in_range_count = 0
                
                for _ in range(samples_per_schema):
                    idx = indices[torch.randint(0, len(indices), (1,)).item()]
                    sample = self.physics_dataset[idx]
                    
                    states = sample['object_states']
                    mask = sample['object_mask']
                    
                    if objects_in_range_count < samples_per_schema // self.objects_in_range_boost:
                        qa = self.generator.generate_qa_pair(states, mask, question_type=QuestionType.OBJECTS_IN_RANGE)
                        objects_in_range_count += 1
                    else:
                        qa = self.generator.generate_qa_pair(states, mask)
                    
                    numerical_targets = self._extract_numerical_targets(qa.metadata, qa.question_type)
                    
                    qa_pairs.append({
                        'states': qa.states,
                        'mask': qa.mask,
                        'question': qa.question,
                        'answer': qa.answer,
                        'question_type': qa.question_type.value,
                        'metadata': qa.metadata,
                        'numerical_targets': numerical_targets
                    })
        
        if self.clevrer_samples:
            clevrer_count = int(len(qa_pairs) * self.clevrer_ratio / (1 - self.clevrer_ratio))
            clevrer_count = min(clevrer_count, len(self.clevrer_samples))
            
            print(f"\nAdding {clevrer_count} CLEVRER samples to training data...")
            
            import random
            selected_clevrer = random.sample(self.clevrer_samples, clevrer_count) if clevrer_count < len(self.clevrer_samples) else self.clevrer_samples
            qa_pairs.extend(selected_clevrer)
            
            random.shuffle(qa_pairs)
        
        return qa_pairs
    
    def _extract_numerical_targets(self, metadata: dict, question_type: QuestionType) -> dict:
        """Extract normalized numerical targets from QA metadata for regression head."""
        targets = {}
        
        if 'min_distance' in metadata:
            val = float(metadata['min_distance'])
            targets['distance'] = min(val / 10.0, 1.0)
        if 'max_speed' in metadata:
            val = float(metadata['max_speed'])
            targets['speed'] = min(val / 1.0, 1.0)
        if 'time_to_collision' in metadata:
            ttc = metadata['time_to_collision']
            val = float(ttc) if ttc != float('inf') else 100.0
            targets['time_to_collision'] = min(val / 100.0, 1.0)
        if 'kinetic_energy' in metadata:
            val = float(metadata['kinetic_energy'])
            targets['kinetic_energy'] = min(val / 10.0, 1.0)
        if 'momentum_magnitude' in metadata:
            val = float(metadata['momentum_magnitude'])
            targets['momentum'] = min(val / 10.0, 1.0)
        if question_type == QuestionType.OBJECT_COUNT and 'count' in metadata:
            val = float(metadata['count'])
            targets['object_count'] = val / 20.0
        
        return targets
    
    def _print_distribution(self):
        type_counts = {}
        for qa in self.qa_pairs:
            qt = qa['question_type']
            type_counts[qt] = type_counts.get(qt, 0) + 1
        
        print("\nQuestion type distribution:")
        for qt, count in sorted(type_counts.items()):
            pct = 100 * count / len(self.qa_pairs)
            print(f"  {qt}: {count} ({pct:.1f}%)")
        print()
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        return self.qa_pairs[idx]


def collate_fn(batch):
    """Collate function for DataLoader with padding for variable-size tensors."""
    max_seq_len = max(item['states'].shape[0] for item in batch)
    max_objects = max(item['states'].shape[1] for item in batch)
    state_dim = batch[0]['states'].shape[2]
    
    batch_size = len(batch)
    padded_states = torch.zeros(batch_size, max_seq_len, max_objects, state_dim)
    padded_masks = torch.zeros(batch_size, max_objects)
    
    for i, item in enumerate(batch):
        seq_len, num_obj, _ = item['states'].shape
        padded_states[i, :seq_len, :num_obj, :] = item['states']
        
        if item['mask'].dim() == 1:
            padded_masks[i, :num_obj] = item['mask'][:num_obj]
        elif item['mask'].dim() == 2:
            padded_masks[i, :num_obj] = item['mask'][0, :num_obj]
        else:
            padded_masks[i, :min(num_obj, item['mask'].shape[-1])] = item['mask'].flatten()[:num_obj]
    
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    question_types = [item['question_type'] for item in batch]
    
    # Preserve per-sample choice data so mixed batches (some MCQ, some not)
    # still route MCQ-eligible samples through the MCQ loss path.
    choices_list = [item.get('choices') for item in batch]
    correct_idx_list = [item.get('correct_choice_idx') for item in batch]

    # Validate each sample individually
    valid_choices = []
    valid_idx = []
    any_valid = False
    for c, idx in zip(choices_list, correct_idx_list):
        if isinstance(c, list) and len(c) > 0 and isinstance(idx, int):
            valid_choices.append(c)
            valid_idx.append(idx)
            any_valid = True
        else:
            valid_choices.append(None)
            valid_idx.append(None)

    if any_valid:
        choices = valid_choices       # List[Optional[List[str]]]
        correct_choice_idx = valid_idx  # List[Optional[int]]
    else:
        choices = None
        correct_choice_idx = None
    
    # Support both old format (distance, speed, etc.) and HDF5 format (count, value)
    numerical_keys = ['distance', 'speed', 'time_to_collision', 
                      'kinetic_energy', 'momentum', 'object_count', 'count', 'value']
    numerical_targets = {}
    
    for key in numerical_keys:
        values = []
        for item in batch:
            if key in item['numerical_targets']:
                values.append(item['numerical_targets'][key])
            else:
                values.append(0.0)
        numerical_targets[key] = torch.tensor(values, dtype=torch.float32)
    
    return {
        'states': padded_states,
        'masks': padded_masks,
        'questions': questions,
        'answers': answers,
        'choices': choices,
        'correct_choice_idx': correct_choice_idx,
        'question_types': question_types,
        'numerical_targets': numerical_targets
    }


def evaluate(model, test_loader, device, num_samples=200):
    """Evaluate model on test set.
    
    For MCQ samples: compare predicted choice index vs correct_choice_idx.
    For non-MCQ samples: LLM generation + text matching.
    """
    model.eval()
    
    results = defaultdict(lambda: {'correct': 0, 'total': 0})
    numerical_errors = defaultdict(list)
    eval_errors = 0
    
    samples_seen = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if samples_seen >= num_samples:
                break
            
            states = batch['states'].to(device)
            masks = batch['masks'].to(device)
            questions = batch['questions']
            answers = batch['answers']
            question_types = batch['question_types']
            numerical_targets = batch['numerical_targets']
            choices_list = batch.get('choices', None)
            correct_idx_list = batch.get('correct_choice_idx', None)
            
            batch_count += 1
            if batch_count == 1:
                print(f"  [EVAL] First batch: {len(questions)} samples, "
                      f"choices={'present' if choices_list is not None else 'None'}, "
                      f"correct_idx={'present' if correct_idx_list is not None else 'None'}")
            
            for i in range(len(questions)):
                qt = question_types[i]
                
                has_mcq = (choices_list is not None
                           and choices_list[i] is not None
                           and isinstance(choices_list[i], list)
                           and len(choices_list[i]) > 0)
                
                try:
                    if has_mcq:
                        # LLM perplexity scoring: pick choice with lowest loss
                        scores = model.score_answer_candidates(
                            physics_states=states[i:i+1],
                            object_mask=masks[i:i+1],
                            question_text=[questions[i]],
                            answer_candidates=[choices_list[i]],
                            max_length=128
                        )
                        pred_idx = torch.argmax(scores, dim=-1).item()
                        
                        # Compare by index (reliable) rather than text (fragile)
                        correct_idx = (correct_idx_list[i]
                                       if correct_idx_list is not None
                                       and correct_idx_list[i] is not None
                                       else None)
                        
                        if correct_idx is not None:
                            is_correct = (pred_idx == correct_idx)
                        else:
                            # Fallback: text match
                            pred = choices_list[i][pred_idx].strip().lower()
                            exp = answers[i].strip().lower()
                            is_correct = (pred == exp or exp in pred or pred in exp)
                        
                        if batch_count == 1 and i == 0:
                            print(f"  [EVAL] Sample 0: {len(choices_list[i])} choices, "
                                  f"pred_idx={pred_idx}, correct_idx={correct_idx}, "
                                  f"correct={is_correct}")
                    else:
                        # LLM generation for non-MCQ
                        gen = model(states[i:i+1], masks[i:i+1], [questions[i]], max_length=20)
                        pred = gen[0].split('\n')[0].strip().lower()
                        exp = answers[i].strip().lower()
                        is_correct = (pred == exp or exp in pred or pred in exp)
                    
                    results[qt]['total'] += 1
                    if is_correct:
                        results[qt]['correct'] += 1
                        
                except Exception as e:
                    eval_errors += 1
                    if eval_errors <= 3:
                        print(f"  [EVAL ERROR] Sample {samples_seen + i}: {type(e).__name__}: {e}")
            
            # Numerical predictions
            try:
                numerical_preds = model.predict_numerical(states, masks)
                for key in numerical_preds:
                    if key in numerical_targets:
                        pred_val = numerical_preds[key].cpu()
                        target = numerical_targets[key]
                        # Only record if target has real data (not all zeros from defaults)
                        if target.abs().sum() > 0:
                            errors = torch.abs(pred_val - target).tolist()
                            numerical_errors[key].extend(errors)
            except Exception as e:
                if eval_errors <= 3:
                    print(f"  [EVAL ERROR] Numerical prediction: {type(e).__name__}: {e}")
            
            samples_seen += len(questions)
    
    print(f"\n  [EVAL] Processed {samples_seen} samples in {batch_count} batches"
          + (f", {eval_errors} errors" if eval_errors else ""))
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print("\nAccuracy by Question Type:")
    total_correct = 0
    total_count = 0
    for qt, counts in sorted(results.items()):
        if counts['total'] > 0:
            acc = 100 * counts['correct'] / counts['total']
            print(f"  {qt}: {acc:.1f}% ({counts['correct']}/{counts['total']})")
            total_correct += counts['correct']
            total_count += counts['total']
    
    if total_count > 0:
        overall_acc = 100 * total_correct / total_count
        print(f"\nOverall Categorical Accuracy: {overall_acc:.1f}%")
    else:
        overall_acc = 0.0
        print("\n  (No samples evaluated — check errors above)")
    
    if numerical_errors:
        print("\nNumerical Prediction MAE:")
        for key, errors in sorted(numerical_errors.items()):
            if errors:
                mae = sum(errors) / len(errors)
                print(f"  {key}: {mae:.4f}")
    
    print("=" * 70)
    
    return overall_acc


def train_phase(model, train_loader, optimizer, device, phase_name, max_epochs=100, patience=2, min_delta=0.001, checkpoint_dir=None, phase_num=1, save_every_n_epochs=3, use_contrastive=False, contrastive_weight=0.1, start_epoch=0, initial_best_loss=None, initial_epochs_without_improvement=0):
    """Train for one phase with early stopping and checkpoint saving.

    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        phase_name: Name of the training phase for logging
        max_epochs: Maximum number of epochs (default 100, but early stopping will trigger first)
        patience: Number of epochs without improvement before stopping
        min_delta: Minimum improvement to count as progress
        checkpoint_dir: Directory to save checkpoints (if None, no checkpoints saved)
        phase_num: Phase number (1, 2, or 3) for checkpoint naming
        save_every_n_epochs: Save checkpoint every N epochs (default 3)
        use_contrastive: Whether to use contrastive loss on prefix tokens
        contrastive_weight: Weight for contrastive loss (default 0.1)
        start_epoch: Resume at this 0-indexed epoch (default 0 = fresh run).
            Pair with ``initial_best_loss`` and ``initial_epochs_without_improvement``
            restored from a saved checkpoint so early-stopping state survives a crash.
        initial_best_loss: Best loss observed before the crash (default: infinity).
        initial_epochs_without_improvement: Early-stopping counter before the crash.
    """
    print(f"\n{'=' * 70}")
    print(f"TRAINING PHASE: {phase_name}")
    print(f"Early stopping: patience={patience}, min_delta={min_delta}")
    if use_contrastive:
        print(f"Contrastive loss: ENABLED (weight={contrastive_weight})")
    if checkpoint_dir:
        print(f"Checkpoints: {checkpoint_dir} (every {save_every_n_epochs} epochs + phase end)")
    print("=" * 70)
    
    best_loss = float('inf') if initial_best_loss is None else float(initial_best_loss)
    epochs_without_improvement = int(initial_epochs_without_improvement)
    if start_epoch > 0:
        print(f"[resume] start_epoch={start_epoch}, best_loss={best_loss:.4f}, "
              f"epochs_without_improvement={epochs_without_improvement}/{patience}")
    # Track whether any training actually happened in this call so we know
    # whether to emit a phase-end checkpoint at the bottom.
    epoch = start_epoch - 1
    avg_loss = best_loss

    # Cumulative Format-A/B counters survive the per-epoch reset_mixed_format_stats()
    # call below, so we can stamp every saved checkpoint with how many of each format
    # actually trained. resume_phase_state() in the Colab notebook reads these to
    # reject sentinels whose recipe doesn't match (root cause of the Phase-4 0% bug).
    cumulative_format_a = 0
    cumulative_format_b = 0

    def _build_recipe_metadata():
        """Recipe metadata to embed in every checkpoint torch.save.

        For V2 adapters, ``include_choices_prob`` and ``mixed_format_seed`` are
        absent (returned as None), and cumulative counters stay at 0. Future
        resume_phase_state() callers compare these against the live config and
        reject mismatched/stale sentinels rather than silently loading them.
        """
        return {
            'adapter_class': type(model).__name__,
            'include_choices_prob': getattr(model, 'include_choices_prob', None),
            'mixed_format_seed': getattr(model, 'mixed_format_seed', None),
            'cumulative_format_a': cumulative_format_a,
            'cumulative_format_b': cumulative_format_b,
        }

    for epoch in range(start_epoch, max_epochs):
        model.train()
        total_loss = 0.0
        cat_loss_sum = 0.0
        num_loss_sum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            states = batch['states'].to(device)
            masks = batch['masks'].to(device)
            questions = batch['questions']
            answers = batch['answers']
            choices = batch.get('choices')              # List[Optional[List[str]]] or None
            correct_choice_idx = batch.get('correct_choice_idx')  # List[Optional[int]] or None
            numerical_targets = {k: v.to(device) for k, v in batch['numerical_targets'].items()}
            
            optimizer.zero_grad()

            if use_contrastive:
                loss, loss_dict = model.compute_combined_loss_with_contrastive(
                    states, masks, questions, answers,
                    choices=choices,
                    correct_choice_idx=correct_choice_idx,
                    numerical_targets=numerical_targets,
                    categorical_weight=1.0,
                    numerical_weight=0.5,
                    contrastive_weight=contrastive_weight
                )
            else:
                loss, loss_dict = model.compute_combined_loss(
                    states, masks, questions, answers,
                    choices=choices,
                    correct_choice_idx=correct_choice_idx,
                    numerical_targets=numerical_targets,
                    categorical_weight=1.0,
                    numerical_weight=0.5
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            cat_loss_sum += loss_dict['categorical'].item()
            num_loss_sum += loss_dict['numerical'].item()
            desc_loss = loss_dict.get('descriptive', torch.tensor(0.0)).item()
            contr_loss = loss_dict.get('contrastive', torch.tensor(0.0)).item()

            postfix = {
                'loss': f"{loss.item():.4f}",
                'cat': f"{loss_dict['categorical'].item():.4f}",
                'desc': f"{desc_loss:.4f}",
                'num': f"{loss_dict['numerical'].item():.4f}"
            }
            if use_contrastive:
                postfix['contr'] = f"{contr_loss:.4f}"
            pbar.set_postfix(postfix)
        
        avg_loss = total_loss / len(train_loader)
        avg_cat = cat_loss_sum / len(train_loader)
        avg_num = num_loss_sum / len(train_loader)
        
        improved = False
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            epochs_without_improvement = 0
            status = "improved ✓"
            improved = True
        else:
            epochs_without_improvement += 1
            status = f"no improvement ({epochs_without_improvement}/{patience})"
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} (cat: {avg_cat:.4f}, num: {avg_num:.4f}) - {status}")

        # V3-only diagnostic: log realised Format-A/B ratio for the epoch and
        # accumulate into the cumulative counters that get stamped into checkpoints.
        # Silently skipped on V2 (no mixed_format_stats method).
        if hasattr(model, 'mixed_format_stats'):
            mf = model.mixed_format_stats()
            cumulative_format_a += mf['n_format_a']
            cumulative_format_b += mf['n_format_b']
            print(f"  [mixed-format] format_b_fraction={mf['format_b_fraction']:.3f} "
                  f"(configured={mf['configured_prob']:.2f}, "
                  f"n_a={mf['n_format_a']}, n_b={mf['n_format_b']}, "
                  f"cumulative n_a={cumulative_format_a}, n_b={cumulative_format_b})")
            model.reset_mixed_format_stats()

        # Save checkpoint every N epochs OR on improvement
        if checkpoint_dir and ((epoch + 1) % save_every_n_epochs == 0 or improved):
            ckpt_path = checkpoint_dir / f"adapter_phase{phase_num}_epoch{epoch+1}_loss{avg_loss:.4f}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'phase': phase_num,
                'epoch': epoch + 1,
                'loss': avg_loss,
                'best_loss': best_loss,
                'epochs_without_improvement': epochs_without_improvement,
                **_build_recipe_metadata(),
            }, ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path.name}")
        
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
            break
    
    # Always save phase-end checkpoint (only if we actually ran at least one epoch)
    if checkpoint_dir and epoch >= start_epoch:
        phase_end_path = checkpoint_dir / f"adapter_phase{phase_num}_complete_loss{best_loss:.4f}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'phase': phase_num,
            'epoch': epoch + 1,
            'loss': avg_loss,
            'best_loss': best_loss,
            'epochs_without_improvement': epochs_without_improvement,
            **_build_recipe_metadata(),
        }, phase_end_path)
        print(f"  → Phase complete checkpoint: {phase_end_path.name}")
        if hasattr(model, 'mixed_format_stats'):
            print(f"  → Cumulative across phase: n_format_a={cumulative_format_a}, "
                  f"n_format_b={cumulative_format_b} "
                  f"(configured include_choices_prob={getattr(model, 'include_choices_prob', None)})")
    
    print(f"\n{phase_name} complete! Best loss: {best_loss:.4f}")
    return best_loss


def validate_physics_usage(model, val_loader, device, num_samples=20):
    """
    Validate that the model is actually using physics information.

    Compares prefix tokens between real physics and zero physics inputs.
    Returns cosine similarity (lower = better physics usage).
    """
    model.eval()
    similarities = []
    differences = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break

            states = batch['states'].to(device)
            masks = batch['masks'].to(device)
            zero_states = torch.zeros_like(states)

            # Get prefix tokens for real and zero physics
            real_features = model.extract_physics_features(states, masks)
            zero_features = model.extract_physics_features(zero_states, masks)

            real_prefix = model.create_prefix_tokens(real_features)
            zero_prefix = model.create_prefix_tokens(zero_features)

            # Flatten and compute cosine similarity
            real_flat = real_prefix.view(real_prefix.size(0), -1)
            zero_flat = zero_prefix.view(zero_prefix.size(0), -1)

            cos_sim = torch.nn.functional.cosine_similarity(real_flat, zero_flat, dim=1)
            diff_norm = (real_flat - zero_flat).norm(dim=1)

            similarities.extend(cos_sim.cpu().tolist())
            differences.extend(diff_norm.cpu().tolist())

    avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
    avg_difference = sum(differences) / len(differences) if differences else 0.0

    return {
        'avg_cosine_similarity': avg_similarity,
        'avg_difference_norm': avg_difference,
        'num_samples': len(similarities)
    }


def train_phase_with_validation(model, train_loader, val_loader, optimizer, device, phase_name,
                                 max_epochs=100, patience=5, min_delta=0.001, checkpoint_dir=None,
                                 phase_num=1, save_every_n_epochs=3, use_contrastive=False,
                                 contrastive_weight=0.1, validate_every_n_epochs=5,
                                 scheduler=None, early_stopping=False,
                                 catapult_patience=5, catapult_multiplier=5.0,
                                 catapult_decay_epochs=3,
                                 overfit_gap_threshold=0.0, overfit_gap_patience=3):
    """
    Train with periodic physics usage validation and plateau catapult.
    
    Plateau catapult: when val loss stalls for `catapult_patience` epochs,
    multiply LR by `catapult_multiplier` to escape the local minimum, then
    exponentially decay back to baseline over `catapult_decay_epochs` epochs.
    
    Args:
        early_stopping: If True, stop training when val loss plateaus. Default False.
        catapult_patience: Epochs without val improvement before catapulting.
        catapult_multiplier: LR boost factor on catapult (e.g. 5.0 = 5x LR).
        catapult_decay_epochs: Epochs to decay back from boosted LR to baseline.
        overfit_gap_threshold: Stop if (val_loss - train_loss) exceeds this for
            overfit_gap_patience consecutive epochs. 0 = disabled. Default 0.
        overfit_gap_patience: Consecutive epochs gap must exceed threshold. Default 3.
    """
    print(f"\n{'=' * 70}")
    print(f"TRAINING PHASE: {phase_name}")
    if early_stopping:
        print(f"Early stopping: patience={patience}, min_delta={min_delta}")
    else:
        print(f"Early stopping: DISABLED")
    print(f"Plateau catapult: patience={catapult_patience}, boost={catapult_multiplier}x, decay={catapult_decay_epochs} epochs")
    if overfit_gap_threshold > 0:
        print(f"Overfit gap stop: gap>{overfit_gap_threshold} for {overfit_gap_patience} epochs")
    if use_contrastive:
        print(f"Contrastive loss: ENABLED (weight={contrastive_weight})")
    print(f"Physics validation: every {validate_every_n_epochs} epochs")
    if checkpoint_dir:
        print(f"Checkpoints: {checkpoint_dir} (every {save_every_n_epochs} epochs + phase end)")
    print("=" * 70)

    best_loss = float('inf')
    epochs_without_improvement = 0
    
    # Plateau catapult state
    # Use initial_lr (set by scheduler) if available, else current lr
    base_lr = optimizer.param_groups[0].get('initial_lr', optimizer.param_groups[0]['lr'])
    catapult_active = False
    catapult_epoch_counter = 0
    catapult_count = 0  # how many times we've catapulted
    consecutive_catapults = 0  # consecutive catapults without improvement (for escalation)
    max_catapult_multiplier = 25.0  # safety cap on escalation
    current_catapult_multiplier = catapult_multiplier  # tracks actual multiplier used for decay
    overfit_gap_counter = 0  # consecutive epochs exceeding gap threshold
    best_ckpt_path = None  # track best checkpoint for reload on gap stop
    print(f"  Catapult base LR: {base_lr:.2e}")

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        cat_loss_sum = 0.0
        num_loss_sum = 0.0
        contr_loss_sum = 0.0
        epoch_mcq_correct = 0
        epoch_mcq_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            states = batch['states'].to(device)
            masks = batch['masks'].to(device)
            questions = batch['questions']
            answers = batch['answers']
            choices = batch.get('choices')              # List[Optional[List[str]]] or None
            correct_choice_idx = batch.get('correct_choice_idx')  # List[Optional[int]] or None
            numerical_targets = {k: v.to(device) for k, v in batch['numerical_targets'].items()}

            optimizer.zero_grad()

            if use_contrastive:
                loss, loss_dict = model.compute_combined_loss_with_contrastive(
                    states, masks, questions, answers,
                    choices=choices,
                    correct_choice_idx=correct_choice_idx,
                    numerical_targets=numerical_targets,
                    categorical_weight=1.0,
                    numerical_weight=0.5,
                    contrastive_weight=contrastive_weight
                )
            else:
                loss, loss_dict = model.compute_combined_loss(
                    states, masks, questions, answers,
                    choices=choices,
                    correct_choice_idx=correct_choice_idx,
                    numerical_targets=numerical_targets,
                    categorical_weight=1.0,
                    numerical_weight=0.5
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None and not catapult_active:
                scheduler.step()

            total_loss += loss.item()
            cat_loss_sum += loss_dict['categorical'].item()
            num_loss_sum += loss_dict['numerical'].item()
            contr_loss = loss_dict.get('contrastive', torch.tensor(0.0)).item()
            contr_loss_sum += contr_loss
            obj_mask_loss = loss_dict.get('object_masking', torch.tensor(0.0)).item()

            # Track MCQ accuracy
            epoch_mcq_correct += loss_dict.get('mcq_correct', 0)
            epoch_mcq_total += loss_dict.get('mcq_total', 0)

            # Running accuracy for progress bar
            running_acc = 100.0 * epoch_mcq_correct / max(epoch_mcq_total, 1)

            postfix = {
                'loss': f"{loss.item():.4f}",
                'cat': f"{loss_dict['categorical'].item():.4f}",
                'acc': f"{running_acc:.1f}%",
            }
            if use_contrastive:
                postfix['contr'] = f"{contr_loss:.4f}"
            if obj_mask_loss > 0:
                postfix['mask'] = f"{obj_mask_loss:.4f}"
            pbar.set_postfix(postfix)

        avg_loss = total_loss / len(train_loader)
        avg_contr = contr_loss_sum / len(train_loader) if use_contrastive else 0
        train_acc = 100.0 * epoch_mcq_correct / max(epoch_mcq_total, 1)

        # ── Validation loss + accuracy every epoch (for overfitting detection) ──
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        with torch.no_grad():
            for vi, vbatch in enumerate(val_loader):
                if vi >= 50:  # Cap at 50 batches for speed
                    break
                vs = vbatch['states'].to(device)
                vm = vbatch['masks'].to(device)
                vq = vbatch['questions']
                va = vbatch['answers']
                vc = vbatch.get('choices')
                vci = vbatch.get('correct_choice_idx')
                vnt = {k: v.to(device) for k, v in vbatch['numerical_targets'].items()}
                vloss, vld = model.compute_combined_loss(
                    vs, vm, vq, va,
                    choices=vc, correct_choice_idx=vci,
                    numerical_targets=vnt
                )
                val_loss_sum += vloss.item()
                val_correct += vld.get('mcq_correct', 0)
                val_total += vld.get('mcq_total', 0)
                val_batches += 1
        model.train()

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)

        # Check for improvement (based on VAL loss to detect overfitting)
        improved = False
        if best_loss - avg_val_loss > min_delta:
            best_loss = avg_val_loss
            epochs_without_improvement = 0
            consecutive_catapults = 0  # reset escalation on improvement
            status = "improved"
            improved = True
        else:
            epochs_without_improvement += 1
            status = f"no improvement ({epochs_without_improvement}/{patience})"

        # Overfitting indicator: train loss << val loss
        gap = avg_val_loss - avg_loss
        gap_str = f"  gap={gap:+.4f}" if abs(gap) > 0.01 else ""
        overfit_warn = " ⚠️ OVERFITTING" if gap > 0.05 and epoch > 2 else ""

        # Current LR for display
        current_lr = optimizer.param_groups[0]['lr']
        lr_str = f"  lr={current_lr:.2e}" if catapult_active or current_lr != base_lr else ""

        print(f"Epoch {epoch+1} - Train: {avg_loss:.4f} ({train_acc:.1f}%) | Val: {avg_val_loss:.4f} ({val_acc:.1f}%, {val_correct}/{val_total}){gap_str}{lr_str}{overfit_warn} - {status}")

        # V3-only diagnostic: log realised Format-A/B ratio for the epoch.
        # Silently skipped on V2 (no mixed_format_stats method).
        if hasattr(model, 'mixed_format_stats'):
            mf = model.mixed_format_stats()
            print(f"  [mixed-format] format_b_fraction={mf['format_b_fraction']:.3f} "
                  f"(configured={mf['configured_prob']:.2f}, "
                  f"n_a={mf['n_format_a']}, n_b={mf['n_format_b']})")
            model.reset_mixed_format_stats()

        # ── Plateau catapult logic (responds to val stagnation) ──
        if catapult_active:
            # Decaying back from catapult boost
            catapult_epoch_counter += 1
            if catapult_epoch_counter >= catapult_decay_epochs:
                # Decay complete — restore baseline LR
                for pg in optimizer.param_groups:
                    pg['lr'] = base_lr
                catapult_active = False
                print(f"  [CATAPULT] Decay complete, LR restored to {base_lr:.2e}")
            else:
                # Exponential decay: boosted_lr * decay_factor^step
                decay_factor = (1.0 / current_catapult_multiplier) ** (1.0 / catapult_decay_epochs)
                new_lr = current_lr * decay_factor
                for pg in optimizer.param_groups:
                    pg['lr'] = new_lr
                print(f"  [CATAPULT] Decaying LR: {new_lr:.2e} (step {catapult_epoch_counter}/{catapult_decay_epochs})")
        elif not improved and epochs_without_improvement >= catapult_patience:
            # Plateau detected — launch catapult!
            catapult_count += 1
            consecutive_catapults += 1
            # Escalate: each consecutive catapult doubles the multiplier (capped)
            effective_multiplier = min(
                catapult_multiplier * (2.0 ** (consecutive_catapults - 1)),
                max_catapult_multiplier
            )
            current_catapult_multiplier = effective_multiplier  # store for decay
            boosted_lr = base_lr * effective_multiplier
            for pg in optimizer.param_groups:
                pg['lr'] = boosted_lr
            catapult_active = True
            catapult_epoch_counter = 0
            epochs_without_improvement = 0  # reset patience counter
            esc_str = f" (escalated {consecutive_catapults}x)" if consecutive_catapults > 1 else ""
            print(f"  [CATAPULT #{catapult_count}] Plateau detected! LR boosted {base_lr:.2e} → {boosted_lr:.2e} ({effective_multiplier:.0f}x){esc_str}")

        # ── Overfitting gap early stopping (responds to train-val divergence) ──
        if overfit_gap_threshold > 0 and epoch > 2:
            gap_val = avg_val_loss - avg_loss  # positive = overfitting
            if gap_val > overfit_gap_threshold:
                overfit_gap_counter += 1
                # On first consecutive overfit epoch, halve base LR (corrective)
                if overfit_gap_counter == 1:
                    old_base = base_lr
                    base_lr = max(base_lr * 0.5, 1e-6)
                    for pg in optimizer.param_groups:
                        pg['lr'] = base_lr
                    print(f"  [OVERFIT] Gap={gap_val:.4f} > {overfit_gap_threshold}, base LR halved: {old_base:.2e} → {base_lr:.2e}")
                if overfit_gap_counter >= overfit_gap_patience:
                    print(f"\n⚠️ OVERFIT GAP STOP: gap={gap_val:.4f} > {overfit_gap_threshold} for {overfit_gap_patience} consecutive epochs")
                    if best_ckpt_path and best_ckpt_path.exists():
                        print(f"  Loading best checkpoint: {best_ckpt_path.name}")
                        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                        model.load_state_dict(ckpt['model_state_dict'])
                    break
            else:
                overfit_gap_counter = 0

        # Physics usage validation (less frequent, more expensive)
        if (epoch + 1) % validate_every_n_epochs == 0:
            print("\n  [PHYSICS VALIDATION]")
            val_results = validate_physics_usage(model, val_loader, device)
            print(f"    Prefix cosine similarity (real vs zero): {val_results['avg_cosine_similarity']:.4f}")
            print(f"    Prefix difference norm: {val_results['avg_difference_norm']:.2f}")
            if val_results['avg_cosine_similarity'] > 0.95:
                print("    WARNING: High similarity suggests model NOT using physics!")
            elif val_results['avg_cosine_similarity'] < 0.8:
                print("    GOOD: Low similarity suggests model IS using physics!")
            print()

        if checkpoint_dir and ((epoch + 1) % save_every_n_epochs == 0 or improved):
            ckpt_path = checkpoint_dir / f"adapter_phase{phase_num}_epoch{epoch+1}_vloss{avg_val_loss:.4f}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'phase': phase_num,
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'best_loss': best_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'use_contrastive': use_contrastive,
                'catapult_count': catapult_count
            }, ckpt_path)
            if improved:
                best_ckpt_path = ckpt_path
            print(f"  Checkpoint saved: {ckpt_path.name}")

        if early_stopping and epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Final validation
    print("\n[FINAL PHYSICS VALIDATION]")
    val_results = validate_physics_usage(model, val_loader, device)
    print(f"  Prefix cosine similarity: {val_results['avg_cosine_similarity']:.4f}")
    print(f"  Prefix difference norm: {val_results['avg_difference_norm']:.2f}")

    return best_loss


def main():
    parser = argparse.ArgumentParser(description="Train Physics-LLM Adapter V2")
    parser.add_argument("--start-phase", type=int, default=1, choices=[1, 2],
                        help="Phase to start training from (1=adapter, 2=LoRA). Use to resume training.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from (loads model weights)")
    parser.add_argument("--physics-checkpoint", type=str, default=None,
                        help="Path to PhysicsFormer checkpoint (.pt file). If not specified, searches default locations.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to physics HDF5 data directory. Overrides config.data_dir if specified.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for saving adapter checkpoints. Defaults to checkpoint_dir/adapter")
    parser.add_argument("--clevrer-data", type=str, default=None,
                        help="Path to CLEVRER-conforming QA JSON (e.g., data/clevrer_conforming_qa.json)")
    parser.add_argument("--contrastive-weight", type=float, default=0.1,
                        help="Weight for contrastive loss (default: 0.1). Contrastive loss is always enabled to prevent physics collapse.")
    parser.add_argument("--validate-every", type=int, default=5,
                        help="Validate physics usage every N epochs (default: 5)")
    parser.add_argument("--no-validation", action="store_true",
                        help="Disable periodic physics usage validation")
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="Number of QA samples to generate (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size (default: 8)")
    parser.add_argument("--include-choices-prob", type=float, default=0.0,
                        help="Tier 1a (mixed-format training) — per-sample probability "
                             "of using Format B (`{question} Options: ... Answer:`) instead "
                             "of Format A (`{question} Answer:`). 0.0 (default) reproduces "
                             "V2 / Phase 3 byte-identically. 0.5 is the recipe in "
                             "ADAPTER_GENERALIZATION_PLAN.md. Setting >0 instantiates "
                             "PhysicsLLMAdapterV3 instead of V2; state-dict layout is "
                             "identical so adapter_phase3.pt warm-loads cleanly.")
    parser.add_argument("--mixed-format-seed", type=int, default=42,
                        help="Seed for the V3 Format-A/B Bernoulli RNG (default: 42).")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHYSICS-LLM ADAPTER V2 TRAINING")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if args.start_phase > 1:
        print(f"\n*** RESUMING FROM PHASE {args.start_phase} ***")
    
    config = TrainingConfig()

    # Override config paths if command-line arguments provided
    if args.data_dir:
        config.data_dir = args.data_dir
        print(f"[CONFIG] Using data directory: {args.data_dir}")

    print("\n[1/6] Loading physics dataset...")
    physics_dataset = HDF5PhysicsDataset(
        data_dir=config.data_dir,
        max_objects=config.max_objects,
        max_seq_length=32,
        schema_curriculum_level=8,
        hdf5_dir=config.data_dir
    )
    print(f"Loaded {len(physics_dataset):,} episodes\n")
    
    print("[2/6] Creating physics reasoning QA dataset...")
    if args.clevrer_data:
        clevrer_data_path = Path(args.clevrer_data)
    else:
        clevrer_data_path = Path(__file__).parent.parent / "data" / "clevrer_conforming_qa.json"
    cache_dir = Path(config.checkpoint_dir).parent / "cache"
    qa_dataset = PhysicsReasoningDataset(
        physics_dataset,
        num_samples=args.num_samples,
        include_metaphor_questions=False,
        metaphor_ratio=0.0,
        clevrer_data_path=str(clevrer_data_path) if clevrer_data_path.exists() else None,
        clevrer_ratio=0.5,
        cache_dir=str(cache_dir)
    )
    print(f"Generated {len(qa_dataset):,} QA samples")
    
    train_size = int(0.9 * len(qa_dataset))
    test_size = len(qa_dataset) - train_size
    split_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(
        qa_dataset, [train_size, test_size], generator=split_generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"Train: {len(train_dataset)} samples (batch_size={args.batch_size})")
    print(f"Test: {len(test_dataset)} samples\n")

    print("[3/6] Loading physics model...")

    checkpoint_dir = Path(config.checkpoint_dir)
    project_root = Path(__file__).parent.parent

    # Use command-line physics checkpoint if provided
    if args.physics_checkpoint:
        physics_checkpoint = Path(args.physics_checkpoint)
        if not physics_checkpoint.exists():
            raise FileNotFoundError(f"Physics checkpoint not found: {physics_checkpoint}")
        print(f"[CONFIG] Using physics checkpoint: {physics_checkpoint}")
    else:
        # Search default locations
        epoch385_checkpoint = project_root / "physics_epoch385_of_785.pt"
        causal_checkpoint = checkpoint_dir / "causal_finetuned_best.pt"
        stage1_checkpoint = checkpoint_dir / "stage1_best.pt"
        stage2_checkpoint = checkpoint_dir / "stage2_best.pt"
        physics_best_checkpoint = checkpoint_dir / "physics_former_best.pt"
        # Also check physics_former directory
        physics_former_best = project_root / "physics_former" / "physics_former_best.pt"

        if epoch385_checkpoint.exists():
            physics_checkpoint = epoch385_checkpoint
            print(f"Using physics_epoch385_of_785.pt checkpoint")
        elif causal_checkpoint.exists():
            physics_checkpoint = causal_checkpoint
            print("Using causal-finetuned checkpoint")
        elif stage2_checkpoint.exists():
            physics_checkpoint = stage2_checkpoint
            print("Using stage2_best.pt checkpoint")
        elif stage1_checkpoint.exists():
            physics_checkpoint = stage1_checkpoint
            print("Using stage1_best.pt checkpoint")
        elif physics_best_checkpoint.exists():
            physics_checkpoint = physics_best_checkpoint
            print("Using physics_former_best.pt checkpoint")
        elif physics_former_best.exists():
            physics_checkpoint = physics_former_best
            print(f"Using physics_former/physics_former_best.pt checkpoint")
        else:
            raise FileNotFoundError(
                f"No physics checkpoint found. Use --physics-checkpoint to specify path.\n"
                f"Searched: {epoch385_checkpoint}, {causal_checkpoint}, {physics_best_checkpoint}, {physics_former_best}"
            )

    print(f"Loading physics checkpoint: {physics_checkpoint}")
    checkpoint = torch.load(physics_checkpoint, map_location=device, weights_only=False)
    
    model_state = checkpoint['model_state_dict']
    
    has_orig_mod = any(k.startswith('_orig_mod.') for k in model_state.keys())
    prefix = '_orig_mod.' if has_orig_mod else ''
    
    schema_key = f'{prefix}schema_classifier.3.bias'
    num_schema_classes = model_state[schema_key].shape[0]
    
    hidden_dim_key = f'{prefix}transformer_layers.0.attention.q_proj.weight'
    hidden_dim = model_state[hidden_dim_key].shape[0]
    
    num_layers = sum(1 for k in model_state.keys() if f'{prefix}transformer_layers.' in k and '.attention.q_proj.weight' in k)
    
    num_heads_key = f'{prefix}transformer_layers.0.attention.attention_bias_net.2.bias'
    num_heads = model_state[num_heads_key].shape[0]
    
    use_swiglu = any('ff.w1.weight' in k for k in model_state.keys())
    use_rope = any('rope.inv_freq' in k for k in model_state.keys())
    use_rmsnorm = not any('norm1.bias' in k for k in model_state.keys())
    
    print(f"Detected from checkpoint: hidden_dim={hidden_dim}, num_layers={num_layers}, num_heads={num_heads}, schema_classes={num_schema_classes}")
    print(f"Architecture features: use_swiglu={use_swiglu}, use_rope={use_rope}, use_rmsnorm={use_rmsnorm}")
    
    physics_model = FullPhysicsFormer(
        state_dim=config.state_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=hidden_dim * 4,
        max_objects=config.max_objects,
        dropout=config.dropout,
        num_schema_classes=num_schema_classes
    ).to(device)
    
    class ModernConfig:
        use_rmsnorm = True
        use_flash_attention = True
        use_rope = True
        use_swiglu = True
        max_seq_length = 128
    
    if use_swiglu or use_rope or use_rmsnorm:
        modern_config = ModernConfig()
        modern_config.use_rmsnorm = use_rmsnorm
        modern_config.use_rope = use_rope
        modern_config.use_swiglu = use_swiglu
        physics_model = apply_modern_improvements(physics_model, modern_config, verbose=True)
    
    if has_orig_mod:
        cleaned_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}
    else:
        cleaned_state = model_state
    
    filtered_state = {k: v for k, v in cleaned_state.items() 
                      if 'rope.cos_cached' not in k and 'rope.sin_cached' not in k}
    physics_model.load_state_dict(filtered_state, strict=False)
    print(f"Physics model loaded successfully from: {physics_checkpoint}")
    
    if args.include_choices_prob > 0.0:
        print(f"\n[4/6] Creating adapter V3 (mixed-format training, "
              f"include_choices_prob={args.include_choices_prob}) ...")
        adapter = create_adapter_v3(
            physics_model=physics_model,
            physics_dim=hidden_dim,
            num_prefix_tokens=16,
            freeze_physics=True,
            freeze_llm=True,
            include_choices_prob=args.include_choices_prob,
            mixed_format_seed=args.mixed_format_seed,
        ).to(device)
    else:
        print("\n[4/6] Creating adapter V2 with DistilGPT-2 (edge-optimized)...")
        adapter = create_adapter_v2(
            physics_model=physics_model,
            physics_dim=hidden_dim,
            num_prefix_tokens=16,
            freeze_physics=True,
            freeze_llm=True
        ).to(device)
    
    # Set output directory for adapter checkpoints
    if args.output_dir:
        adapter_checkpoint_dir = Path(args.output_dir)
    else:
        adapter_checkpoint_dir = Path(config.checkpoint_dir) / "adapter"
    adapter_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CONFIG] Saving checkpoints to: {adapter_checkpoint_dir}")

    # Load checkpoint if resuming
    if args.checkpoint:
        print(f"\n[RESUME] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        adapter.load_state_dict(ckpt['model_state_dict'])
        print(f"[RESUME] Loaded from phase {ckpt.get('phase', '?')}, epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?'):.4f}")
    
    print("\n[5/6] Training...")

    # Determine whether to use validation
    use_validation = not args.no_validation
    train_func = train_phase_with_validation if use_validation else train_phase

    print(f"[CONFIG] Contrastive loss: ENABLED (weight={args.contrastive_weight})")
    if use_validation:
        print(f"[CONFIG] Physics validation: every {args.validate_every} epochs")
    else:
        print("[CONFIG] Physics validation: DISABLED")

    # Phase 1: Adapter + Numerical Head
    if args.start_phase <= 1:
        adapter.set_training_phase('adapter')
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, adapter.parameters()),
            lr=1e-4,
            weight_decay=0.01
        )
        if use_validation:
            train_phase_with_validation(
                adapter, train_loader, test_loader, optimizer, device,
                phase_name="Phase 1: Adapter + Numerical Head",
                max_epochs=20, patience=5, min_delta=0.001,
                checkpoint_dir=adapter_checkpoint_dir, phase_num=1, save_every_n_epochs=3,
                use_contrastive=True, contrastive_weight=args.contrastive_weight,
                validate_every_n_epochs=args.validate_every
            )
        else:
            train_phase(
                adapter, train_loader, optimizer, device,
                phase_name="Phase 1: Adapter + Numerical Head",
                max_epochs=20, patience=5, min_delta=0.001,
                checkpoint_dir=adapter_checkpoint_dir, phase_num=1, save_every_n_epochs=3,
                use_contrastive=True, contrastive_weight=args.contrastive_weight
            )
    else:
        print(f"[SKIP] Phase 1 (starting from phase {args.start_phase})")

    # Phase 2: LoRA on LLM attention (replaces old Phases 2+3)
    # ~0.2M extra trainable params vs 60-80M from full unfreezing
    adapter.set_training_phase('lora', lora_rank=8, lora_alpha=16.0)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, adapter.parameters()),
        lr=5e-5,
        weight_decay=0.01
    )
    if use_validation:
        train_phase_with_validation(
            adapter, train_loader, test_loader, optimizer, device,
            phase_name="Phase 2: Adapter + LoRA",
            max_epochs=30, patience=5, min_delta=0.0005,
            checkpoint_dir=adapter_checkpoint_dir, phase_num=2, save_every_n_epochs=5,
            use_contrastive=True, contrastive_weight=args.contrastive_weight,
            validate_every_n_epochs=args.validate_every
        )
    else:
        train_phase(
            adapter, train_loader, optimizer, device,
            phase_name="Phase 2: Adapter + LoRA",
            max_epochs=30, patience=5, min_delta=0.0005,
            checkpoint_dir=adapter_checkpoint_dir, phase_num=2, save_every_n_epochs=5,
            use_contrastive=True, contrastive_weight=args.contrastive_weight
        )

    # Merge LoRA weights into base LLM for clean inference
    adapter.merge_lora()
    
    print("\n[6/6] Final evaluation...")
    accuracy = evaluate(adapter, test_loader, device, num_samples=500)
    
    save_path = adapter_checkpoint_dir / "adapter_v2_best.pt"
    torch.save({
        'model_state_dict': adapter.state_dict(),
        'llm_name': 'distilgpt2',
        'accuracy': accuracy,
        'question_types': [qt.value for qt in PHYSICS_QUESTION_TYPES]
    }, save_path)
    print(f"\nAdapter saved to: {save_path}")
    
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE - Final Accuracy: {accuracy:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
