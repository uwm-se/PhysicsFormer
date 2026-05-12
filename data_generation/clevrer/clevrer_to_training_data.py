"""
Copyright (c) 2026 Style Machine LLC. All rights reserved.

PROPRIETARY AND CONFIDENTIAL. This software is provided for academic review
and research purposes only. Unauthorized copying, modification, distribution,
or use of this software, via any medium, is strictly prohibited without prior
written permission from Style Machine LLC.
"""

"""
CLEVRER to Training Data Converter

Converts CLEVRER benchmark questions into the PhysicsReasoningDataset format
for training the Physics-LLM adapter on CLEVRER-style questions.

Output format: HDF5 for reproducibility and cross-platform compatibility.

New question types added:
- CAUSAL_CHAIN: "What caused X?" (explanatory)
- FUTURE_PREDICTION: "What will happen next?" (predictive)
- COUNTERFACTUAL_REASONING: "What if X were removed?" (counterfactual)
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm

import numpy as np
import torch
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

from clevrer_benchmark.scene_converter import (
    load_clevrer_scene,
    clevrer_scene_to_state_tensor
)


class CLEVRERQuestionType(Enum):
    CAUSAL_CHAIN = "causal_chain"
    FUTURE_PREDICTION = "future_prediction"
    COUNTERFACTUAL_REASONING = "counterfactual_reasoning"
    # Descriptive subtypes for multi-head classifier
    DESCRIPTIVE_COUNT = "descriptive_count"
    DESCRIPTIVE_EXIST = "descriptive_exist"
    DESCRIPTIVE_COLOR = "descriptive_color"
    DESCRIPTIVE_SHAPE = "descriptive_shape"
    DESCRIPTIVE_MATERIAL = "descriptive_material"


@dataclass
class CLEVRERSample:
    states: torch.Tensor
    mask: torch.Tensor
    question: str
    answer: str
    question_type: str
    metadata: Dict
    numerical_targets: Dict


def extract_answer_from_choices(question: Dict) -> Tuple[str, str]:
    """Extract the correct answer from CLEVRER multiple-choice format."""
    choices = question.get('choices', [])
    q_text = question.get('question', '')
    
    correct_choices = []
    wrong_choices = []
    
    for choice in choices:
        choice_text = choice.get('choice', '')
        answer_label = choice.get('answer', '')
        
        if answer_label == 'correct':
            correct_choices.append(choice_text)
        elif answer_label == 'wrong':
            wrong_choices.append(choice_text)
    
    if correct_choices:
        answer = correct_choices[0]
        formatted_q = f"{q_text} Options: {', '.join([c.get('choice', '') for c in choices])}"
        return formatted_q, answer
    
    return q_text, question.get('answer', 'unknown')


def classify_descriptive_subtype(question_text: str) -> str:
    """Classify a descriptive question into its subtype for multi-head training."""
    q_lower = question_text.lower()
    
    if "how many" in q_lower:
        return CLEVRERQuestionType.DESCRIPTIVE_COUNT.value
    elif "what color" in q_lower or "what is the color" in q_lower:
        return CLEVRERQuestionType.DESCRIPTIVE_COLOR.value
    elif "what shape" in q_lower or "what is the shape" in q_lower:
        return CLEVRERQuestionType.DESCRIPTIVE_SHAPE.value
    elif "what material" in q_lower or "what is the material" in q_lower:
        return CLEVRERQuestionType.DESCRIPTIVE_MATERIAL.value
    else:
        # Default to exist for yes/no questions
        return CLEVRERQuestionType.DESCRIPTIVE_EXIST.value


def convert_descriptive_question(question: Dict, metadata: Dict) -> Dict:
    """Convert CLEVRER descriptive question to training format."""
    q_text = question.get('question', '')
    answer = question.get('answer', '')
    
    # Handle program-based answers (CLEVRER uses programs for descriptive)
    if isinstance(answer, list):
        answer = str(answer[0]) if answer else "unknown"
    answer = str(answer).lower().strip()
    
    # Classify subtype for multi-head routing
    subtype = classify_descriptive_subtype(q_text)
    
    return {
        'question': q_text,
        'answer': answer,
        'question_type': subtype,
        'metadata': {
            'clevrer_type': 'descriptive',
            'descriptive_subtype': subtype,
            'original_question': q_text,
        },
        'numerical_targets': {
            'distance': 0.0,
            'speed': 0.0,
            'time_to_collision': 0.0,
            'kinetic_energy': 0.0,
            'momentum': 0.0,
            'object_count': float(metadata.get('num_objects', 0)),
        }
    }


def convert_explanatory_question(question: Dict, metadata: Dict) -> Dict:
    """Convert CLEVRER explanatory question to training format."""
    q_text, answer = extract_answer_from_choices(question)
    
    templates = [
        f"Based on the physics of this scene: {q_text}",
        f"Analyzing the causal relationships: {q_text}",
        f"Considering what caused the events: {q_text}",
    ]
    
    formatted_question = random.choice(templates)
    
    return {
        'question': formatted_question,
        'answer': answer,
        'question_type': CLEVRERQuestionType.CAUSAL_CHAIN.value,
        'metadata': {
            'clevrer_type': 'explanatory',
            'original_question': question.get('question', ''),
            'choices': question.get('choices', []),
        },
        'numerical_targets': {
            'distance': 0.0,
            'speed': 0.0,
            'time_to_collision': 0.0,
            'kinetic_energy': 0.0,
            'momentum': 0.0,
            'object_count': float(metadata.get('num_objects', 0)),
        }
    }


def convert_predictive_question(question: Dict, metadata: Dict) -> Dict:
    """Convert CLEVRER predictive question to training format."""
    q_text, answer = extract_answer_from_choices(question)
    
    templates = [
        f"Predicting future events: {q_text}",
        f"Based on current trajectories: {q_text}",
        f"What will happen next? {q_text}",
    ]
    
    formatted_question = random.choice(templates)
    
    return {
        'question': formatted_question,
        'answer': answer,
        'question_type': CLEVRERQuestionType.FUTURE_PREDICTION.value,
        'metadata': {
            'clevrer_type': 'predictive',
            'original_question': question.get('question', ''),
            'choices': question.get('choices', []),
        },
        'numerical_targets': {
            'distance': 0.0,
            'speed': 0.0,
            'time_to_collision': 0.0,
            'kinetic_energy': 0.0,
            'momentum': 0.0,
            'object_count': float(metadata.get('num_objects', 0)),
        }
    }


def convert_counterfactual_question(question: Dict, metadata: Dict) -> Dict:
    """Convert CLEVRER counterfactual question to training format."""
    q_text, answer = extract_answer_from_choices(question)
    
    templates = [
        f"In a hypothetical scenario: {q_text}",
        f"Reasoning about alternatives: {q_text}",
        f"If we changed the scene: {q_text}",
    ]
    
    formatted_question = random.choice(templates)
    
    return {
        'question': formatted_question,
        'answer': answer,
        'question_type': CLEVRERQuestionType.COUNTERFACTUAL_REASONING.value,
        'metadata': {
            'clevrer_type': 'counterfactual',
            'original_question': question.get('question', ''),
            'choices': question.get('choices', []),
        },
        'numerical_targets': {
            'distance': 0.0,
            'speed': 0.0,
            'time_to_collision': 0.0,
            'kinetic_energy': 0.0,
            'momentum': 0.0,
            'object_count': float(metadata.get('num_objects', 0)),
        }
    }


def convert_clevrer_scene_questions(
    scene_path: Path,
    questions: List[Dict],
    include_descriptive: bool = True
) -> List[Dict]:
    """Convert all questions for a single scene (including descriptive)."""
    try:
        scene = load_clevrer_scene(str(scene_path))
        states, masks, metadata = clevrer_scene_to_state_tensor(scene)
    except Exception as e:
        print(f"Error loading scene {scene_path}: {e}")
        return []
    
    states_tensor = torch.from_numpy(states).float()
    mask_tensor = torch.from_numpy(masks).float()
    
    samples = []
    
    for question in questions:
        q_type = question.get('question_type', 'descriptive')
        
        if q_type == 'descriptive':
            if not include_descriptive:
                continue
            sample_data = convert_descriptive_question(question, metadata)
        elif q_type == 'explanatory':
            sample_data = convert_explanatory_question(question, metadata)
        elif q_type == 'predictive':
            sample_data = convert_predictive_question(question, metadata)
        elif q_type == 'counterfactual':
            sample_data = convert_counterfactual_question(question, metadata)
        else:
            continue
        
        sample_data['states'] = states_tensor
        sample_data['mask'] = mask_tensor
        samples.append(sample_data)
    
    return samples


def convert_clevrer_dataset(
    clevrer_dir: Path,
    max_scenes: Optional[int] = None,
    split: str = 'validation',
    include_descriptive: bool = True
) -> List[Dict]:
    """Convert CLEVRER dataset to training format (including descriptive questions)."""
    scene_dir = clevrer_dir / 'scenes'
    questions_file = clevrer_dir / 'questions' / f'{split}.json'
    
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    
    print(f"Loading CLEVRER questions from {questions_file}...")
    with open(questions_file, 'r') as f:
        all_questions = json.load(f)
    
    questions_by_scene = {}
    for scene_data in all_questions:
        scene_idx = scene_data.get('scene_index', scene_data.get('video_index', 0))
        questions_by_scene[scene_idx] = scene_data.get('questions', [])
    
    # Scene files are in subdirectories like annotation_10000-11000/
    scene_files = sorted(scene_dir.glob('**/*.json'))
    if max_scenes:
        scene_files = scene_files[:max_scenes]
    
    print(f"Converting {len(scene_files)} scenes...")
    
    all_samples = []
    stats = {'descriptive': 0, 'explanatory': 0, 'predictive': 0, 'counterfactual': 0}
    
    for scene_file in tqdm(scene_files, desc="Converting scenes"):
        scene_id = scene_file.stem
        scene_num = int(scene_id.split('_')[-1]) if '_' in scene_id else 0
        
        questions = questions_by_scene.get(scene_num, [])
        if not questions:
            continue
        
        samples = convert_clevrer_scene_questions(scene_file, questions, include_descriptive)
        
        for sample in samples:
            clevrer_type = sample['metadata'].get('clevrer_type', 'unknown')
            if clevrer_type in stats:
                stats[clevrer_type] += 1
        
        all_samples.extend(samples)
    
    print(f"\nConverted {len(all_samples)} CLEVRER samples:")
    for q_type, count in stats.items():
        print(f"  {q_type}: {count}")
    
    return all_samples


def save_clevrer_training_data(
    samples: List[Dict],
    output_path: Path
):
    """Save converted CLEVRER samples to HDF5 format."""
    # Change extension to .h5
    if output_path.suffix == '.json':
        output_path = output_path.with_suffix('.h5')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(samples)
    if n_samples == 0:
        print("No samples to save")
        return

    # Scan ALL samples for max object count (scenes have variable numbers)
    seq_len = samples[0]['states'].shape[0]
    state_dim = samples[0]['states'].shape[2]
    max_objects = 0
    object_count_hist = {}
    for s in samples:
        n_obj = s['states'].shape[1]
        max_objects = max(max_objects, n_obj)
        object_count_hist[n_obj] = object_count_hist.get(n_obj, 0) + 1

    print(f"Object count distribution across {n_samples:,} samples:")
    for n_obj in sorted(object_count_hist.keys()):
        count = object_count_hist[n_obj]
        pct = 100 * count / n_samples
        print(f"  {n_obj} objects: {count:,} samples ({pct:.1f}%)")
    print(f"Using max_objects={max_objects} (no truncation)")

    state_shape = (seq_len, max_objects, state_dim)
    mask_shape = (seq_len, max_objects)

    with h5py.File(output_path, 'w') as f:
        # Create datasets for tensor data
        states_ds = f.create_dataset(
            'states',
            shape=(n_samples,) + state_shape,
            dtype='float32',
            compression='gzip'
        )
        masks_ds = f.create_dataset(
            'masks',
            shape=(n_samples,) + mask_shape,
            dtype='float32',
            compression='gzip'
        )

        # String data stored as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        questions_ds = f.create_dataset('questions', shape=(n_samples,), dtype=dt)
        answers_ds = f.create_dataset('answers', shape=(n_samples,), dtype=dt)
        question_types_ds = f.create_dataset('question_types', shape=(n_samples,), dtype=dt)
        metadata_ds = f.create_dataset('metadata', shape=(n_samples,), dtype=dt)
        numerical_targets_ds = f.create_dataset('numerical_targets', shape=(n_samples,), dtype=dt)

        # Fill datasets (pad smaller scenes to max_objects)
        padded_count = 0
        for i, sample in enumerate(tqdm(samples, desc="Saving to HDF5")):
            states_np = sample['states'].numpy()
            mask_np = sample['mask'].numpy()
            n_obj = states_np.shape[1]

            if n_obj < max_objects:
                pad_obj = max_objects - n_obj
                states_np = np.pad(states_np, ((0, 0), (0, pad_obj), (0, 0)))
                mask_np = np.pad(mask_np, ((0, 0), (0, pad_obj)))
                padded_count += 1

            states_ds[i] = states_np
            masks_ds[i] = mask_np
            questions_ds[i] = sample['question']
            answers_ds[i] = sample['answer']
            question_types_ds[i] = sample['question_type']
            metadata_ds[i] = json.dumps(sample['metadata'])
            numerical_targets_ds[i] = json.dumps(sample['numerical_targets'])

        # Store metadata (include individual dims for notebook compatibility)
        f.attrs['num_samples'] = n_samples
        f.attrs['seq_len'] = seq_len
        f.attrs['num_objects'] = max_objects
        f.attrs['state_dim'] = state_dim
        f.attrs['state_shape'] = state_shape
        f.attrs['format_version'] = '2.0'

    print(f"  Padded {padded_count:,} samples (had fewer than {max_objects} objects)")
    print(f"Saved {n_samples} samples to {output_path}")


def load_clevrer_training_data(input_path: Path) -> List[Dict]:
    """Load converted CLEVRER samples from HDF5 format."""
    # Support both .h5 and .json (legacy)
    if input_path.suffix == '.json':
        input_path = input_path.with_suffix('.h5')

    samples = []

    with h5py.File(input_path, 'r') as f:
        n_samples = f.attrs['num_samples']

        for i in range(n_samples):
            sample = {
                'states': torch.tensor(f['states'][i], dtype=torch.float32),
                'mask': torch.tensor(f['masks'][i], dtype=torch.float32),
                'question': f['questions'][i],
                'answer': f['answers'][i],
                'question_type': f['question_types'][i],
                'metadata': json.loads(f['metadata'][i]),
                'numerical_targets': json.loads(f['numerical_targets'][i]),
            }
            samples.append(sample)

    return samples


def convert_clevrer_dataset_batched(
    clevrer_dir: Path,
    output_path: Path,
    batch_size: int = 1000,
    include_descriptive: bool = True
):
    """Convert CLEVRER dataset in batches, saving incrementally to HDF5."""
    import gc

    # Change extension to .h5
    if output_path.suffix == '.json':
        output_path = output_path.with_suffix('.h5')

    scene_dir = clevrer_dir / 'scenes'
    questions_file = clevrer_dir / 'questions' / 'validation.json'

    print(f"Loading CLEVRER questions from {questions_file}...")
    with open(questions_file, 'r') as f:
        all_questions = json.load(f)

    questions_by_scene = {}
    for scene_data in all_questions:
        scene_idx = scene_data.get('scene_index', scene_data.get('video_index', 0))
        questions_by_scene[scene_idx] = scene_data.get('questions', [])

    scene_files = sorted(scene_dir.glob('**/*.json'))
    total_scenes = len(scene_files)
    print(f"Found {total_scenes} scenes, processing in batches of {batch_size}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {'descriptive': 0, 'explanatory': 0, 'predictive': 0, 'counterfactual': 0}
    total_samples = 0

    # First pass: collect all samples and track max object count
    all_samples = []
    max_objects = 0
    object_count_hist = {}

    for batch_start in range(0, total_scenes, batch_size):
        batch_end = min(batch_start + batch_size, total_scenes)
        batch_files = scene_files[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1

        print(f"\nProcessing batch {batch_num}: scenes {batch_start}-{batch_end}")

        for scene_file in tqdm(batch_files, desc=f"Batch {batch_num}"):
            scene_id = scene_file.stem
            scene_num = int(scene_id.split('_')[-1]) if '_' in scene_id else 0

            questions = questions_by_scene.get(scene_num, [])
            if not questions:
                continue

            try:
                samples = convert_clevrer_scene_questions(scene_file, questions, include_descriptive)

                for sample in samples:
                    clevrer_type = sample['metadata'].get('clevrer_type', 'unknown')
                    if clevrer_type in stats:
                        stats[clevrer_type] += 1

                    n_obj = sample['states'].shape[1]
                    max_objects = max(max_objects, n_obj)
                    object_count_hist[n_obj] = object_count_hist.get(n_obj, 0) + 1

                    all_samples.append(sample)
                    total_samples += 1
            except Exception as e:
                print(f"  Error processing {scene_file}: {e}")
                continue

        gc.collect()
        print(f"  Total samples so far: {total_samples} (max objects: {max_objects})")

    # Report object count distribution
    print(f"\nObject count distribution across {total_samples:,} samples:")
    for n_obj in sorted(object_count_hist.keys()):
        count = object_count_hist[n_obj]
        pct = 100 * count / total_samples
        print(f"  {n_obj} objects: {count:,} samples ({pct:.1f}%)")
    print(f"Using max_objects={max_objects} (no truncation)")

    # Second pass: write to HDF5 with uniform shape
    if all_samples and max_objects > 0:
        seq_len = all_samples[0]['states'].shape[0]
        state_dim = all_samples[0]['states'].shape[2]
        state_shape = (seq_len, max_objects, state_dim)
        mask_shape = (seq_len, max_objects)

        print(f"\nWriting {total_samples} samples to HDF5 with shape {state_shape}...")

        with h5py.File(output_path, 'w') as f:
            # Create datasets
            states_ds = f.create_dataset(
                'states', shape=(total_samples,) + state_shape,
                dtype='float32', compression='gzip'
            )
            masks_ds = f.create_dataset(
                'masks', shape=(total_samples,) + mask_shape,
                dtype='float32', compression='gzip'
            )

            dt = h5py.special_dtype(vlen=str)
            questions_ds = f.create_dataset('questions', shape=(total_samples,), dtype=dt)
            answers_ds = f.create_dataset('answers', shape=(total_samples,), dtype=dt)
            question_types_ds = f.create_dataset('question_types', shape=(total_samples,), dtype=dt)
            metadata_ds = f.create_dataset('metadata', shape=(total_samples,), dtype=dt)
            numerical_targets_ds = f.create_dataset('numerical_targets', shape=(total_samples,), dtype=dt)

            padded_count = 0
            for i, sample in enumerate(tqdm(all_samples, desc="Writing HDF5")):
                states_np = sample['states'].numpy()
                mask_np = sample['mask'].numpy()
                n_obj = states_np.shape[1]

                if n_obj < max_objects:
                    pad_obj = max_objects - n_obj
                    states_np = np.pad(states_np, ((0, 0), (0, pad_obj), (0, 0)))
                    mask_np = np.pad(mask_np, ((0, 0), (0, pad_obj)))
                    padded_count += 1

                states_ds[i] = states_np
                masks_ds[i] = mask_np
                questions_ds[i] = sample['question']
                answers_ds[i] = sample['answer']
                question_types_ds[i] = sample['question_type']
                metadata_ds[i] = json.dumps(sample['metadata'])
                numerical_targets_ds[i] = json.dumps(sample['numerical_targets'])

            f.attrs['num_samples'] = total_samples
            f.attrs['seq_len'] = seq_len
            f.attrs['num_objects'] = max_objects
            f.attrs['state_dim'] = state_dim
            f.attrs['state_shape'] = state_shape
            f.attrs['format_version'] = '2.0'

        print(f"  Padded {padded_count:,} samples (had fewer than {max_objects} objects)")

    print(f"\nFinal stats:")
    for q_type, count in stats.items():
        print(f"  {q_type}: {count}")
    print(f"Total: {total_samples}")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CLEVRER to training data')
    parser.add_argument('--clevrer_dir', type=str, default='D:\\clevrer',
                        help='Path to CLEVRER dataset')
    parser.add_argument('--output', type=str,
                        default='D:\\physics-former-data\\clevrer_training_data.h5',
                        help='Output path for training data (HDF5 format)')
    parser.add_argument('--max_scenes', type=int, default=None,
                        help='Maximum scenes to convert')
    parser.add_argument('--batch', action='store_true',
                        help='Use batched processing for large datasets')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for batched processing')
    
    args = parser.parse_args()
    
    if args.batch:
        convert_clevrer_dataset_batched(
            clevrer_dir=Path(args.clevrer_dir),
            output_path=Path(args.output),
            batch_size=args.batch_size
        )
    else:
        samples = convert_clevrer_dataset(
            clevrer_dir=Path(args.clevrer_dir),
            max_scenes=args.max_scenes
        )
        save_clevrer_training_data(samples, Path(args.output))
