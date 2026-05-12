"""
Regenerate CLEVRER HDF5 with correct max_objects.

Streams scenes directly to HDF5 — never holds more than one scene in RAM.
Uses max_objects=6 (pre-scanned from 5000 CLEVRER validation scenes).
"""
import os
import sys, json, os, gc
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'clevrer_benchmark'))

from clevrer_benchmark.scene_converter import load_clevrer_scene, clevrer_scene_to_state_tensor

# --- Config ---
CLEVRER_DIR = Path(os.environ.get('CLEVRER_DIR', 'clevrer'))
QUESTIONS_FILE = CLEVRER_DIR / 'questions' / 'clevrer_validation.json'
SCENE_ROOT = CLEVRER_DIR / 'annotations' / 'validation'
OUTPUT_PATH = Path('$PHYSICS_DATA_DIR/clevrer_training_expanded_v2.h5')

MAX_OBJECTS = 6   # Pre-scanned: scenes have 3-6 objects
SEQ_LEN = 128
STATE_DIM = 35


def classify_question(q):
    """Simple CLEVRER question type classifier."""
    qt = q.get('question_type', 'descriptive')
    return qt


def extract_answer(question):
    """Extract answer text from CLEVRER question."""
    # For MC questions, find the correct choice
    choices = question.get('choices', [])
    if choices:
        for c in choices:
            if c.get('answer') == 'correct':
                return c.get('choice', '')
    # For open-ended
    return question.get('answer', '')


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load questions
    print(f'Loading questions from {QUESTIONS_FILE}...')
    with open(QUESTIONS_FILE) as f:
        all_questions = json.load(f)

    questions_by_scene = {}
    for sd in all_questions:
        idx = sd.get('scene_index', sd.get('video_index', 0))
        questions_by_scene[idx] = sd.get('questions', [])
    del all_questions

    # Find scene files
    scene_files = sorted(SCENE_ROOT.glob('**/*.json'))
    print(f'Found {len(scene_files)} scenes, {len(questions_by_scene)} with questions')

    # --- Pass 1: Count total samples (quick, no tensors) ---
    print('Pass 1: Counting samples...')
    total_samples = 0
    scene_question_counts = {}
    for sf in tqdm(scene_files, desc='Counting'):
        scene_num = int(sf.stem.split('_')[-1])
        qs = questions_by_scene.get(scene_num, [])
        n = len(qs)
        if n > 0:
            scene_question_counts[scene_num] = (sf, n)
            total_samples += n
    print(f'Total samples: {total_samples:,} across {len(scene_question_counts)} scenes')

    # --- Create HDF5 ---
    print(f'\nPass 2: Converting and writing to {OUTPUT_PATH}...')
    print(f'  Shape: ({total_samples}, {SEQ_LEN}, {MAX_OBJECTS}, {STATE_DIM})')

    dt = h5py.special_dtype(vlen=str)

    with h5py.File(str(OUTPUT_PATH), 'w') as hf:
        states_ds = hf.create_dataset(
            'states', shape=(total_samples, SEQ_LEN, MAX_OBJECTS, STATE_DIM),
            dtype='float32', chunks=(min(100, total_samples), SEQ_LEN, MAX_OBJECTS, STATE_DIM),
            compression='gzip', compression_opts=4)
        masks_ds = hf.create_dataset(
            'masks', shape=(total_samples, SEQ_LEN, MAX_OBJECTS),
            dtype='float32', chunks=(min(100, total_samples), SEQ_LEN, MAX_OBJECTS),
            compression='gzip')
        questions_ds = hf.create_dataset('questions', shape=(total_samples,), dtype=dt)
        answers_ds = hf.create_dataset('answers', shape=(total_samples,), dtype=dt)
        question_types_ds = hf.create_dataset('question_types', shape=(total_samples,), dtype=dt)
        metadata_ds = hf.create_dataset('metadata', shape=(total_samples,), dtype=dt)
        numerical_targets_ds = hf.create_dataset(
            'numerical_targets', shape=(total_samples, 2),
            dtype='float32', compression='gzip')

        idx = 0
        obj_hist = Counter()
        errors = 0

        for scene_num, (sf, n_questions) in tqdm(scene_question_counts.items(),
                                                   desc='Writing', total=len(scene_question_counts)):
            try:
                scene = load_clevrer_scene(str(sf))
                states, masks, meta = clevrer_scene_to_state_tensor(scene)
                # states: (T, N, 35), masks: (T, N)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f'  Error loading {sf.name}: {e}')
                # Skip all questions for this scene
                for _ in range(n_questions):
                    # Write empty placeholder
                    states_ds[idx] = np.zeros((SEQ_LEN, MAX_OBJECTS, STATE_DIM), dtype=np.float32)
                    masks_ds[idx] = np.zeros((SEQ_LEN, MAX_OBJECTS), dtype=np.float32)
                    questions_ds[idx] = ''
                    answers_ds[idx] = ''
                    question_types_ds[idx] = 'error'
                    metadata_ds[idx] = '{}'
                    numerical_targets_ds[idx] = [0.0, 0.0]
                    idx += 1
                continue

            n_obj = states.shape[1]
            obj_hist[n_obj] += 1

            # Pad/truncate to uniform shape
            T = states.shape[0]
            padded_states = np.zeros((SEQ_LEN, MAX_OBJECTS, STATE_DIM), dtype=np.float32)
            padded_masks = np.zeros((SEQ_LEN, MAX_OBJECTS), dtype=np.float32)

            t_end = min(T, SEQ_LEN)
            o_end = min(n_obj, MAX_OBJECTS)
            padded_states[:t_end, :o_end, :] = states[:t_end, :o_end, :]
            padded_masks[:t_end, :o_end] = masks[:t_end, :o_end]

            questions = questions_by_scene[scene_num]
            num_objects_float = float(n_obj)

            for q in questions:
                qt = classify_question(q)
                answer = extract_answer(q)
                q_text = q.get('question', '')
                choices = q.get('choices', [])

                meta_json = json.dumps({
                    'clevrer_type': qt,
                    'original_question': q_text,
                    'choices': choices,
                    'num_objects': n_obj,
                    'scene_index': scene_num,
                })

                states_ds[idx] = padded_states
                masks_ds[idx] = padded_masks
                questions_ds[idx] = q_text
                answers_ds[idx] = answer
                question_types_ds[idx] = qt
                metadata_ds[idx] = meta_json
                numerical_targets_ds[idx] = [num_objects_float, 0.0]
                idx += 1

            # Free scene data immediately
            del states, masks, meta, padded_states, padded_masks, scene

            if idx % 10000 == 0:
                gc.collect()

        # Store attrs
        hf.attrs['num_samples'] = total_samples
        hf.attrs['seq_len'] = SEQ_LEN
        hf.attrs['num_objects'] = MAX_OBJECTS
        hf.attrs['state_dim'] = STATE_DIM
        hf.attrs['format_version'] = '2.0'

    print(f'\nDone! Wrote {idx} samples to {OUTPUT_PATH}')
    print(f'Object count distribution (per scene):')
    for n in sorted(obj_hist.keys()):
        c = obj_hist[n]
        print(f'  {n} objects: {c:,} scenes')
    print(f'Errors: {errors}')

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f'File size: {size_mb:.1f} MB')


if __name__ == '__main__':
    main()
