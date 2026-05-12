"""
CLEVRER Benchmark: LLMs WITH Scene Information

Provides LLMs with the same physical information Physics-LLM receives,
converted to text format. This is a fair comparison.

Usage:
    python run_llm_with_scene.py --model gpt4 --max_questions 100
    python run_llm_with_scene.py --model gemini --max_questions 100
    python run_llm_with_scene.py --model llama-70b --max_questions 100  # via Together AI

    # Predictive-only supplementary evaluation (tighter CIs on the predictive headline):
    python run_llm_with_scene.py --model gpt4 --question_types predictive --max_questions 500
"""

import argparse
import json
import zipfile
import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time

# API clients
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Model configurations for API-based inference
MODEL_CONFIGS = {
    "gpt4": {"provider": "openai", "model_id": "gpt-4o", "display_name": "GPT-4o"},
    "gpt4-turbo": {"provider": "openai", "model_id": "gpt-4-turbo", "display_name": "GPT-4 Turbo"},
    "gpt5": {"provider": "openai", "model_id": "gpt-5.2", "display_name": "GPT-5.2"},
    "gemini": {"provider": "gemini", "model_id": "gemini-2.0-flash", "display_name": "Gemini 2.0 Flash"},
    "gemini-pro": {"provider": "gemini", "model_id": "gemini-1.5-pro-latest", "display_name": "Gemini 1.5 Pro"},
    # Anthropic Claude
    "claude": {"provider": "anthropic", "model_id": "claude-sonnet-4-20250514", "display_name": "Claude Sonnet 4"},
    "claude-4.5": {"provider": "anthropic", "model_id": "claude-sonnet-4-5-20250929", "display_name": "Claude 4.5 Sonnet"},
    "opus-4.5": {"provider": "anthropic", "model_id": "claude-opus-4-5-20251101", "display_name": "Claude Opus 4.5", "thinking": True},
    # Together AI models (serverless endpoints)
    "llama-70b": {"provider": "together", "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "display_name": "Llama-3.1-70B"},
    "llama-3.3-70b": {"provider": "together", "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "display_name": "Llama-3.3-70B"},
    "llama-8b": {"provider": "together", "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "display_name": "Llama-3.1-8B"},
    "mixtral": {"provider": "together", "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "display_name": "Mixtral-8x7B"},
    "qwen-72b": {"provider": "together", "model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "display_name": "Qwen3-235B (22B active)"},
    "qwen-7b": {"provider": "together", "model_id": "Qwen/Qwen2.5-7B-Instruct-Turbo", "display_name": "Qwen2.5-7B"},
    "deepseek-v3": {"provider": "together", "model_id": "deepseek-ai/DeepSeek-V3", "display_name": "DeepSeek-V3"},
    # Fireworks AI models (alternative)
    "llama-70b-fw": {"provider": "fireworks", "model_id": "accounts/fireworks/models/llama-v3-70b-instruct", "display_name": "Llama-3-70B (FW)"},
}


CLEVRER_DIR = Path(os.environ.get('CLEVRER_DIR', 'clevrer'))
QUESTIONS_FILE = CLEVRER_DIR / "questions" / "clevrer_validation.json"
ANNOTATIONS_ZIP = CLEVRER_DIR / "annotations" / "annotation_validation.zip"


@dataclass
class ModelResult:
    model_name: str
    total: int = 0
    correct: int = 0
    by_type: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {'correct': 0, 'total': 0})
    )
    
    def accuracy(self) -> float:
        return self.correct / max(1, self.total) * 100
    
    def add(self, q_type: str, is_correct: bool):
        self.total += 1
        self.correct += int(is_correct)
        self.by_type[q_type]['total'] += 1
        self.by_type[q_type]['correct'] += int(is_correct)


def load_annotation(scene_index: int, zip_file: zipfile.ZipFile) -> Optional[Dict]:
    """Load annotation for a scene from the zip file."""
    # Find the right subfolder (annotations are in 1000-scene chunks)
    chunk_start = (scene_index // 1000) * 1000
    chunk_end = chunk_start + 1000
    folder = f"annotation_{chunk_start}-{chunk_end}"
    filename = f"{folder}/annotation_{scene_index}.json"
    
    try:
        with zip_file.open(filename) as f:
            return json.load(f)
    except KeyError:
        return None


def create_scene_description(annotation: Dict, frame_idx: int = 64) -> str:
    """Convert CLEVRER annotation to text description.
    
    Uses middle frame (64) by default to capture typical scene state.
    """
    objects = annotation.get('object_property', [])
    trajectories = annotation.get('motion_trajectory', [])
    collisions = annotation.get('collision', [])
    
    # Get object states at specified frame
    frame_data = None
    for frame in trajectories:
        if frame.get('frame_id') == frame_idx:
            frame_data = frame
            break
    
    if not frame_data:
        frame_data = trajectories[min(frame_idx, len(trajectories)-1)] if trajectories else {'objects': []}
    
    # Build object descriptions
    obj_descriptions = []
    for obj_prop in objects:
        obj_id = obj_prop['object_id']
        color = obj_prop['color']
        material = obj_prop['material']
        shape = obj_prop['shape']
        
        # Find this object's state in frame data
        obj_state = None
        for obj in frame_data.get('objects', []):
            if obj['object_id'] == obj_id:
                obj_state = obj
                break
        
        if obj_state:
            loc = obj_state['location']
            vel = obj_state['velocity']
            speed = (vel[0]**2 + vel[1]**2 + vel[2]**2) ** 0.5
            
            # Describe motion
            if speed < 0.1:
                motion = "stationary"
            else:
                # Determine direction
                if abs(vel[0]) > abs(vel[1]):
                    direction = "right" if vel[0] > 0 else "left"
                else:
                    direction = "forward" if vel[1] > 0 else "backward"
                motion = f"moving {direction} (speed={speed:.2f})"
            
            obj_descriptions.append(
                f"  - {color} {material} {shape}: position=({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f}), {motion}"
            )
        else:
            obj_descriptions.append(f"  - {color} {material} {shape}: (not visible)")
    
    # NOTE: Collision events removed for fair comparison with Physics-LLM
    # Physics-LLM only receives single-frame state tensor without collision labels
    
    scene_text = f"""SCENE (frame {frame_idx} of 128):
Objects in scene:
{chr(10).join(obj_descriptions)}"""
    
    return scene_text


def create_prompt_with_scene(question: Dict, scene_text: str, no_tools: bool = False) -> str:
    """Create prompt with scene information."""
    q_text = question['question']
    choices = question.get('choices', [])
    
    if choices:
        choice_strs = []
        for i, c in enumerate(choices):
            choice_text = c.get('choice', c) if isinstance(c, dict) else str(c)
            choice_strs.append(f"{chr(65+i)}) {choice_text}")
        choices_text = "\nOptions:\n" + "\n".join(choice_strs)
    else:
        choices_text = ""
    
    no_tools_instruction = ""
    if no_tools:
        no_tools_instruction = "(Use only mental reasoning - no tools, calculators, or code.)\n"
    
    return f"""You are answering physics reasoning questions about a simulation.
{no_tools_instruction}
{scene_text}

Question: {q_text}{choices_text}

IMPORTANT: You MUST respond with EXACTLY ONE character: A, B, C, or D.
Do NOT include any other text, explanation, punctuation, or reasoning.
Just output the single letter answer."""


def check_answer(predicted: str, ground_truth: str, choices: List = None) -> bool:
    """Check if answer is correct."""
    pred = predicted.lower().strip()
    if pred:
        pred = pred.split()[0].split('\n')[0].rstrip('.')
    gt = ground_truth.lower().strip()
    
    # Empty prediction is always wrong
    if not pred:
        return False
    
    if pred == gt:
        return True
    
    # Yes/No normalization
    if pred in ['yes', 'true'] and gt in ['yes', 'true']:
        return True
    if pred in ['no', 'false'] and gt in ['no', 'false']:
        return True
    
    # MCQ letter matching
    if choices and len(pred) == 1 and pred.isalpha():
        idx = ord(pred.upper()) - ord('A')
        if 0 <= idx < len(choices):
            choice = choices[idx]
            if isinstance(choice, dict) and choice.get('answer') == 'correct':
                return True
    
    return False


def call_openai_compatible(client, prompt: str, model: str, provider: str = "openai", max_retries: int = 3) -> str:
    """Call OpenAI-compatible API (works for OpenAI, Together AI, Fireworks) with retry logic."""
    for attempt in range(max_retries):
        try:
            # GPT-5.x models use max_completion_tokens instead of max_tokens
            if "gpt-5" in model:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=50,
                    temperature=0.0,
                )
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.0,
                )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            if "500" in error_str or "Internal server error" in error_str:
                if attempt < max_retries - 1:
                    print(f"  {provider} 500 error, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(2)  # Wait before retry
                    continue
            print(f"  {provider} error: {e}")
            return ""
    return ""


def call_gemini(client, model_id: str, prompt: str, max_retries: int = 3) -> str:
    """Call Gemini API with retry logic (google-genai SDK)."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            return (response.text or "").strip()
        except Exception as e:
            error_str = str(e)
            if "500" in error_str or "Internal" in error_str or "UNAVAILABLE" in error_str:
                if attempt < max_retries - 1:
                    print(f"  Gemini error, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(2)
                    continue
            print(f"  Gemini error: {e}")
            return ""
    return ""


def call_anthropic(client, prompt: str, model: str, max_retries: int = 3, use_thinking: bool = False) -> str:
    """Call Anthropic Claude API with retry logic."""
    for attempt in range(max_retries):
        try:
            if use_thinking:
                # Extended thinking mode for Opus models
                message = client.messages.create(
                    model=model,
                    max_tokens=16000,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 10000
                    },
                    messages=[{"role": "user", "content": prompt + "\n\nAfter thinking, respond with ONLY a single letter: A, B, C, or D."}]
                )
                # Extract text from thinking response (may have thinking blocks)
                response = ""
                for block in message.content:
                    if hasattr(block, 'text'):
                        response = block.text.strip().upper()
            else:
                message = client.messages.create(
                    model=model,
                    max_tokens=5,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "Answer:"}
                    ]
                )
                response = message.content[0].text.strip().upper()
            # Extract just the letter from response
            for char in response:
                if char in 'ABCD':
                    return char
            return response[:1] if response else ""
        except Exception as e:
            error_str = str(e)
            if "500" in error_str or "overloaded" in error_str.lower():
                if attempt < max_retries - 1:
                    print(f"  Anthropic error, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(2)
                    continue
            print(f"  Anthropic error: {e}")
            return ""
    return ""


def get_client_for_provider(provider: str, model_id: str):
    """Get API client for the specified provider."""
    if provider == "openai":
        if not HAS_OPENAI:
            raise RuntimeError("OpenAI not installed: pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAI(api_key=api_key), model_id
    
    elif provider == "gemini":
        if not HAS_GEMINI:
            raise RuntimeError("google-genai not installed: pip install -U google-genai")
        # Prefer GEMINI_API_KEY (Gemini-specific) over GOOGLE_API_KEY (shared
        # with other Google Cloud services); fall back for backward compat.
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Neither GEMINI_API_KEY nor GOOGLE_API_KEY set")
        return genai.Client(api_key=api_key), model_id
    
    elif provider == "together":
        if not HAS_OPENAI:
            raise RuntimeError("OpenAI not installed (needed for Together AI): pip install openai")
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY not set. Get one at https://together.ai")
        return OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1"), model_id
    
    elif provider == "fireworks":
        if not HAS_OPENAI:
            raise RuntimeError("OpenAI not installed (needed for Fireworks): pip install openai")
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY not set. Get one at https://fireworks.ai")
        return OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1"), model_id
    
    elif provider == "anthropic":
        if not HAS_ANTHROPIC:
            raise RuntimeError("Anthropic not installed: pip install anthropic")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return anthropic.Anthropic(api_key=api_key), model_id
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_benchmark(
    model_type: str,
    max_questions: int = 100,
    output_file: str = None,
    no_tools: bool = False,
    question_types: tuple = ("explanatory", "predictive", "counterfactual"),
    heldout_scenes: Optional[set] = None,
    valid_only: bool = False,
) -> ModelResult:
    """Run benchmark with scene information.

    Args:
        question_types: Tuple of CLEVRER question types to evaluate. Defaults
            to all three causal types (matching the 1K-pool primary protocol).
            Pass a single-element tuple like ``("predictive",)`` to produce a
            predictive-only supplementary measurement with a larger n than the
            ~163 predictive items captured in the natural-distribution 1K pool.
        heldout_scenes: If provided, only iterate scenes whose scene_index is
            in this set. Used for the paired held-out evaluation that matches
            the 1,998-item partition Grounded-Physics LM reports on.
        valid_only: If True, skip questions where no choice is labeled
            ``'correct'`` in ground truth. Matches the valid-only filter
            Grounded-Physics LM's compute_paper_stats.py applies, so LLM
            item counts match Ours exactly (n=1,998 on held-out; per-type
            710/361/927 explanatory/predictive/counterfactual).
    """
    
    # Get model config
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_type]
    provider = config["provider"]
    model_id = config["model_id"]
    display_name = config["display_name"]
    
    # Get client for provider
    client, model_id = get_client_for_provider(provider, model_id)
    
    model_name = f"{display_name} (no tools)" if no_tools else display_name
    
    # Load questions
    with open(QUESTIONS_FILE, 'r') as f:
        all_scenes = json.load(f)
    
    # Open annotations zip
    zip_file = zipfile.ZipFile(ANNOTATIONS_ZIP, 'r')
    
    result = ModelResult(model_name=model_name)
    wrong_answers = []
    
    question_count = 0
    skip_count = 0  # For resuming from incremental save
    SAVE_EVERY = 50  # Incremental save interval
    
    # Resume from incremental checkpoint if it exists
    if output_file:
        partial_path = Path(output_file).with_suffix('.partial.json')
        if partial_path.exists():
            try:
                saved = json.loads(partial_path.read_text())
                skip_count = saved.get('questions_done', 0)
                result.correct = saved.get('correct', 0)
                result.total = saved.get('total', 0)
                for k, v in saved.get('by_type', {}).items():
                    result.by_type[k] = {'correct': v['correct'], 'total': v['total']}
                wrong_path_partial = partial_path.with_suffix('.wrong.partial.json')
                if wrong_path_partial.exists():
                    wrong_answers = json.loads(wrong_path_partial.read_text())
                print(f"  >> Resuming from checkpoint: {skip_count} questions done")
            except Exception as e:
                print(f"  >> Failed to load checkpoint, starting fresh: {e}")
                skip_count = 0
    
    def _save_incremental():
        """Save incremental checkpoint."""
        if not output_file:
            return
        partial_path = Path(output_file).with_suffix('.partial.json')
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model': model_name,
            'questions_done': question_count,
            'total': result.total,
            'correct': result.correct,
            'accuracy': result.accuracy(),
            'by_type': {k: {'correct': v['correct'], 'total': v['total'],
                          'accuracy': v['correct']/max(1,v['total'])*100}
                       for k, v in result.by_type.items()}
        }
        with open(partial_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        wrong_path_partial = partial_path.with_suffix('.wrong.partial.json')
        with open(wrong_path_partial, 'w') as f:
            json.dump(wrong_answers, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"CLEVRER Benchmark: {model_name}")
    print(f"{'='*60}")
    
    for scene in all_scenes:
        scene_index = scene.get('scene_index')

        # Held-out scene filter: restrict to the 501-scene partition Grounded-
        # Physics LM reports on (scene_index in [14499, 14999]). When the filter
        # is active this makes the LLM evaluation a true paired comparison with
        # Ours on identical items.
        if heldout_scenes is not None and scene_index not in heldout_scenes:
            continue

        # Load annotation for this scene
        annotation = load_annotation(scene_index, zip_file)
        if not annotation:
            continue
        
        # Create scene description
        scene_text = create_scene_description(annotation)
        
        for q in scene.get('questions', []):
            q_type = q.get('question_type', 'descriptive')
            
            # Restrict to the configured question types. The default tuple
            # (explanatory, predictive, counterfactual) reproduces the 1K-pool
            # protocol; passing a smaller subset produces a per-type
            # supplementary measurement at higher per-type n for the same total
            # max_questions budget.
            if q_type not in question_types:
                continue

            # Valid-only filter: skip questions where ground truth has no
            # 'correct' choice (degenerate items that would count as wrong
            # by construction). Matches compute_paper_stats.py --valid_only.
            if valid_only:
                has_correct_choice = any(
                    isinstance(c, dict) and c.get('answer') == 'correct'
                    for c in q.get('choices', [])
                )
                if not has_correct_choice:
                    continue

            if question_count >= max_questions:
                break
            
            # Skip already-processed questions when resuming
            question_count += 1
            if question_count <= skip_count:
                continue
            
            # Get ground truth
            answer = q.get('answer', '')
            if isinstance(answer, list):
                answer = answer[0] if answer else ''
            ground_truth = str(answer).lower()
            
            # Create prompt with scene
            prompt = create_prompt_with_scene(q, scene_text, no_tools=no_tools)
            
            # Call model
            if provider == "gemini":
                predicted = call_gemini(client, model_id, prompt)
                time.sleep(0.5)  # Rate limit
            elif provider == "anthropic":
                use_thinking = config.get("thinking", False)
                predicted = call_anthropic(client, prompt, model_id, use_thinking=use_thinking)
                time.sleep(0.5 if use_thinking else 0.3)  # Rate limit
            else:
                predicted = call_openai_compatible(client, prompt, model_id, provider)
                if provider in ["together", "fireworks"]:
                    time.sleep(0.3)  # Rate limit for API providers
            
            # Check answer
            is_correct = check_answer(predicted, ground_truth, q.get('choices'))
            result.add(q_type, is_correct)
            
            if not is_correct:
                wrong_answers.append({
                    'scene_index': scene_index,
                    'question': q.get('question'),
                    'question_type': q_type,
                    'expected': ground_truth,
                    'predicted': predicted,
                    'scene_text': scene_text[:500]  # Truncate for readability
                })
            
            status = "OK" if is_correct else "X"
            print(f"[{question_count}/{max_questions}] {q_type}: {status} (pred={predicted[:20]})")
            
            # Incremental save
            if question_count % SAVE_EVERY == 0:
                _save_incremental()
        
        if question_count >= max_questions:
            break
    
    zip_file.close()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"Overall: {result.correct}/{result.total} ({result.accuracy():.1f}%)")
    print(f"\nBy question type:")
    for qtype, stats in result.by_type.items():
        acc = stats['correct'] / max(1, stats['total']) * 100
        print(f"  {qtype}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = {
            'model': model_name,
            'total': result.total,
            'correct': result.correct,
            'accuracy': result.accuracy(),
            'by_type': {k: {'correct': v['correct'], 'total': v['total'], 
                          'accuracy': v['correct']/max(1,v['total'])*100} 
                       for k, v in result.by_type.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save wrong answers
        wrong_path = output_path.with_suffix('.wrong.json')
        with open(wrong_path, 'w') as f:
            json.dump(wrong_answers, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Wrong answers saved to: {wrong_path}")
        
        # Clean up partial checkpoint files on successful completion
        partial_path = output_path.with_suffix('.partial.json')
        if partial_path.exists():
            partial_path.unlink()
        wrong_partial = partial_path.with_suffix('.wrong.partial.json')
        if wrong_partial.exists():
            wrong_partial.unlink()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="CLEVRER benchmark with scene information")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--max_questions", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_tools", action="store_true", help="Add explicit no-tools instruction")
    parser.add_argument(
        "--question_types", type=str, default="explanatory,predictive,counterfactual",
        help=("CSV of CLEVRER question types to evaluate. Default reproduces "
              "the 1K-pool protocol (all 3 causal types). Pass 'predictive' "
              "alone for a predictive-only supplementary run with larger n."))
    parser.add_argument(
        "--heldout_scenes_json", type=str, default=None,
        help=("Path to a JSON file containing a list of scene_index ints to "
              "restrict evaluation to. Produced by extracting the 501 held-out "
              "scenes from the CLEVRER training H5; pairs with Ours' primary "
              "evaluation partition so the LLM and Ours rows share items."))
    parser.add_argument(
        "--valid_only", action="store_true",
        help=("Exclude CLEVRER questions where no choice is labeled 'correct' "
              "(degenerate items). Matches compute_paper_stats.py --valid_only; "
              "with --heldout_scenes_json yields the same n=1,998 partition "
              "as Grounded-Physics LM's primary headline."))
    args = parser.parse_args()

    qtypes = tuple(t.strip() for t in args.question_types.split(",") if t.strip())
    valid = {"explanatory", "predictive", "counterfactual"}
    bad = [t for t in qtypes if t not in valid]
    if bad:
        parser.error(f"--question_types contains unknown values {bad}; valid: {sorted(valid)}")

    heldout_scenes = None
    if args.heldout_scenes_json:
        with open(args.heldout_scenes_json, 'r', encoding='utf-8') as f:
            heldout_scenes = set(int(s) for s in json.load(f))
        print(f"Held-out scene filter: {len(heldout_scenes)} scenes loaded from "
              f"{args.heldout_scenes_json}")

    if args.output is None:
        suffix = "_notools" if args.no_tools else ""
        # Distinguish held-out output files from the legacy 1K-pool files so
        # downstream loaders (_compute_1k_cis.py) can switch cleanly.
        partition_tag = "_heldout" if heldout_scenes else ""
        args.output = (f"clevrer_benchmark/results/"
                       f"{args.model}_with_scene{partition_tag}{suffix}.json")

    run_benchmark(args.model, args.max_questions, args.output,
                  no_tools=args.no_tools, question_types=qtypes,
                  heldout_scenes=heldout_scenes,
                  valid_only=args.valid_only)


if __name__ == "__main__":
    main()
