"""
CLEVRER Benchmark Evaluator for Physics-LLM

Runs Physics-LLM on CLEVRER questions and computes accuracy metrics
for Explanatory, Predictive, and Counterfactual question types.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .scene_converter import clevrer_scene_to_state_tensor, load_clevrer_scene
from .question_mapper import (
    CLEVRERQuestionMapper,
    CLEVRERQuestionType,
    PhysicsLLMQuestionType
)


@dataclass
class EvaluationResult:
    """Results for a single question evaluation."""
    scene_id: str
    question_id: str
    question_text: str
    clevrer_type: str
    physics_llm_type: str
    ground_truth: Any
    predicted: str
    correct: bool
    score: float


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    total_questions: int = 0
    correct_count: int = 0
    total_score: float = 0.0
    
    by_clevrer_type: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: {'correct': 0, 'total': 0, 'score': 0.0}))
    by_physics_type: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: {'correct': 0, 'total': 0, 'score': 0.0}))
    
    individual_results: List[EvaluationResult] = field(default_factory=list)
    
    def add_result(self, result: EvaluationResult):
        """Add a single evaluation result."""
        self.total_questions += 1
        self.correct_count += int(result.correct)
        self.total_score += result.score
        
        self.by_clevrer_type[result.clevrer_type]['total'] += 1
        self.by_clevrer_type[result.clevrer_type]['correct'] += int(result.correct)
        self.by_clevrer_type[result.clevrer_type]['score'] += result.score
        
        self.by_physics_type[result.physics_llm_type]['total'] += 1
        self.by_physics_type[result.physics_llm_type]['correct'] += int(result.correct)
        self.by_physics_type[result.physics_llm_type]['score'] += result.score
        
        self.individual_results.append(result)
    
    def get_accuracy(self) -> float:
        """Get overall accuracy."""
        if self.total_questions == 0:
            return 0.0
        return self.correct_count / self.total_questions
    
    def get_accuracy_by_type(self, type_name: str, by_clevrer: bool = True) -> float:
        """Get accuracy for a specific question type."""
        type_dict = self.by_clevrer_type if by_clevrer else self.by_physics_type
        if type_name not in type_dict or type_dict[type_name]['total'] == 0:
            return 0.0
        return type_dict[type_name]['correct'] / type_dict[type_name]['total']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            'overall': {
                'total_questions': self.total_questions,
                'correct': self.correct_count,
                'accuracy': self.get_accuracy(),
                'mean_score': self.total_score / max(1, self.total_questions)
            },
            'by_clevrer_type': {
                k: {
                    'total': v['total'],
                    'correct': v['correct'],
                    'accuracy': v['correct'] / max(1, v['total']),
                    'mean_score': v['score'] / max(1, v['total'])
                }
                for k, v in self.by_clevrer_type.items()
            },
            'by_physics_llm_type': {
                k: {
                    'total': v['total'],
                    'correct': v['correct'],
                    'accuracy': v['correct'] / max(1, v['total']),
                    'mean_score': v['score'] / max(1, v['total'])
                }
                for k, v in self.by_physics_type.items()
            }
        }


class CLEVREREvaluator:
    """Evaluates Physics-LLM on CLEVRER benchmark."""
    
    def __init__(
        self,
        physics_llm_model: Any,
        question_mapper: Optional[CLEVRERQuestionMapper] = None,
        device: str = 'cuda'
    ):
        """
        Initialize evaluator.
        
        Args:
            physics_llm_model: The Physics-LLM model to evaluate
            question_mapper: Question mapper instance (creates default if None)
            device: Device to run model on
        """
        self.model = physics_llm_model
        self.mapper = question_mapper or CLEVRERQuestionMapper()
        self.device = device
        self.results = BenchmarkResults()
    
    def load_scene_data(self, scene_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load and convert a CLEVRER scene."""
        scene = load_clevrer_scene(str(scene_path))
        return clevrer_scene_to_state_tensor(scene)
    
    def run_physics_llm(
        self,
        states: np.ndarray,
        masks: np.ndarray,
        question_data: Dict[str, Any]
    ) -> str:
        """
        Run Physics-LLM on a question.
        
        Args:
            states: [T, N, 35] state tensor
            masks: [T, N] validity masks
            question_data: Processed question data
            
        Returns:
            Model's predicted answer as string
        """
        import torch
        
        states_tensor = torch.from_numpy(states).float().unsqueeze(0).to(self.device)
        masks_tensor = torch.from_numpy(masks).float().unsqueeze(0).to(self.device)
        
        question_type = question_data.get('physics_llm_type', 'trajectory_prediction')
        target_objects = question_data.get('target_objects', [])
        
        choices = question_data.get('choices')
        if isinstance(choices, list) and choices:
            choice_texts = [c.get('choice', c) if isinstance(c, dict) else str(c) for c in choices]
        else:
            choice_texts = None
        
        with torch.no_grad():
            output = self.model.answer_question(
                states=states_tensor,
                masks=masks_tensor,
                question_type=question_type,
                target_objects=target_objects,
                question_text=question_data.get('question_text', ''),
                choices=choice_texts
            )
        
        return str(output)
    
    def evaluate_scene(
        self,
        scene_path: Path,
        questions: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """Evaluate all questions for a single scene."""
        states, masks, metadata = self.load_scene_data(scene_path)
        scene_id = scene_path.stem
        
        filtered_questions = self.mapper.filter_questions(questions)
        results = []
        
        for idx, q in enumerate(filtered_questions):
            processed = self.mapper.process_question(q, metadata)
            
            try:
                predicted = self.run_physics_llm(states, masks, processed)
            except Exception as e:
                print(f"Error running model on {scene_id} Q{idx}: {e}")
                predicted = "error"
            
            eval_result = self.mapper.evaluate_answer(predicted, processed)
            
            result = EvaluationResult(
                scene_id=scene_id,
                question_id=f"{scene_id}_q{idx}",
                question_text=processed.get('question_text', ''),
                clevrer_type=processed.get('clevrer_type', 'unknown'),
                physics_llm_type=processed.get('question_type', 'unknown'),
                ground_truth=processed.get('ground_truth'),
                predicted=predicted,
                correct=eval_result['correct'],
                score=eval_result['score']
            )
            
            results.append(result)
            self.results.add_result(result)
        
        return results
    
    def evaluate_dataset(
        self,
        scene_dir: Path,
        questions_file: Path,
        max_scenes: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> BenchmarkResults:
        """
        Evaluate on full CLEVRER dataset.
        
        Args:
            scene_dir: Directory containing scene JSON files
            questions_file: JSON file containing questions for all scenes
            max_scenes: Maximum number of scenes to evaluate (None for all)
            progress_callback: Optional callback for progress updates
            
        Returns:
            BenchmarkResults with all evaluation metrics
        """
        with open(questions_file, 'r') as f:
            all_questions = json.load(f)
        
        questions_by_scene = defaultdict(list)
        for q in all_questions:
            scene_id = q.get('scene_index', q.get('video_index', 0))
            questions_by_scene[scene_id].append(q)
        
        scene_files = sorted(scene_dir.glob('*.json'))
        if max_scenes:
            scene_files = scene_files[:max_scenes]
        
        for idx, scene_file in enumerate(scene_files):
            scene_id = int(scene_file.stem.split('_')[-1]) if '_' in scene_file.stem else idx
            questions = questions_by_scene.get(scene_id, [])
            
            if questions:
                self.evaluate_scene(scene_file, questions)
            
            if progress_callback:
                progress_callback(idx + 1, len(scene_files))
        
        return self.results
    
    def save_results(self, output_path: Path):
        """Save evaluation results to JSON file."""
        results_dict = self.results.to_dict()
        
        results_dict['individual_results'] = [
            {
                'scene_id': r.scene_id,
                'question_id': r.question_id,
                'question_text': r.question_text,
                'clevrer_type': r.clevrer_type,
                'physics_llm_type': r.physics_llm_type,
                'ground_truth': r.ground_truth,
                'predicted': r.predicted,
                'correct': r.correct,
                'score': r.score
            }
            for r in self.results.individual_results
        ]
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def print_summary(self):
        """Print evaluation summary to console."""
        results = self.results.to_dict()
        
        print("\n" + "=" * 60)
        print("CLEVRER BENCHMARK RESULTS - Physics-LLM")
        print("=" * 60)
        
        overall = results['overall']
        print(f"\nOverall Accuracy: {overall['accuracy']:.1%} ({overall['correct']}/{overall['total_questions']})")
        print(f"Mean Score: {overall['mean_score']:.3f}")
        
        print("\n--- By CLEVRER Question Type ---")
        for q_type, metrics in results['by_clevrer_type'].items():
            print(f"  {q_type}: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
        
        print("\n--- By Physics-LLM Question Type ---")
        for q_type, metrics in results['by_physics_llm_type'].items():
            print(f"  {q_type}: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
        
        print("=" * 60)


class BaselineEvaluator:
    """Evaluator for baseline models (GPT-4, etc.) on CLEVRER."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.results = BenchmarkResults()
    
    def generate_text_description(self, metadata: Dict[str, Any]) -> str:
        """Generate text description of scene for text-only baselines."""
        objects = metadata.get('objects', [])
        
        desc_parts = [f"Scene contains {len(objects)} objects:"]
        for idx, obj in enumerate(objects):
            desc_parts.append(
                f"  Object {idx + 1}: {obj.get('color', 'unknown')} "
                f"{obj.get('material', '')} {obj.get('shape', 'object')}"
            )
        
        collisions = metadata.get('collisions', [])
        if collisions:
            desc_parts.append(f"\n{len(collisions)} collision events occur in the scene.")
        
        return "\n".join(desc_parts)
    
    def query_baseline(self, scene_description: str, question: str) -> str:
        """Query baseline model. Override for specific implementations."""
        raise NotImplementedError("Subclass must implement query_baseline")
    
    def evaluate_scene(
        self,
        metadata: Dict[str, Any],
        questions: List[Dict[str, Any]],
        scene_id: str
    ) -> List[EvaluationResult]:
        """Evaluate baseline on a single scene."""
        mapper = CLEVRERQuestionMapper()
        scene_description = self.generate_text_description(metadata)
        
        filtered_questions = mapper.filter_questions(questions)
        results = []
        
        for idx, q in enumerate(filtered_questions):
            processed = mapper.process_question(q, metadata)
            
            try:
                predicted = self.query_baseline(scene_description, processed['question_text'])
            except Exception as e:
                print(f"Error querying baseline on {scene_id} Q{idx}: {e}")
                predicted = "error"
            
            eval_result = mapper.evaluate_answer(predicted, processed)
            
            result = EvaluationResult(
                scene_id=scene_id,
                question_id=f"{scene_id}_q{idx}",
                question_text=processed.get('question_text', ''),
                clevrer_type=processed.get('clevrer_type', 'unknown'),
                physics_llm_type=processed.get('question_type', 'unknown'),
                ground_truth=processed.get('ground_truth'),
                predicted=predicted,
                correct=eval_result['correct'],
                score=eval_result['score']
            )
            
            results.append(result)
            self.results.add_result(result)
        
        return results
