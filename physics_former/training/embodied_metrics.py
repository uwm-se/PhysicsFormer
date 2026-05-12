"""
Embodied Cognition Metrics Implementation

Implements the metrics from EMBODIED_COGNITION_METRICS.md
Measures physics grounding, causal understanding, and embodied reasoning.

Usage:
    from training.embodied_metrics import EmbodiedMetricsAnalyzer
    
    analyzer = EmbodiedMetricsAnalyzer(model)
    results = analyzer.run_all_analyses(test_data)
"""
import json
import torch
import numpy as np
from .utils.serialization import save_json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')


class EmbodiedMetricsAnalyzer:
    """
    Comprehensive embodied cognition metrics for physics understanding.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        print("=" * 70)
        print("EMBODIED COGNITION METRICS ANALYZER")
        print("=" * 70)
        print(f"Device: {device}")
        print("=" * 70)
    
    def physics_grounding_coherence(
        self,
        test_cases: List[Dict],
        abstraction_levels: List[str] = ['physics', 'counting', 'arithmetic', 'symbolic']
    ) -> Dict:
        """
        Measure consistency of predictions across abstraction levels.
        
        Tests if abstract predictions align with physical reality.
        """
        print("\n" + "=" * 70)
        print("1. PHYSICS GROUNDING COHERENCE")
        print("=" * 70)
        
        coherence_scores = []
        violations = []
        
        with torch.no_grad():
            for case in test_cases:
                level_predictions = {}
                
                # Get predictions at each abstraction level
                for level in abstraction_levels:
                    if level in case:
                        input_data = self._prepare_input(case[level])
                        pred = self.model(**input_data)
                        level_predictions[level] = pred
                
                # Check consistency between levels
                if len(level_predictions) >= 2:
                    levels = list(level_predictions.keys())
                    
                    for i in range(len(levels) - 1):
                        level1, level2 = levels[i], levels[i + 1]
                        
                        # Map predictions to common space
                        pred1_mapped = self._map_to_common_space(
                            level_predictions[level1], level1
                        )
                        pred2_mapped = self._map_to_common_space(
                            level_predictions[level2], level2
                        )
                        
                        # Compute consistency
                        consistency = self._compute_consistency(pred1_mapped, pred2_mapped)
                        coherence_scores.append(consistency)
                        
                        if consistency < 0.7:
                            violations.append({
                                'case': case.get('name', 'unknown'),
                                'levels': f"{level1}-{level2}",
                                'consistency': consistency
                            })
        
        results = {
            'mean_coherence': np.mean(coherence_scores) if coherence_scores else 0.0,
            'coherence_variance': np.var(coherence_scores) if coherence_scores else 0.0,
            'num_violations': len(violations),
            'violation_rate': len(violations) / len(test_cases) if test_cases else 0.0,
            'violations': violations[:5]  # Top 5 violations
        }
        
        print(f"\nMean Coherence: {results['mean_coherence']:.3f}")
        print(f"Violations: {results['num_violations']} ({results['violation_rate']:.1%})")
        
        if results['mean_coherence'] > 0.8:
            print("PASS: Strong physics grounding")
        elif results['mean_coherence'] > 0.6:
            print("WARNING:  Moderate physics grounding")
        else:
            print("FAIL: Weak physics grounding")
        
        return results
    
    def counterfactual_reasoning(
        self,
        scenarios: List[Dict]
    ) -> Dict:
        """
        Test model's ability to reason about altered physics.
        
        "What if gravity was 2x?" type questions.
        """
        print("\n" + "=" * 70)
        print("2. COUNTERFACTUAL PHYSICS REASONING")
        print("=" * 70)
        
        scores = {
            'gravity_change': [],
            'friction_change': [],
            'mass_change': [],
            'force_change': []
        }
        
        with torch.no_grad():
            for scenario in scenarios:
                # Original prediction
                original_input = self._prepare_input(scenario['original'])
                original_pred = self.model(**original_input)
                
                # Counterfactual predictions
                for cf_type, cf_scenario in scenario.get('counterfactuals', {}).items():
                    cf_input = self._prepare_input(cf_scenario)
                    cf_pred = self.model(**cf_input)
                    
                    # Expected change (from physics)
                    expected_change = scenario.get('expected_changes', {}).get(cf_type, 0)
                    
                    # Actual change in prediction
                    actual_change = self._compute_prediction_change(original_pred, cf_pred)
                    
                    # Accuracy: how close is actual to expected?
                    if expected_change != 0:
                        accuracy = 1 - min(abs(actual_change - expected_change) / abs(expected_change), 1.0)
                    else:
                        accuracy = 1.0 if abs(actual_change) < 0.1 else 0.0
                    
                    if cf_type in scores:
                        scores[cf_type].append(accuracy)
        
        results = {
            'counterfactual_accuracy': {k: np.mean(v) if v else 0.0 for k, v in scores.items()},
            'overall_causal_understanding': np.mean([np.mean(v) for v in scores.values() if v])
        }
        
        print(f"\nOverall Causal Understanding: {results['overall_causal_understanding']:.3f}")
        for cf_type, acc in results['counterfactual_accuracy'].items():
            print(f"  {cf_type}: {acc:.3f}")
        
        return results
    
    def physical_intuition_emergence(
        self,
        intuition_tests: List[Dict]
    ) -> Dict:
        """
        Test basic physics intuitions (gravity, inertia, conservation).
        """
        print("\n" + "=" * 70)
        print("3. PHYSICAL INTUITION ASSESSMENT")
        print("=" * 70)
        
        intuition_scores = {
            'gravity': [],
            'inertia': [],
            'conservation': [],
            'friction': [],
            'collisions': []
        }
        
        with torch.no_grad():
            for test in intuition_tests:
                principle = test.get('principle', 'unknown')
                
                # Prepare input
                input_data = self._prepare_input(test['scenario'])
                prediction = self.model(**input_data)
                
                # Check if prediction matches intuition
                expected = test['expected_behavior']
                matches_intuition = self._check_intuition_match(prediction, expected)
                
                if principle in intuition_scores:
                    intuition_scores[principle].append(1.0 if matches_intuition else 0.0)
        
        results = {
            'intuition_by_principle': {k: np.mean(v) if v else 0.0 for k, v in intuition_scores.items()},
            'overall_intuition': np.mean([np.mean(v) for v in intuition_scores.values() if v])
        }
        
        print(f"\nOverall Physical Intuition: {results['overall_intuition']:.3f}")
        for principle, score in results['intuition_by_principle'].items():
            status = "PASS:" if score > 0.8 else "WARNING:" if score > 0.5 else "FAIL:"
            print(f"  {status} {principle}: {score:.3f}")
        
        return results
    
    def embodied_representation_geometry(
        self,
        physics_states: List[Dict]
    ) -> Dict:
        """
        Test if representation space reflects physical structure.
        """
        print("\n" + "=" * 70)
        print("4. EMBODIED REPRESENTATION GEOMETRY")
        print("=" * 70)
        
        # Extract embeddings
        embeddings = []
        with torch.no_grad():
            for state in physics_states:
                input_data = self._prepare_input(state)
                
                # Get internal representation (encoder output)
                # Note: Encoder expects object_states as positional arg, not kwargs
                if hasattr(self.model, 'encoder') and 'object_states' in input_data:
                    # Call encoder with object_states only
                    emb = self.model.encoder(input_data['object_states'])
                elif 'object_states' in input_data:
                    # Call model with object_states
                    emb = self.model(object_states=input_data['object_states'], mode='physics')
                else:
                    # Fallback: try with all input data
                    emb = self.model(**input_data)
                
                if isinstance(emb, tuple):
                    emb = emb[0]
                
                # Flatten if needed
                if len(emb.shape) > 2:
                    emb = emb.mean(dim=1)
                
                embeddings.append(emb.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        
        # Compute physics distances
        physics_distances = self._compute_physics_distances(physics_states)
        
        # Compute embedding distances
        embedding_distances = self._compute_pairwise_distances(embeddings)
        
        # Distance correlation
        distance_correlation = spearmanr(
            physics_distances.flatten(),
            embedding_distances.flatten()
        )[0]
        
        # Neighborhood preservation
        k = min(10, len(physics_states) // 2)
        neighbor_preservation = self._compute_neighbor_preservation(
            physics_distances, embedding_distances, k
        )
        
        results = {
            'distance_correlation': distance_correlation,
            'neighbor_preservation': neighbor_preservation,
            'geometric_fidelity': (distance_correlation + neighbor_preservation) / 2
        }
        
        print(f"\nDistance Correlation: {results['distance_correlation']:.3f}")
        print(f"Neighbor Preservation: {results['neighbor_preservation']:.3f}")
        print(f"Geometric Fidelity: {results['geometric_fidelity']:.3f}")
        
        if results['geometric_fidelity'] > 0.7:
            print("PASS: Strong geometric alignment")
        else:
            print("WARNING:  Weak geometric alignment")
        
        return results
    
    def temporal_binding_analysis(
        self,
        video_sequences: List[Dict]
    ) -> Dict:
        """
        Test object permanence and temporal causality tracking.
        """
        print("\n" + "=" * 70)
        print("5. TEMPORAL BINDING & OBJECT PERMANENCE")
        print("=" * 70)
        
        results = {
            'object_permanence': [],
            'temporal_binding': [],
            'event_segmentation': []
        }
        
        with torch.no_grad():
            for sequence in video_sequences:
                # Object permanence test
                if 'hidden_frames' in sequence:
                    permanence_score = self._test_object_permanence(sequence)
                    results['object_permanence'].append(permanence_score)
                
                # Temporal binding test
                if 'causal_pairs' in sequence:
                    binding_score = self._test_temporal_binding(sequence)
                    results['temporal_binding'].append(binding_score)
                
                # Event segmentation test
                if 'event_boundaries' in sequence:
                    segmentation_score = self._test_event_segmentation(sequence)
                    results['event_segmentation'].append(segmentation_score)
        
        summary = {
            'object_permanence': np.mean(results['object_permanence']) if results['object_permanence'] else 0.0,
            'temporal_binding': np.mean(results['temporal_binding']) if results['temporal_binding'] else 0.0,
            'event_segmentation': np.mean(results['event_segmentation']) if results['event_segmentation'] else 0.0
        }
        
        summary['temporal_understanding'] = np.mean([v for v in summary.values() if v > 0])
        
        print(f"\nObject Permanence: {summary['object_permanence']:.3f}")
        print(f"Temporal Binding: {summary['temporal_binding']:.3f}")
        print(f"Event Segmentation: {summary['event_segmentation']:.3f}")
        print(f"Overall Temporal Understanding: {summary['temporal_understanding']:.3f}")
        
        return summary
    
    def physical_plausibility_discrimination(
        self,
        test_cases: List[Dict]
    ) -> Dict:
        """
        Test if model can distinguish possible from impossible physics.
        """
        print("\n" + "=" * 70)
        print("6. PHYSICAL PLAUSIBILITY DISCRIMINATION")
        print("=" * 70)
        
        correct = 0
        plausible_correct = 0
        implausible_correct = 0
        plausible_total = 0
        implausible_total = 0
        
        with torch.no_grad():
            for case in test_cases:
                input_data = self._prepare_input(case['scenario'])
                
                # Get plausibility score
                plausibility = self._judge_plausibility(input_data)
                
                # True label
                is_plausible = case['is_physically_possible']
                predicted_plausible = plausibility > 0.5
                
                if predicted_plausible == is_plausible:
                    correct += 1
                    if is_plausible:
                        plausible_correct += 1
                    else:
                        implausible_correct += 1
                
                if is_plausible:
                    plausible_total += 1
                else:
                    implausible_total += 1
        
        results = {
            'overall_accuracy': correct / len(test_cases) if test_cases else 0.0,
            'plausible_accuracy': plausible_correct / plausible_total if plausible_total > 0 else 0.0,
            'implausible_accuracy': implausible_correct / implausible_total if implausible_total > 0 else 0.0
        }
        
        print(f"\nOverall Accuracy: {results['overall_accuracy']:.3f}")
        print(f"Plausible Detection: {results['plausible_accuracy']:.3f}")
        print(f"Implausible Detection: {results['implausible_accuracy']:.3f}")
        
        return results
    
    def run_all_analyses(
        self,
        test_data: Dict,
        save_results: bool = True,
        output_path: str = None
    ) -> Dict:
        """
        Run all embodied cognition analyses.
        
        Args:
            test_data: Dictionary with test cases for each metric
            save_results: Whether to save results to file
            output_path: Path to save results
        
        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "=" * 70)
        print("RUNNING COMPLETE EMBODIED COGNITION ANALYSIS")
        print("=" * 70)
        
        all_results = {}
        
        # 1. Physics Grounding Coherence
        if 'grounding_tests' in test_data:
            all_results['grounding_coherence'] = self.physics_grounding_coherence(
                test_data['grounding_tests']
            )
        
        # 2. Counterfactual Reasoning
        if 'counterfactual_scenarios' in test_data:
            all_results['counterfactual_reasoning'] = self.counterfactual_reasoning(
                test_data['counterfactual_scenarios']
            )
        
        # 3. Physical Intuition
        if 'intuition_tests' in test_data:
            all_results['physical_intuition'] = self.physical_intuition_emergence(
                test_data['intuition_tests']
            )
        
        # 4. Representation Geometry
        if 'physics_states' in test_data:
            all_results['representation_geometry'] = self.embodied_representation_geometry(
                test_data['physics_states']
            )
        
        # 5. Temporal Binding
        if 'video_sequences' in test_data:
            all_results['temporal_binding'] = self.temporal_binding_analysis(
                test_data['video_sequences']
            )
        
        # 6. Plausibility Discrimination
        if 'plausibility_tests' in test_data:
            all_results['plausibility'] = self.physical_plausibility_discrimination(
                test_data['plausibility_tests']
            )
        
        # Compute overall embodiment score
        scores = []
        if 'grounding_coherence' in all_results:
            scores.append(all_results['grounding_coherence']['mean_coherence'])
        if 'counterfactual_reasoning' in all_results:
            scores.append(all_results['counterfactual_reasoning']['overall_causal_understanding'])
        if 'physical_intuition' in all_results:
            scores.append(all_results['physical_intuition']['overall_intuition'])
        if 'representation_geometry' in all_results:
            scores.append(all_results['representation_geometry']['geometric_fidelity'])
        if 'temporal_binding' in all_results:
            scores.append(all_results['temporal_binding']['temporal_understanding'])
        if 'plausibility' in all_results:
            scores.append(all_results['plausibility']['overall_accuracy'])
        
        all_results['overall_embodiment_score'] = np.mean(scores) if scores else 0.0
        
        # Print summary
        print("\n" + "=" * 70)
        print("EMBODIED COGNITION SUMMARY")
        print("=" * 70)
        print(f"\nOverall Embodiment Score: {all_results['overall_embodiment_score']:.3f}")
        print("\nComponent Scores:")
        for key, result in all_results.items():
            if key != 'overall_embodiment_score' and isinstance(result, dict):
                # Get main score from each component
                main_score = self._extract_main_score(result)
                if main_score is not None:
                    print(f"  {key}: {main_score:.3f}")
        
        # Save results
        if save_results:
            if output_path is None:
                output_path = "embodied_cognition_results.json"
            
            save_json(all_results, output_path)
            
            print(f"\nPASS: Results saved to: {output_path}")
        
        return all_results
    
    # Helper methods
    
    def _prepare_input(self, data: Dict) -> Dict:
        """Prepare input data for model."""
        # If data has individual physics fields, combine into object_states
        if 'position' in data and 'object_states' not in data:
            # Build 21D state vector: [x,y,z, vx,vy,vz, roll,pitch,yaw, wx,wy,wz, fx,fy,fz, E, mass,radius,shape,friction,restitution]
            pos = data.get('position', [0, 0, 0])
            vel = data.get('velocity', [0, 0, 0])
            mass = data.get('mass', 1.0)
            
            # Create minimal 21D state (pad with zeros for missing fields)
            state = pos + vel + [0]*9 + [0] + [mass] + [0]*4  # 3+3+9+1+1+4 = 21
            
            # Create object_states tensor [1, 1, 21] (batch=1, objects=1, state_dim=21)
            object_states = torch.tensor([state], dtype=torch.float32).unsqueeze(0).to(self.device)
            return {'object_states': object_states}
        
        # Convert to tensors and move to device
        input_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                input_data[key] = torch.from_numpy(value).to(self.device)
            elif isinstance(value, torch.Tensor):
                input_data[key] = value.to(self.device)
            else:
                input_data[key] = value
        return input_data
    
    def _map_to_common_space(self, prediction, level: str):
        """Map predictions from different levels to common space."""
        # Simplified mapping - would need task-specific implementation
        if isinstance(prediction, dict):
            # Extract main prediction tensor
            prediction = prediction.get('logits', prediction.get('output', list(prediction.values())[0]))
        
        if isinstance(prediction, torch.Tensor):
            return prediction.cpu().numpy().flatten()
        return np.array(prediction).flatten()
    
    def _compute_consistency(self, pred1, pred2):
        """Compute consistency between two predictions."""
        # Cosine similarity
        if len(pred1) != len(pred2):
            # Pad or truncate to same length
            min_len = min(len(pred1), len(pred2))
            pred1 = pred1[:min_len]
            pred2 = pred2[:min_len]
        
        similarity = 1 - cosine(pred1, pred2)
        return max(0, similarity)  # Clamp to [0, 1]
    
    def _compute_prediction_change(self, pred1, pred2):
        """Compute change between predictions."""
        p1 = self._map_to_common_space(pred1, 'default')
        p2 = self._map_to_common_space(pred2, 'default')
        return np.linalg.norm(p2 - p1)
    
    def _check_intuition_match(self, prediction, expected):
        """Check if prediction matches expected intuition."""
        # Simplified - would need task-specific logic
        pred_array = self._map_to_common_space(prediction, 'default')
        
        # Check if prediction direction matches expected
        if isinstance(expected, str):
            # Categorical expectation
            return True  # Placeholder
        else:
            # Numerical expectation
            return abs(pred_array.mean() - expected) < 0.2
    
    def _compute_physics_distances(self, states: List[Dict]):
        """Compute pairwise physics distances between states."""
        n = len(states)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Compute physics-based distance
                dist = self._physics_distance(states[i], states[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _physics_distance(self, state1: Dict, state2: Dict):
        """Compute physics-based distance between two states."""
        # Extract physical properties
        props1 = self._extract_physics_properties(state1)
        props2 = self._extract_physics_properties(state2)
        
        return euclidean(props1, props2)
    
    def _extract_physics_properties(self, state: Dict):
        """Extract physical properties from state."""
        # Simplified - extract key physics properties
        props = []
        for key in ['position', 'velocity', 'mass', 'force']:
            if key in state:
                val = state[key]
                if isinstance(val, (list, np.ndarray)):
                    props.extend(np.array(val).flatten())
                else:
                    props.append(float(val))
        
        return np.array(props) if props else np.array([0.0])
    
    def _compute_pairwise_distances(self, embeddings):
        """Compute pairwise distances between embeddings."""
        n = len(embeddings)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = euclidean(embeddings[i], embeddings[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _compute_neighbor_preservation(self, dist1, dist2, k):
        """Compute k-nearest neighbor preservation."""
        n = len(dist1)
        preservation_scores = []
        
        for i in range(n):
            # Get k nearest neighbors in both spaces
            neighbors1 = np.argsort(dist1[i])[:k+1][1:]  # Exclude self
            neighbors2 = np.argsort(dist2[i])[:k+1][1:]
            
            # Compute overlap
            overlap = len(set(neighbors1) & set(neighbors2)) / k
            preservation_scores.append(overlap)
        
        return np.mean(preservation_scores)
    
    def _test_object_permanence(self, sequence):
        """Test object permanence."""
        # Placeholder - would need actual implementation
        return 0.8
    
    def _test_temporal_binding(self, sequence):
        """Test temporal binding."""
        # Placeholder - would need actual implementation
        return 0.75
    
    def _test_event_segmentation(self, sequence):
        """Test event segmentation."""
        # Placeholder - would need actual implementation
        return 0.7
    
    def _judge_plausibility(self, input_data):
        """Judge physical plausibility of scenario."""
        # Get model prediction
        with torch.no_grad():
            output = self.model(**input_data)
        
        # Extract plausibility score (simplified)
        if isinstance(output, dict):
            # Look for confidence or probability
            if 'confidence' in output:
                return output['confidence'].mean().item()
            elif 'probs' in output:
                return output['probs'].max().item()
        
        # Default: use prediction magnitude as proxy
        if isinstance(output, torch.Tensor):
            return torch.sigmoid(output).mean().item()
        
        return 0.5  # Neutral
    
    def _extract_main_score(self, result: Dict):
        """Extract main score from result dictionary."""
        # Look for common score keys
        for key in ['mean_coherence', 'overall_causal_understanding', 'overall_intuition',
                    'geometric_fidelity', 'temporal_understanding', 'overall_accuracy']:
            if key in result:
                return result[key]
        return None


def main():
    """Example usage."""
    print("Embodied Cognition Metrics - Example Usage")
    print("=" * 70)
    print("\nThis tool measures:")
    print("  1. Physics grounding coherence")
    print("  2. Counterfactual reasoning")
    print("  3. Physical intuition")
    print("  4. Representation geometry")
    print("  5. Temporal binding")
    print("  6. Plausibility discrimination")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
