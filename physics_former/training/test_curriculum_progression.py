"""
Test script for schema curriculum progression.

Tests:
1. ProgressiveCurriculum initialization
2. Schema progression logic
3. Sequence length progression logic
4. Auto-stop when complete
5. Dataset update_seq_length with Subset wrapper
6. Full integration flow
"""

import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.progressive_curriculum import ProgressiveCurriculum
from training.datasets.cached_physics_dataset import CachedPhysicsDataset
from training.configs.config import TrainingConfig


def test_progressive_curriculum_init():
    """Test 1: Curriculum initialization"""
    print("\n" + "="*70)
    print("TEST 1: Progressive Curriculum Initialization")
    print("="*70)
    
    curriculum = ProgressiveCurriculum(
        initial_seq_length=32,
        target_seq_length=256,
        initial_schema_level=1,
        target_schema_level=8,
        min_epochs_per_phase=5,
        convergence_patience=2,
        improvement_threshold=0.05
    )
    
    # Verify initial state
    assert curriculum.current_seq_length == 32, "Initial seq length should be 32"
    assert curriculum.current_schema_level == 1, "Initial schema level should be 1"
    assert curriculum.target_schema_level == 8, "Target schema level should be 8"
    assert curriculum.sequence_progression == [32, 64, 128, 256], "Sequence progression incorrect"
    assert curriculum.schema_progression == [1, 2, 3, 4, 5, 6, 7, 8], "Schema progression incorrect"
    
    print("✅ Initialization correct")
    print(f"   Schema progression: {curriculum.schema_progression}")
    print(f"   Sequence progression: {curriculum.sequence_progression}")
    return curriculum


def test_schema_progression(curriculum):
    """Test 2: Schema progression logic"""
    print("\n" + "="*70)
    print("TEST 2: Schema Progression Logic")
    print("="*70)
    
    # Simulate training with convergence
    # Need min_epochs_per_phase (5) + convergence_patience (2) epochs
    
    # Epochs 1-5: Improving significantly (>5% each time)
    for i in range(5):
        loss = 1.0 - (i * 0.15)  # Decreasing loss by 15% each time
        result = curriculum.update(loss)
        print(f"Epoch {i+1}: loss={loss:.2f}, should_progress={result['should_progress']}")
        assert not result['should_progress'], f"Should not progress at epoch {i+1}"
    
    # Epochs 6-8: Minor improvement (<5%) to trigger patience
    for i in range(3):
        loss = 0.25 - (i * 0.01)  # Very small improvements
        result = curriculum.update(loss)
        print(f"Epoch {5+i+1}: loss={loss:.2f}, should_progress={result['should_progress']}, patience={curriculum.epochs_without_improvement}")
    
    # Should progress after epoch 7
    assert result['should_progress'], "Should progress after convergence"
    assert 'new_schema_level' in result, "Should have new_schema_level"
    assert result['new_schema_level'] == 2, "Should progress to schema level 2"
    
    print(f"✅ Schema progression works!")
    print(f"   Progressed to schema level: {result['new_schema_level']}")
    return curriculum


def test_sequence_progression():
    """Test 3: Sequence length progression after all schemas"""
    print("\n" + "="*70)
    print("TEST 3: Sequence Length Progression")
    print("="*70)
    
    # Create curriculum at final schema level
    curriculum = ProgressiveCurriculum(
        initial_seq_length=32,
        target_seq_length=256,
        initial_schema_level=8,  # Start at final schema level
        target_schema_level=8,
        min_epochs_per_phase=3,
        convergence_patience=1,
        improvement_threshold=0.05
    )
    
    # Simulate convergence
    for i in range(3):
        curriculum.update(1.0 - i * 0.1)
    
    # Trigger progression
    result = curriculum.update(0.7)
    result = curriculum.update(0.7)
    
    assert result['should_progress'], "Should progress sequence length"
    assert 'new_seq_length' in result, "Should have new_seq_length"
    assert result['new_seq_length'] == 64, "Should progress to seq length 64"
    assert 'new_batch_size' in result, "Should have new_batch_size"
    
    print(f"✅ Sequence progression works!")
    print(f"   New seq length: {result['new_seq_length']}")
    print(f"   New batch size: {result['new_batch_size']}")


def test_completion_detection():
    """Test 4: Auto-stop when curriculum complete"""
    print("\n" + "="*70)
    print("TEST 4: Completion Detection")
    print("="*70)
    
    # Create curriculum at final state
    curriculum = ProgressiveCurriculum(
        initial_seq_length=256,  # Final seq length
        target_seq_length=256,
        initial_schema_level=8,  # Final schema level
        target_schema_level=8,
        min_epochs_per_phase=3,
        convergence_patience=1,
        improvement_threshold=0.05
    )
    
    # Check completion
    is_complete = curriculum.is_complete()
    assert is_complete, "Should be complete at final schema + seq length"
    
    print(f"✅ Completion detection works!")
    print(f"   is_complete(): {is_complete}")
    print(f"   Schema level: {curriculum.current_schema_level}/{curriculum.target_schema_level}")
    print(f"   Seq length: {curriculum.current_seq_length}")


def test_dataset_subset_wrapper():
    """Test 5: Dataset update_seq_length with Subset wrapper"""
    print("\n" + "="*70)
    print("TEST 5: Dataset Subset Wrapper Handling")
    print("="*70)
    
    from torch.utils.data import Subset as TorchSubset
    
    config = TrainingConfig()
    config.cache_dataset_to_ram = True
    config.max_seq_length = 32
    
    # Check if data exists
    data_dir = Path("$PHYSICS_DATA_DIR/physics")
    if not data_dir.exists():
        print("⚠️  SKIPPED: Physics data not found at $PHYSICS_DATA_DIR/physics")
        return
    
    # Create dataset
    print("Loading dataset...")
    dataset = CachedPhysicsDataset(
        data_dir=str(data_dir),
        max_seq_length=32,
        max_objects=20,
        schema_curriculum_level=1,
        max_episodes_per_file=100  # Small for testing
    )
    
    # Wrap in Subset (simulates validation split)
    subset = TorchSubset(dataset, indices=list(range(min(50, len(dataset)))))
    
    # Create dataloader
    dataloader = DataLoader(subset, batch_size=4, shuffle=False)
    
    print(f"Dataset type: {type(dataloader.dataset).__name__}")
    print(f"Dataset length: {len(dataloader.dataset)}")
    
    # Test unwrapping and updating
    from torch.utils.data import Subset
    ds = dataloader.dataset
    if isinstance(ds, Subset):
        ds = ds.dataset
        print(f"Unwrapped to: {type(ds).__name__}")
    
    # Check for update_seq_length method
    has_method = hasattr(ds, 'update_seq_length')
    assert has_method, "Underlying dataset should have update_seq_length"
    
    # Test updating
    print("Updating seq_length from 32 to 64...")
    ds.update_seq_length(64)
    assert ds.max_seq_length == 64, "Seq length should be updated"
    
    print(f"✅ Subset wrapper handling works!")
    print(f"   Can unwrap Subset to access CachedPhysicsDataset")
    print(f"   update_seq_length() works correctly")


def test_full_curriculum_flow():
    """Test 6: Full curriculum progression flow"""
    print("\n" + "="*70)
    print("TEST 6: Full Curriculum Flow (Simulated)")
    print("="*70)
    
    curriculum = ProgressiveCurriculum(
        initial_seq_length=32,
        target_seq_length=128,  # Shorter for testing
        initial_schema_level=1,
        target_schema_level=3,  # Only 3 levels for testing
        min_epochs_per_phase=2,  # Faster progression
        convergence_patience=1,
        improvement_threshold=0.05
    )
    
    total_epochs = 0
    progressions = []
    
    print("\nSimulating training...")
    
    while not curriculum.is_complete():
        # Simulate improving loss
        loss = 1.0 / (total_epochs + 1)
        result = curriculum.update(loss)
        total_epochs += 1
        
        if result['should_progress']:
            progression = {
                'epoch': total_epochs,
                'schema_level': result.get('new_schema_level'),
                'seq_length': result.get('new_seq_length')
            }
            progressions.append(progression)
            print(f"   Epoch {total_epochs}: PROGRESSED - {progression}")
        
        # Safety limit
        if total_epochs > 50:
            print("   ⚠️  Safety limit reached")
            break
    
    print(f"\n✅ Full flow completed!")
    print(f"   Total epochs: {total_epochs}")
    print(f"   Progressions: {len(progressions)}")
    print(f"   Final state: schema={curriculum.current_schema_level}, seq={curriculum.current_seq_length}")
    print(f"   Is complete: {curriculum.is_complete()}")
    
    # Verify we progressed through all levels
    schema_progressions = [p for p in progressions if p['schema_level'] is not None]
    seq_progressions = [p for p in progressions if p['seq_length'] is not None]
    
    print(f"\n   Schema progressions: {len(schema_progressions)}")
    print(f"   Sequence progressions: {len(seq_progressions)}")
    
    assert len(schema_progressions) >= 2, "Should have at least 2 schema progressions"
    assert curriculum.current_schema_level == 3, "Should reach schema level 3"


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("SCHEMA CURRICULUM PROGRESSION TEST SUITE")
    print("="*70)
    
    try:
        # Test 1: Initialization
        curriculum = test_progressive_curriculum_init()
        
        # Test 2: Schema progression
        test_schema_progression(curriculum)
        
        # Test 3: Sequence progression
        test_sequence_progression()
        
        # Test 4: Completion detection
        test_completion_detection()
        
        # Test 5: Dataset subset wrapper
        test_dataset_subset_wrapper()
        
        # Test 6: Full flow
        test_full_curriculum_flow()
        
        # Summary
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nSchema curriculum progression is working correctly:")
        print("  ✅ Curriculum initialization")
        print("  ✅ Schema level progression")
        print("  ✅ Sequence length progression")
        print("  ✅ Completion detection")
        print("  ✅ Dataset Subset wrapper handling")
        print("  ✅ Full curriculum flow")
        print("\nReady for production training! 🚀")
        
        return True
        
    except AssertionError as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST ERROR!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
