"""
Test script for strict diagnostic mode.

Demonstrates how the strict diagnostic stops immediately on errors
and dumps comprehensive diagnostic information.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.diagnostics import StrictIndexErrorDiagnostic


def test_strict_diagnostic():
    """Test strict diagnostic on physics data."""
    
    # Example data directories to try
    test_dirs = [
        Path("data/physics_episodes"),
        Path("data/physics"),
        Path("D:/physics-llm-data/physics"),
    ]
    
    for data_dir in test_dirs:
        if data_dir.exists():
            print(f"\n{'='*80}")
            print(f"Testing strict diagnostic on: {data_dir}")
            print(f"{'='*80}\n")
            
            # Create strict diagnostic
            diagnostic = StrictIndexErrorDiagnostic(data_dir=data_dir)
            
            # Run diagnostic (will stop on first error)
            success = diagnostic.run_strict_diagnostic(max_files=3)
            
            if success:
                print("\nPASS: All checks passed!")
                return True
            else:
                print("\nFAIL: Diagnostic failed - see output above for details")
                return False
    
    print("\nFAIL: No valid data directories found")
    print("\nTried:")
    for d in test_dirs:
        print(f"  - {d}")
    
    print("\nGenerate data first:")
    print("  cd physics_former/data_generation")
    print("  python generate_all_data.py")
    
    return False


def test_with_dataset_loader():
    """Test strict mode integrated with dataset loader."""
    
    from training.datasets.json_physics_dataset_safe import SafeJSONPhysicsDataset
    
    data_dir = Path("data/physics_episodes")
    
    if not data_dir.exists():
        print(f"FAIL: Data directory not found: {data_dir}")
        return False
    
    print(f"\n{'='*80}")
    print("Testing strict mode with dataset loader")
    print(f"{'='*80}\n")
    
    try:
        # This will run strict diagnostic before loading
        dataset = SafeJSONPhysicsDataset(
            data_dir=data_dir,
            strict_mode=True,  # Enable strict validation
            max_objects=10,
            max_seq_length=100
        )
        
        print(f"\nPASS: Dataset initialized successfully")
        print(f"   Total episodes: {len(dataset)}")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"   Sample shape: {sample['states'].shape}")
        
        return True
    
    except RuntimeError as e:
        print(f"\nFAIL: Dataset initialization failed: {e}")
        print("   See diagnostic output above for details")
        return False


if __name__ == "__main__":
    print("="*80)
    print("STRICT DIAGNOSTIC MODE TEST")
    print("="*80)
    
    # Test 1: Standalone strict diagnostic
    print("\n[TEST 1] Standalone Strict Diagnostic")
    success1 = test_strict_diagnostic()
    
    # Test 2: Integrated with dataset loader
    if success1:
        print("\n[TEST 2] Strict Mode with Dataset Loader")
        success2 = test_with_dataset_loader()
    else:
        print("\n[TEST 2] Skipped (Test 1 failed)")
        success2 = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Test 1 (Standalone): {'PASS: PASSED' if success1 else 'FAIL: FAILED'}")
    print(f"Test 2 (Integrated): {'PASS: PASSED' if success2 else 'FAIL: FAILED'}")
    print("="*80)
    
    sys.exit(0 if (success1 and success2) else 1)
