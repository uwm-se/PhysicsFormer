"""
Test script for index error diagnostic tool.

This demonstrates how to use the diagnostic tool to identify indexing issues.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.diagnostics.index_error_diagnostic import IndexErrorDiagnostic


def test_diagnostic_on_sample_data():
    """Test diagnostic tool on sample data directory."""
    
    # Example usage with different data directories
    test_dirs = [
        Path("data/physics_episodes"),
        Path("data/physics"),
        Path("D:/physics-llm-data/physics"),
    ]
    
    for data_dir in test_dirs:
        if data_dir.exists():
            print(f"\n{'='*70}")
            print(f"Testing diagnostic on: {data_dir}")
            print(f"{'='*70}")
            
            diagnostic = IndexErrorDiagnostic(data_dir=data_dir, verbose=True)
            report = diagnostic.run_full_diagnostic(max_files=3)  # Test first 3 files
            
            # Print summary
            print(f"\nPASS: Diagnostic complete")
            print(f"  Errors: {len(report['errors'])}")
            print(f"  Warnings: {len(report['warnings'])}")
            
            return report
    
    print("FAIL: No valid data directories found")
    print("\nTried:")
    for d in test_dirs:
        print(f"  - {d}")
    
    return None


def test_specific_file():
    """Test diagnostic on a specific file."""
    
    # Example: Test a specific JSON file
    test_file = Path("data/physics_episodes/schema_001.json")
    
    if test_file.exists():
        print(f"\n{'='*70}")
        print(f"Testing specific file: {test_file}")
        print(f"{'='*70}")
        
        diagnostic = IndexErrorDiagnostic(data_dir=test_file.parent, verbose=True)
        result = diagnostic.diagnose_file(test_file, max_episodes_to_check=5)
        
        print(f"\nPASS: File diagnostic complete")
        print(f"  Valid: {result['valid']}")
        print(f"  Episodes: {result['episode_count']}")
        print(f"  Issues: {result['issues']}")
        
        return result
    else:
        print(f"FAIL: File not found: {test_file}")
        return None


if __name__ == "__main__":
    print("="*70)
    print("INDEX ERROR DIAGNOSTIC TEST")
    print("="*70)
    
    # Test on sample data
    report = test_diagnostic_on_sample_data()
    
    # Test specific file (optional)
    # file_result = test_specific_file()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
