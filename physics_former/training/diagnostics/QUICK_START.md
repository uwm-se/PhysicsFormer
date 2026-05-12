# Quick Start: Diagnosing Indexing Errors

## Problem: Training Pipeline Indexing Errors

When you see errors like:
```
IndexError: list index out of range
KeyError: 'states'
RuntimeError: shape mismatch
```

## Solution: 3-Step Diagnostic Process

### Step 1: Run Diagnostic

**Option A: Strict Mode (Recommended for Debugging)**

Stops immediately on first error with full diagnostic dump:

```bash
cd physics_former
python -m training.diagnostics.index_error_diagnostic_strict data/physics_episodes
```

**Option B: Regular Mode**

Checks all files and reports summary:

```bash
cd physics_former
python -m training.diagnostics.index_error_diagnostic data/physics_episodes
```

Both will:
- PASS: Check all JSON files
- PASS: Validate episode structure
- PASS: Test indexing consistency
- PASS: Report errors and warnings

**Use Strict Mode when:**
- Debugging specific errors
- Validating new data
- Need detailed error context

**Use Regular Mode when:**
- Want overview of all issues
- Checking multiple files
- Need summary report

### Step 2: Review the Report

Look for:
- **Indexing errors** -> Files have wrong episode counts
- **Missing fields** -> Episodes lack required data
- **Corrupted episodes** -> Data is malformed

Example output:
```
📊 Summary Statistics:
  Total files: 10
  Valid files: 9
  Total episodes: 10000
  Corrupted episodes: 5
  Indexing errors: 2
  Missing fields: 3

FAIL: Errors Found: 2
  1. Episode 42 in schema_001.json missing required field: states
  2. IndexError accessing episode 99 in schema_002.json

💡 Recommendations:
  - Fix indexing errors by regenerating corrupted files
  - Ensure all episodes have required fields (states, schema)
```

### Step 3: Fix Issues

#### Option A: Use Safe Dataset Loader with Strict Mode

Replace your dataset with the safe version and enable strict validation:

```python
# OLD:
from physics_former.training.datasets.json_physics_dataset import JSONPhysicsDataset

# NEW:
from physics_former.training.datasets.json_physics_dataset_safe import SafeJSONPhysicsDataset

dataset = SafeJSONPhysicsDataset(
    data_dir=Path("data/physics_episodes"),
    max_objects=10,
    max_seq_length=100,
    strict_mode=True,       # Stop on first error with full diagnostic
    validate_on_init=True,  # Validate on load
    skip_corrupted=True     # Skip bad episodes (if strict_mode=False)
)

# If we get here, all data is valid (strict mode passed)
# Check for any warnings
dataset.print_error_report()
```

#### Option B: Regenerate Corrupted Files

If diagnostic identifies specific files:

```bash
# Delete corrupted file
rm data/physics_episodes/schema_001.json

# Regenerate
cd physics_former/data_generation
python generate_pybullet_data.py --schema schema_001 --episodes 1000
```

## Common Scenarios

### Scenario 1: "No episodes found"

**Diagnostic shows:**
```
WARNING:  No JSON files found in data/physics_episodes
```

**Solution:**
```bash
# Generate data first
cd physics_former/data_generation
python generate_all_data.py
```

### Scenario 2: "IndexError during training"

**Diagnostic shows:**
```
FAIL: IndexError accessing episode 99 in schema_002.json
```

**Solution:**
```python
# Use safe loader that skips corrupted episodes
dataset = SafeJSONPhysicsDataset(
    data_dir=data_dir,
    skip_corrupted=True
)
```

### Scenario 3: "Missing required fields"

**Diagnostic shows:**
```
FAIL: Episode 42 missing required field: states
```

**Solution:**
```bash
# Regenerate the specific file
python generate_pybullet_data.py --schema schema_001 --episodes 1000
```

### Scenario 4: "Memory issues with large files"

**Diagnostic shows:**
```
WARNING:  ijson not available
```

**Solution:**
```bash
# Install streaming parser
pip install ijson

# Or limit episodes per file
dataset = SafeJSONPhysicsDataset(
    data_dir=data_dir,
    max_episodes_per_file=1000  # Limit to 1000 per file
)
```

## Integration with Training

### Before Training (Recommended)

```python
from pathlib import Path
from physics_former.training.diagnostics.index_error_diagnostic import IndexErrorDiagnostic

# Validate data before training
print("🔍 Validating physics data...")
diagnostic = IndexErrorDiagnostic(data_dir=Path("data/physics_episodes"))
report = diagnostic.run_full_diagnostic(max_files=5)  # Check first 5 files

if len(report['errors']) > 0:
    print("FAIL: Data validation failed!")
    print("Fix errors before training or use SafeJSONPhysicsDataset")
    sys.exit(1)

print("PASS: Data validation passed - starting training")
```

### During Training (Error Recovery)

```python
try:
    pipeline.train_stage_1(physics_loader, epochs=epochs)
except (IndexError, KeyError) as e:
    print(f"FAIL: Error during training: {e}")
    
    # Save checkpoint
    emergency_path = pipeline.checkpoint_dir / "emergency.pt"
    pipeline.save_checkpoint(emergency_path, stage=1, epoch=0)
    
    # Run diagnostic
    diagnostic = IndexErrorDiagnostic(data_dir=Path("data/physics_episodes"))
    report = diagnostic.run_full_diagnostic()
    
    # Save report
    with open("error_diagnostic.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("💾 Diagnostic report saved to error_diagnostic.json")
    raise
```

## Advanced Usage

### Check Specific Files Only

```bash
# Check only files matching pattern
python -m training.diagnostics.index_error_diagnostic data/physics_episodes --max-files 3
```

### Save Diagnostic Report

```bash
# Save to JSON for later analysis
python -m training.diagnostics.index_error_diagnostic data/physics_episodes --output diagnostic_report.json
```

### Programmatic Usage

```python
from physics_former.training.diagnostics.index_error_diagnostic import IndexErrorDiagnostic

# Create diagnostic
diagnostic = IndexErrorDiagnostic(
    data_dir=Path("data/physics_episodes"),
    verbose=True
)

# Check specific file
result = diagnostic.diagnose_file(
    Path("data/physics_episodes/schema_001.json"),
    max_episodes_to_check=10
)

print(f"Valid: {result['valid']}")
print(f"Episodes: {result['episode_count']}")
print(f"Issues: {result['issues']}")
```

## Troubleshooting the Diagnostic Tool

### Issue: "Module not found"

```bash
# Ensure you're in the right directory
cd physics_former

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/windsurf-project"
```

### Issue: "Permission denied"

```bash
# Check file permissions
ls -la data/physics_episodes/

# Fix permissions if needed
chmod 644 data/physics_episodes/*.json
```

### Issue: "Out of memory"

```bash
# Use quiet mode and limit files
python -m training.diagnostics.index_error_diagnostic data/physics_episodes --max-files 5 --quiet
```

## Summary

1. **Run diagnostic** to identify issues
2. **Review report** to understand problems
3. **Fix issues** using safe loader or regeneration
4. **Validate** before training
5. **Monitor** during training

## Next Steps

- See `README.md` for detailed documentation
- See `test_diagnostic.py` for usage examples
- See `json_physics_dataset_safe.py` for safe dataset loader
