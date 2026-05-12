# Physics Dataset Indexing Diagnostic System

## Overview

This diagnostic system helps systematically identify and resolve indexing errors when loading physics data files for training. It performs comprehensive checks on data integrity, structure, and accessibility.

## Features

### 1. **Directory Structure Validation**
- Verifies data directory exists and is accessible
- Checks for presence of JSON files
- Reports file count and structure

### 2. **JSON Format Validation**
- Validates JSON syntax and structure
- Ensures files contain arrays of episodes
- Detects malformed JSON

### 3. **Episode Structure Validation**
- Checks for required fields (`states`, `schema`)
- Validates timestep structure
- Verifies object arrays exist

### 4. **Object State Field Validation**
- Validates presence of physics state fields:
  - `position` (3D vector)
  - `velocity` (3D vector)
  - `orientation` (4D quaternion)
  - `angular_velocity` (3D vector)
  - `mass` (scalar)
  - `size` (3D vector)
- Checks field types and dimensions

### 5. **Indexing Consistency Checks**
- Tests random access to episodes
- Detects IndexError issues
- Validates episode ordering

### 6. **Streaming Access Validation**
- Tests ijson streaming parser
- Validates large file handling
- Checks memory-efficient access

## Usage

### Command Line

```bash
# Basic usage - check all files in directory
python -m physics_former.training.diagnostics.index_error_diagnostic data/physics_episodes

# Check only first 5 files
python -m physics_former.training.diagnostics.index_error_diagnostic data/physics_episodes --max-files 5

# Quiet mode (less verbose)
python -m physics_former.training.diagnostics.index_error_diagnostic data/physics_episodes --quiet

# Save report to JSON file
python -m physics_former.training.diagnostics.index_error_diagnostic data/physics_episodes --output report.json
```

### Python API

```python
from pathlib import Path
from physics_former.training.diagnostics.index_error_diagnostic import IndexErrorDiagnostic

# Initialize diagnostic tool
diagnostic = IndexErrorDiagnostic(
    data_dir=Path("data/physics_episodes"),
    verbose=True
)

# Run full diagnostic
report = diagnostic.run_full_diagnostic(max_files=10)

# Check specific file
file_result = diagnostic.diagnose_file(
    Path("data/physics_episodes/schema_001.json"),
    max_episodes_to_check=5
)

# Access results
print(f"Total errors: {len(report['errors'])}")
print(f"Total warnings: {len(report['warnings'])}")
print(f"Statistics: {report['stats']}")
```

## Common Issues and Solutions

### Issue 1: IndexError when accessing episodes

**Symptoms:**
```
IndexError: list index out of range
```

**Diagnosis:**
- Check if `file_index` is correctly built
- Verify episode counts match actual file contents
- Ensure no off-by-one errors

**Solution:**
```python
# The diagnostic will identify which files have indexing issues
# Regenerate those specific files or adjust indexing logic
```

### Issue 2: Missing required fields

**Symptoms:**
```
KeyError: 'states' or 'schema'
```

**Diagnosis:**
- Episode missing required fields
- Incomplete data generation

**Solution:**
- Regenerate episodes with complete fields
- Add validation during data generation

### Issue 3: Malformed JSON

**Symptoms:**
```
json.JSONDecodeError: Expecting value
```

**Diagnosis:**
- Corrupted JSON file
- Incomplete file write

**Solution:**
- Delete and regenerate corrupted files
- Add atomic file writes during generation

### Issue 4: Shape mismatches

**Symptoms:**
```
RuntimeError: shape mismatch
```

**Diagnosis:**
- Inconsistent array dimensions
- Missing padding
- Wrong state dimensions

**Solution:**
- Verify state_dim matches data (default: 21)
- Check max_objects setting
- Ensure proper padding

### Issue 5: Memory issues with large files

**Symptoms:**
```
MemoryError or system slowdown
```

**Diagnosis:**
- Files too large to load at once
- Missing streaming parser (ijson)

**Solution:**
```bash
# Install ijson for streaming
pip install ijson

# Use max_episodes_per_file limit
dataset = JSONPhysicsDataset(
    data_dir=data_dir,
    max_episodes_per_file=1000  # Limit episodes per file
)
```

## Diagnostic Report Structure

```python
{
    'stats': {
        'total_files': int,
        'valid_files': int,
        'total_episodes': int,
        'corrupted_episodes': int,
        'indexing_errors': int,
        'shape_mismatches': int,
        'missing_fields': int
    },
    'errors': [
        {
            'message': str,
            'exception': str,
            'traceback': str
        }
    ],
    'warnings': [str],
    'file_results': [
        {
            'file': str,
            'valid': bool,
            'episode_count': int,
            'issues': [str]
        }
    ]
}
```

## Integration with Training Pipeline

### Before Training

```python
from physics_former.training.diagnostics.index_error_diagnostic import IndexErrorDiagnostic

# Validate data before starting training
diagnostic = IndexErrorDiagnostic(data_dir=Path("data/physics_episodes"))
report = diagnostic.run_full_diagnostic()

if len(report['errors']) > 0:
    print("FAIL: Data validation failed - fix errors before training")
    sys.exit(1)

# Proceed with training
pipeline = CLSTrainingPipeline(...)
```

### During Training (Error Recovery)

```python
try:
    pipeline.train_stage_1(physics_loader, epochs=epochs)
except IndexError as e:
    print(f"FAIL: IndexError during training: {e}")
    
    # Run diagnostic to identify issue
    diagnostic = IndexErrorDiagnostic(data_dir=Path("data/physics_episodes"))
    report = diagnostic.run_full_diagnostic()
    
    # Save diagnostic report
    with open("error_diagnostic.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("💾 Diagnostic report saved to error_diagnostic.json")
```

## Performance Considerations

### Large Datasets

For datasets with many files or large files:

1. **Use `--max-files` to limit scope:**
   ```bash
   python -m physics_former.training.diagnostics.index_error_diagnostic data --max-files 10
   ```

2. **Run diagnostics on subset first:**
   ```python
   # Check first 3 files in detail
   report = diagnostic.run_full_diagnostic(max_files=3)
   ```

3. **Use streaming validation:**
   - Ensure ijson is installed
   - Diagnostic automatically uses streaming for large files

### Memory Usage

The diagnostic tool is designed to be memory-efficient:
- Uses streaming JSON parsing (ijson)
- Checks episodes in batches
- Doesn't load entire dataset into memory

## Extending the Diagnostic

### Adding Custom Checks

```python
class CustomDiagnostic(IndexErrorDiagnostic):
    def check_custom_field(self, episode: Dict) -> bool:
        """Add custom validation logic."""
        if 'custom_field' not in episode:
            self.log_warning("Missing custom field")
            return False
        return True
    
    def diagnose_file(self, file_path: Path, max_episodes_to_check: int = 10):
        """Override to add custom checks."""
        result = super().diagnose_file(file_path, max_episodes_to_check)
        
        # Add custom validation
        # ...
        
        return result
```

## Troubleshooting

### Diagnostic Tool Issues

**Issue: "ijson not available"**
```bash
pip install ijson
```

**Issue: "Permission denied"**
- Check file permissions
- Run with appropriate user privileges

**Issue: "Out of memory"**
- Reduce `max_files` parameter
- Use `--quiet` flag to reduce output
- Close other applications

## Best Practices

1. **Run diagnostics before training:**
   - Catch issues early
   - Save training time

2. **Save diagnostic reports:**
   - Use `--output` flag
   - Keep reports for debugging

3. **Fix issues systematically:**
   - Start with errors (not warnings)
   - Regenerate corrupted files
   - Validate fixes with diagnostic

4. **Integrate into CI/CD:**
   - Run diagnostics in data generation pipeline
   - Fail builds if validation fails

5. **Monitor during training:**
   - Catch runtime issues
   - Log indexing errors
   - Save error checkpoints

## Related Files

- `json_physics_dataset.py` - Dataset loader that uses indexed access
- `cls_pipeline.py` - Training pipeline that loads physics data
- `generate_pybullet_data.py` - Data generation script

## Support

For issues or questions:
1. Run diagnostic with `--output` to save report
2. Check error messages and recommendations
3. Review common issues section above
4. Regenerate data if necessary
