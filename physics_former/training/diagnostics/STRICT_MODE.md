# Strict Mode: Stop on First Error with Full Diagnostic Dump

## Overview

The strict diagnostic mode immediately stops indexing when an error occurs and dumps comprehensive diagnostic information to the console. This is ideal for debugging and ensuring data quality before training.

## Usage

### Command Line

```bash
# Run strict diagnostic
python -m physics_former.training.diagnostics.index_error_diagnostic_strict data/physics_episodes

# Check only first 3 files
python -m physics_former.training.diagnostics.index_error_diagnostic_strict data/physics_episodes --max-files 3
```

### Python API

```python
from pathlib import Path
from physics_former.training.diagnostics import StrictIndexErrorDiagnostic

# Create strict diagnostic
diagnostic = StrictIndexErrorDiagnostic(data_dir=Path("data/physics_episodes"))

# Run diagnostic (stops on first error)
success = diagnostic.run_strict_diagnostic()

if not success:
    print("Fix errors before continuing")
    sys.exit(1)
```

### With Dataset Loader

```python
from physics_former.training.datasets.json_physics_dataset_safe import SafeJSONPhysicsDataset

# Enable strict mode during initialization
dataset = SafeJSONPhysicsDataset(
    data_dir=Path("data/physics_episodes"),
    strict_mode=True  # Runs strict diagnostic before loading
)

# If we get here, all data is valid
print("PASS: Data validated - ready for training")
```

## Diagnostic Output

When an error is detected, you'll see comprehensive output like this:

```
================================================================================
ALERT: INDEXING ERROR DETECTED - STOPPING IMMEDIATELY
================================================================================

⏰ Timestamp: 2025-11-23 17:51:23

FAIL: ERROR: Episode 42 in schema_001.json missing required field: states

📍 CONTEXT:
  Data Directory: data/physics_episodes
  Current File: schema_001.json
  Episode Index: 42
  Files Checked: 1
  Episodes Checked: 43

🔥 EXCEPTION:
  Type: IndexingError
  Message: Missing required field 'states' at episode 42

📚 TRACEBACK:
  File "index_error_diagnostic_strict.py", line 234, in check_episode_structure
    raise IndexingError(...)
  ...

📄 FILE DETAILS:
  Path: data/physics_episodes/schema_001.json
  Size: 15,234,567 bytes (14.53 MB)
  Readable: True

📦 EPISODE DETAILS:
  Type: dict
  Keys: ['schema', 'metadata', 'object_properties']
  States Type: N/A (missing)
  Schema: collision_basic

⚖️  EXPECTED vs ACTUAL:
  Expected: Field 'states' present
  Actual: Field 'states' missing

💡 IMMEDIATE ACTIONS:
  1. Regenerate file with complete episode structure
  2. Ensure data generation includes all required fields
  3. Delete and regenerate: schema_001.json
  4. Save this diagnostic output for debugging
  5. Check logs for additional context

🖥️  SYSTEM INFO:
  Python: 3.10.0
  Platform: win32
  ijson Available: True

================================================================================
INDEXING STOPPED - FIX ERROR BEFORE CONTINUING
================================================================================
```

## What Gets Checked

### 1. Directory Structure
- Directory exists
- Directory is accessible
- Contains JSON files

### 2. JSON Validity
- Valid JSON syntax
- File is readable
- Contains array of episodes

### 3. Episode Structure
- Episode is a dictionary
- Has required fields: `states`, `schema`
- States is a list
- States is not empty

### 4. Timestep Structure
- First timestep is a dictionary
- Has `objects` field
- Objects is a list

### 5. Indexing Consistency
- Can access first episode (index 0)
- Can access middle episode
- Can access last episode
- No IndexError exceptions

## Common Error Scenarios

### Scenario 1: Missing Data Directory

**Output:**
```
FAIL: ERROR: Data directory does not exist: data/physics_episodes

📍 CONTEXT:
  Expected: Directory exists
  Actual: Directory not found

💡 IMMEDIATE ACTIONS:
  1. Create the data directory or check the path
  2. Run data generation: python generate_all_data.py
```

**Solution:**
```bash
cd physics_former/data_generation
python generate_all_data.py
```

### Scenario 2: Invalid JSON

**Output:**
```
FAIL: ERROR: Invalid JSON in schema_001.json

🔥 EXCEPTION:
  Type: JSONDecodeError
  Message: Expecting value: line 1 column 1 (char 0)

💡 IMMEDIATE ACTIONS:
  1. Delete corrupted JSON file
  2. Regenerate the file
  3. Check disk space and file permissions
```

**Solution:**
```bash
rm data/physics_episodes/schema_001.json
python generate_pybullet_data.py --schema schema_001 --episodes 1000
```

### Scenario 3: Missing Required Field

**Output:**
```
FAIL: ERROR: Episode 42 missing required field: states

📦 EPISODE DETAILS:
  Available fields: ['schema', 'metadata']
  Missing field: states

💡 IMMEDIATE ACTIONS:
  1. Regenerate file with complete episode structure
  2. Ensure data generation includes all required fields
```

**Solution:**
```bash
# Regenerate the file
python generate_pybullet_data.py --schema schema_001 --episodes 1000
```

### Scenario 4: IndexError

**Output:**
```
FAIL: ERROR: IndexError accessing episode 99 in schema_001.json

📊 ARRAY DETAILS:
  array_length: 50
  requested_index: 99
  valid_range: 0 to 49

💡 IMMEDIATE ACTIONS:
  1. Regenerate the corrupted file
  2. Check episode count matches file contents
```

**Solution:**
```bash
# Delete and regenerate
rm data/physics_episodes/schema_001.json
python generate_pybullet_data.py --schema schema_001 --episodes 100
```

## Integration with Training Pipeline

### Before Training (Recommended)

```python
from pathlib import Path
from physics_former.training.diagnostics import StrictIndexErrorDiagnostic

print("🔍 Validating physics data (strict mode)...")

diagnostic = StrictIndexErrorDiagnostic(data_dir=Path("data/physics_episodes"))
success = diagnostic.run_strict_diagnostic()

if not success:
    print("FAIL: Data validation failed!")
    print("Fix errors shown above before starting training")
    sys.exit(1)

print("PASS: Data validation passed - starting training")

# Proceed with training
pipeline = CLSTrainingPipeline(...)
```

### In Dataset Initialization

```python
# Option 1: Strict mode in dataset
dataset = SafeJSONPhysicsDataset(
    data_dir=Path("data/physics_episodes"),
    strict_mode=True  # Validates before loading
)

# Option 2: Manual validation first
from physics_former.training.diagnostics import StrictIndexErrorDiagnostic

diagnostic = StrictIndexErrorDiagnostic(data_dir=Path("data/physics_episodes"))
if not diagnostic.run_strict_diagnostic():
    raise RuntimeError("Data validation failed")

dataset = JSONPhysicsDataset(data_dir=Path("data/physics_episodes"))
```

### In CI/CD Pipeline

```yaml
# .github/workflows/validate_data.yml
name: Validate Physics Data

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run strict diagnostic
        run: |
          python -m physics_former.training.diagnostics.index_error_diagnostic_strict data/physics_episodes
      
      - name: Upload diagnostic report
        if: failure()
        uses: actions/upload-artifact@v2
        with:
          name: diagnostic-output
          path: diagnostic_output.txt
```

## Comparison: Strict vs Regular Mode

| Feature | Regular Mode | Strict Mode |
|---------|-------------|-------------|
| **Stops on error** | No | Yes |
| **Diagnostic dump** | Summary only | Full details |
| **Error context** | Basic | Comprehensive |
| **Traceback** | Optional | Always shown |
| **Recommendations** | Generic | Specific |
| **Use case** | Production | Debugging |
| **Performance** | Fast | Slightly slower |

## When to Use Strict Mode

### PASS: Use Strict Mode When:
- Debugging indexing errors
- Validating newly generated data
- Setting up a new environment
- After data generation changes
- In CI/CD pipelines
- Before important training runs

### FAIL: Don't Use Strict Mode When:
- Data is already validated
- In production with known-good data
- Performance is critical
- You want to skip corrupted episodes

## Performance Impact

- **Overhead:** ~10-20% slower than regular mode
- **Memory:** Same as regular mode (streaming)
- **Disk I/O:** Slightly more (detailed checks)

For most use cases, the performance impact is negligible compared to the debugging time saved.

## Best Practices

1. **Always run strict mode on new data:**
   ```bash
   python -m physics_former.training.diagnostics.index_error_diagnostic_strict data/physics_episodes
   ```

2. **Save diagnostic output:**
   ```bash
   python -m physics_former.training.diagnostics.index_error_diagnostic_strict data/physics_episodes 2>&1 | tee diagnostic.log
   ```

3. **Fix errors immediately:**
   - Don't ignore errors
   - Regenerate corrupted files
   - Validate fixes

4. **Use in development, not production:**
   - Strict mode for debugging
   - Regular/safe mode for production

5. **Integrate with version control:**
   - Run on pre-commit hooks
   - Validate in CI/CD
   - Block merges with errors

## Troubleshooting

### Issue: "Module not found"

```bash
# Ensure correct path
cd physics_former
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "Too much output"

```bash
# Redirect to file
python -m physics_former.training.diagnostics.index_error_diagnostic_strict data/physics_episodes > diagnostic.txt 2>&1
```

### Issue: "Takes too long"

```bash
# Check only first few files
python -m physics_former.training.diagnostics.index_error_diagnostic_strict data/physics_episodes --max-files 3
```

## Summary

Strict mode provides:
- PASS: Immediate error detection
- PASS: Comprehensive diagnostic output
- PASS: Specific recommendations
- PASS: Full context for debugging
- PASS: Prevents training with bad data

Use it to ensure data quality and catch errors early!
