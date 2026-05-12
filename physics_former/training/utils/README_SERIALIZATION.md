# Centralized Serialization Utilities

## Problem Solved

Previously, type conversion code was scattered everywhere:
- Manual `.item()` calls on tensors
- Custom `int(np.int64(...))` conversions
- Repeated type checking logic
- JSON serialization errors with numpy/torch types

## Solution

**Single import, automatic conversion:**

```python
from utils.serialization import save_json, to_serializable
```

## Usage Examples

### 1. Save Any Data to JSON

```python
from utils.serialization import save_json

# Works with tensors, numpy arrays, nested dicts, etc.
metrics = {
    'accuracy': torch.tensor(0.95),
    'loss': np.float32(0.123),
    'counts': np.array([10, 20, 30]),
    'nested': {
        'tensor': torch.tensor([1, 2, 3]),
        'path': Path('/some/path')
    }
}

# Just save it - automatic conversion!
save_json(metrics, 'metrics.json')
```

### 2. Convert Individual Values

```python
from utils.serialization import to_serializable

# Convert any value
accuracy = torch.tensor(0.95)
serializable_accuracy = to_serializable(accuracy)  # → 0.95 (Python float)

# Works with nested structures
data = {
    'results': [torch.tensor(x) for x in [1, 2, 3]],
    'metadata': {'count': np.int64(42)}
}
clean_data = to_serializable(data)  # All types converted
```

### 3. Prepare Metrics for Logging

```python
from utils.serialization import prepare_metrics_for_logging

# Before logging to TensorBoard, WandB, etc.
metrics = {
    'train_loss': loss_tensor,
    'val_accuracy': acc_tensor
}

# Convert all at once
loggable_metrics = prepare_metrics_for_logging(metrics)
logger.log_metrics(loggable_metrics)
```

### 4. SerializableDict (Auto-Converting Dictionary)

```python
from utils.serialization import SerializableDict

# Dictionary that auto-converts on assignment
metrics = SerializableDict()
metrics['accuracy'] = torch.tensor(0.95)  # Automatically converted to 0.95
metrics['loss'] = np.float32(0.123)       # Automatically converted to 0.123

# Can be directly saved to JSON
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)  # No errors!
```

## What It Handles

### PyTorch Tensors
```python
torch.tensor(0.95)           → 0.95
torch.tensor([1, 2, 3])      → [1, 2, 3]
```

### NumPy Types
```python
np.int64(42)                 → 42
np.float32(3.14)             → 3.14
np.array([1, 2, 3])          → [1, 2, 3]
```

### Path Objects
```python
Path('/some/path')           → '/some/path'
```

### Nested Structures
```python
{
    'tensor': torch.tensor(1),
    'nested': {
        'array': np.array([1, 2])
    }
}
→
{
    'tensor': 1,
    'nested': {
        'array': [1, 2]
    }
}
```

## Migration Guide

### Before (Manual Conversion)

```python
# Old way - scattered conversions
def save_metrics(self, filepath):
    data = {
        'accuracy': float(self.accuracy.item()),
        'counts': [int(x) for x in self.counts.numpy()],
        'nested': {
            int(k): float(v.item()) 
            for k, v in self.confusion_matrix.items()
        }
    }
    with open(filepath, 'w') as f:
        json.dump(data, f)
```

### After (Centralized)

```python
# New way - automatic!
from utils.serialization import save_json

def save_metrics(self, filepath):
    data = {
        'accuracy': self.accuracy,
        'counts': self.counts,
        'nested': self.confusion_matrix
    }
    save_json(data, filepath)
```

## Files Updated

✅ **temporal_metrics.py** - Simplified from 20 lines to 5 lines
✅ **metrics_logger.py** - All JSON saves use centralized utility
✅ **Future files** - Just import and use!

## Benefits

1. **No more type errors** - Handles all conversions automatically
2. **Less code** - Single import instead of repeated logic
3. **Consistent** - Same conversion rules everywhere
4. **Maintainable** - Fix once, works everywhere
5. **Extensible** - Easy to add new type handlers

## Testing

```bash
# Test the serialization utility
python -m training.utils.serialization
```

This will show examples of all supported type conversions.

## When to Use

### Always Use For:
- ✅ Saving metrics to JSON
- ✅ Logging to TensorBoard/WandB
- ✅ Saving checkpoints metadata
- ✅ Exporting results
- ✅ API responses

### Don't Need For:
- ❌ Internal tensor operations (keep as tensors)
- ❌ Model forward pass (keep as tensors)
- ❌ Loss computation (keep as tensors)

## API Reference

### `to_serializable(obj: Any) -> Any`
Convert any object to JSON-serializable format.

### `save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None`
Save data to JSON file with automatic conversion.

### `load_json(filepath: Union[str, Path]) -> Any`
Load data from JSON file.

### `prepare_metrics_for_logging(metrics: Dict[str, Any]) -> Dict[str, Any]`
Prepare metrics dictionary for logging systems.

### `SerializableDict`
Dictionary subclass that auto-converts values on assignment.

---

**Bottom line**: Import once, never worry about type conversion again! 🎉
