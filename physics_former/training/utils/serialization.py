"""
Centralized serialization utilities for JSON/YAML export.

Handles conversion of PyTorch tensors, NumPy arrays, and other non-serializable types
to Python native types that can be safely serialized.
"""

import json
from typing import Any, Dict, List, Union
from pathlib import Path


def to_serializable(obj: Any) -> Any:
    """
    Convert any object to a JSON-serializable format.
    
    Handles:
    - PyTorch tensors → Python lists/scalars
    - NumPy arrays → Python lists/scalars
    - NumPy int/float types → Python int/float
    - Nested dicts/lists recursively
    - Path objects → strings
    
    Args:
        obj: Any object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    # Import here to avoid circular dependencies
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
    
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False
    
    # Handle None, bool, str (already serializable)
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    
    # Handle Path objects
    if isinstance(obj, Path):
        return str(obj)
    
    # Handle PyTorch tensors
    if has_torch and isinstance(obj, torch.Tensor):
        # Move to CPU and convert to numpy, then to list
        obj_cpu = obj.detach().cpu()
        if obj_cpu.numel() == 1:
            # Scalar tensor
            return obj_cpu.item()
        else:
            # Multi-element tensor
            return obj_cpu.numpy().tolist()
    
    # Handle NumPy types
    if has_numpy:
        # NumPy scalars
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # NumPy arrays
        if isinstance(obj, np.ndarray):
            if obj.size == 1:
                # Scalar array
                return to_serializable(obj.item())
            else:
                # Multi-element array
                return obj.tolist()
    
    # Handle Python native numeric types (already serializable)
    if isinstance(obj, (int, float)):
        return obj
    
    # Handle dictionaries recursively
    if isinstance(obj, dict):
        return {to_serializable(k): to_serializable(v) for k, v in obj.items()}
    
    # Handle lists/tuples recursively
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    
    # Handle sets
    if isinstance(obj, set):
        return [to_serializable(x) for x in obj]
    
    # Fallback: try to convert to string
    try:
        return str(obj)
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file with automatic type conversion.
    
    Args:
        data: Data to save (will be converted to serializable format)
        filepath: Path to save JSON file
        indent: JSON indentation (default: 2)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    serializable_data = to_serializable(data)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=indent)


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


class SerializableDict(dict):
    """
    Dictionary that automatically converts values to serializable format on access.
    
    Usage:
        metrics = SerializableDict()
        metrics['accuracy'] = torch.tensor(0.95)
        metrics['loss'] = np.float32(0.123)
        
        # Can be directly saved to JSON
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f)
    """
    
    def __setitem__(self, key, value):
        """Convert key and value to serializable format before storing."""
        super().__setitem__(to_serializable(key), to_serializable(value))
    
    def update(self, *args, **kwargs):
        """Override update to convert all values."""
        if args:
            if len(args) > 1:
                raise TypeError(f"update expected at most 1 arguments, got {len(args)}")
            other = args[0]
            if isinstance(other, dict):
                for key, value in other.items():
                    self[key] = value
            else:
                for key, value in other:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value


# Convenience function for metrics logging
def prepare_metrics_for_logging(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare metrics dictionary for logging (TensorBoard, WandB, etc.).
    
    Converts all tensor/numpy values to Python scalars.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Dictionary with serializable values
    """
    return to_serializable(metrics)


# Example usage
if __name__ == "__main__":
    import torch
    import numpy as np
    
    # Test data with various types
    test_data = {
        'tensor_scalar': torch.tensor(0.95),
        'tensor_array': torch.tensor([1, 2, 3]),
        'numpy_int': np.int64(42),
        'numpy_float': np.float32(3.14),
        'numpy_array': np.array([1.0, 2.0, 3.0]),
        'nested': {
            'accuracy': torch.tensor(0.85),
            'counts': np.array([10, 20, 30])
        },
        'path': Path('/some/path'),
        'normal_int': 123,
        'normal_float': 4.56,
        'normal_str': 'hello',
        'normal_bool': True,
        'normal_none': None
    }
    
    print("Original data types:")
    for key, value in test_data.items():
        print(f"  {key}: {type(value)}")
    
    print("\nConverted data:")
    converted = to_serializable(test_data)
    print(json.dumps(converted, indent=2))
    
    print("\nAll types are JSON-serializable!")
