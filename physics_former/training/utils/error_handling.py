"""
Standardized Error Handling for PhysicsFormer

Provides consistent error handling, logging, and recovery mechanisms
across the entire codebase.
"""

import sys
import traceback
import logging
from pathlib import Path
from typing import Optional, Callable, Any
from functools import wraps

# ============================================================================
# Custom Exceptions
# ============================================================================

class PhysicsFormerError(Exception):
    """Base exception for all PhysicsFormer errors."""
    pass


class ConfigurationError(PhysicsFormerError):
    """Raised when configuration is invalid."""
    pass


class DataError(PhysicsFormerError):
    """Raised when data is invalid or missing."""
    pass


class ModelError(PhysicsFormerError):
    """Raised when model operation fails."""
    pass


class TrainingError(PhysicsFormerError):
    """Raised when training fails."""
    pass


class CheckpointError(PhysicsFormerError):
    """Raised when checkpoint operations fail."""
    pass


class ValidationError(PhysicsFormerError):
    """Raised when validation fails."""
    pass


class CUDAError(PhysicsFormerError):
    """Raised when CUDA operations fail."""
    pass


# ============================================================================
# Error Handler Class
# ============================================================================

class ErrorHandler:
    """
    Centralized error handling with logging and recovery.
    
    Usage:
        handler = ErrorHandler(log_file='training.log')
        
        # Option 1: Context manager
        with handler.catch(ModelError, "Loading model"):
            model.load_checkpoint(path)
        
        # Option 2: Decorator
        @handler.handle_errors(reraise=True)
        def train_model():
            # training code
            pass
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        verbose: bool = True
    ):
        """
        Initialize error handler.
        
        Args:
            log_file: Path to log file (optional)
            log_level: Logging level
            verbose: Print errors to console
        """
        self.verbose = verbose
        self.logger = self._setup_logger(log_file, log_level)
    
    def _setup_logger(self, log_file: Optional[str], log_level: int) -> logging.Logger:
        """Setup logger with file and console handlers."""
        logger = logging.getLogger('PhysicsFormer')
        logger.setLevel(log_level)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def catch(
        self,
        exception_type: type = Exception,
        context: str = "",
        reraise: bool = False,
        default_return: Any = None
    ):
        """
        Context manager for catching and handling errors.
        
        Args:
            exception_type: Type of exception to catch
            context: Context description for logging
            reraise: Whether to reraise the exception
            default_return: Default value to return on error
        
        Example:
            with handler.catch(ValueError, "Parsing config"):
                config = parse_config(file)
        """
        return _ErrorContext(
            self.logger,
            exception_type,
            context,
            reraise,
            default_return
        )
    
    def handle_errors(
        self,
        exception_type: type = Exception,
        reraise: bool = True,
        default_return: Any = None
    ):
        """
        Decorator for handling errors in functions.
        
        Args:
            exception_type: Type of exception to catch
            reraise: Whether to reraise the exception
            default_return: Default value to return on error
        
        Example:
            @handler.handle_errors(reraise=True)
            def train_model():
                # training code
                pass
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except exception_type as e:
                    self.logger.error(
                        f"Error in {func.__name__}: {str(e)}",
                        exc_info=True
                    )
                    if reraise:
                        raise
                    return default_return
            return wrapper
        return decorator
    
    def log_error(
        self,
        error: Exception,
        context: str = "",
        include_traceback: bool = True
    ):
        """
        Log an error with context.
        
        Args:
            error: The exception
            context: Context description
            include_traceback: Whether to include full traceback
        """
        message = f"{context}: {str(error)}" if context else str(error)
        
        if include_traceback:
            self.logger.error(message, exc_info=True)
        else:
            self.logger.error(message)
    
    def log_warning(self, message: str):
        """Log a warning."""
        self.logger.warning(message)
    
    def log_info(self, message: str):
        """Log info message."""
        self.logger.info(message)


class _ErrorContext:
    """Internal context manager for error handling."""
    
    def __init__(
        self,
        logger: logging.Logger,
        exception_type: type,
        context: str,
        reraise: bool,
        default_return: Any
    ):
        self.logger = logger
        self.exception_type = exception_type
        self.context = context
        self.reraise = reraise
        self.default_return = default_return
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return True
        
        if issubclass(exc_type, self.exception_type):
            message = f"{self.context}: {str(exc_val)}" if self.context else str(exc_val)
            self.logger.error(message, exc_info=True)
            
            if self.reraise:
                return False  # Reraise the exception
            else:
                return True  # Suppress the exception
        
        return False  # Don't suppress other exceptions


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_config(config: Any, required_attrs: list[str]) -> None:
    """
    Validate that config has all required attributes.
    
    Args:
        config: Configuration object
        required_attrs: List of required attribute names
    
    Raises:
        ConfigurationError: If any required attribute is missing
    """
    missing = [attr for attr in required_attrs if not hasattr(config, attr)]
    if missing:
        raise ConfigurationError(
            f"Configuration missing required attributes: {', '.join(missing)}"
        )


def validate_checkpoint(checkpoint_path: Path) -> None:
    """
    Validate that checkpoint file exists and is readable.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Raises:
        CheckpointError: If checkpoint is invalid
    """
    if not checkpoint_path.exists():
        raise CheckpointError(f"Checkpoint not found: {checkpoint_path}")
    
    if not checkpoint_path.is_file():
        raise CheckpointError(f"Checkpoint path is not a file: {checkpoint_path}")
    
    if checkpoint_path.stat().st_size == 0:
        raise CheckpointError(f"Checkpoint file is empty: {checkpoint_path}")


def validate_data_path(data_path: Path, required_files: Optional[list[str]] = None) -> None:
    """
    Validate that data directory exists and contains required files.
    
    Args:
        data_path: Path to data directory
        required_files: List of required file names (optional)
    
    Raises:
        DataError: If data is invalid
    """
    if not data_path.exists():
        raise DataError(f"Data directory not found: {data_path}")
    
    if not data_path.is_dir():
        raise DataError(f"Data path is not a directory: {data_path}")
    
    if required_files:
        missing = [f for f in required_files if not (data_path / f).exists()]
        if missing:
            raise DataError(
                f"Data directory missing required files: {', '.join(missing)}"
            )


def validate_cuda_available() -> None:
    """
    Validate that CUDA is available.
    
    Raises:
        CUDAError: If CUDA is not available
    """
    import torch
    if not torch.cuda.is_available():
        raise CUDAError(
            "CUDA is not available. Please check your PyTorch installation "
            "and GPU drivers."
        )


def validate_memory_available(required_gb: float) -> None:
    """
    Validate that sufficient GPU memory is available.
    
    Args:
        required_gb: Required memory in GB
    
    Raises:
        CUDAError: If insufficient memory
    """
    import torch
    if not torch.cuda.is_available():
        return
    
    available_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if available_gb < required_gb:
        raise CUDAError(
            f"Insufficient GPU memory. Required: {required_gb:.1f} GB, "
            f"Available: {available_gb:.1f} GB"
        )


# ============================================================================
# Recovery Helpers
# ============================================================================

def safe_load_checkpoint(checkpoint_path: Path, map_location: str = 'cpu'):
    """
    Safely load checkpoint with error handling.
    
    Args:
        checkpoint_path: Path to checkpoint
        map_location: Device to map tensors to
    
    Returns:
        Checkpoint dict or None if failed
    """
    import torch
    
    try:
        validate_checkpoint(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        return checkpoint
    except Exception as e:
        logging.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def safe_save_checkpoint(checkpoint: dict, checkpoint_path: Path) -> bool:
    """
    Safely save checkpoint with error handling.
    
    Args:
        checkpoint: Checkpoint dict
        checkpoint_path: Path to save to
    
    Returns:
        True if successful, False otherwise
    """
    import torch
    
    try:
        # Create directory
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to temporary file first
        temp_path = checkpoint_path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        
        # Rename to final path (atomic operation)
        temp_path.replace(checkpoint_path)
        
        return True
    except Exception as e:
        logging.error(f"Failed to save checkpoint {checkpoint_path}: {e}")
        return False


def retry_on_failure(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Retry function on failure.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Delay between retries (seconds)
        exceptions: Tuple of exceptions to catch
    
    Returns:
        Function result
    
    Raises:
        Last exception if all retries fail
    """
    import time
    
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                logging.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logging.error(f"All {max_retries} attempts failed")
    
    raise last_exception


# ============================================================================
# Error Handler Factory (replaces global singleton)
# ============================================================================

def create_error_handler(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    verbose: bool = True
) -> ErrorHandler:
    """
    Create a new ErrorHandler instance.
    
    Use this instead of a global singleton to ensure:
    - Clean state for each training run
    - Testability (can create fresh instances)
    - No hidden dependencies
    - Thread safety (each thread can have its own handler)
    
    Args:
        log_file: Path to log file
        log_level: Logging level
        verbose: Print to console
    
    Returns:
        New ErrorHandler instance
    """
    return ErrorHandler(log_file, log_level, verbose)


# ============================================================================
# Convenience Functions
# ============================================================================

def handle_cuda_oom(func: Callable) -> Callable:
    """
    Decorator to handle CUDA out of memory errors.
    
    Provides helpful suggestions when OOM occurs.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.error(
                    "CUDA out of memory! Try:\n"
                    "  1. Reduce batch_size in config\n"
                    "  2. Enable gradient_checkpointing\n"
                    "  3. Use conservative config\n"
                    "  4. Reduce model size (hidden_dim, num_layers)"
                )
                raise CUDAError(f"CUDA OOM: {e}") from e
            raise
    return wrapper


def handle_file_not_found(func: Callable) -> Callable:
    """
    Decorator to handle file not found errors with helpful messages.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logging.error(
                f"File not found: {e}\n"
                "Make sure data is generated and paths are correct."
            )
            raise DataError(f"File not found: {e}") from e
    return wrapper


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Exceptions
    'PhysicsFormerError',
    'ConfigurationError',
    'DataError',
    'ModelError',
    'TrainingError',
    'CheckpointError',
    'ValidationError',
    'CUDAError',
    
    # Error Handler
    'ErrorHandler',
    'create_error_handler',
    
    # Validation
    'validate_config',
    'validate_checkpoint',
    'validate_data_path',
    'validate_cuda_available',
    'validate_memory_available',
    
    # Recovery
    'safe_load_checkpoint',
    'safe_save_checkpoint',
    'retry_on_failure',
    
    # Decorators
    'handle_cuda_oom',
    'handle_file_not_found',
]
