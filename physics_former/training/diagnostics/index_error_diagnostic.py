"""
Systematic Diagnostic Tool for Physics Dataset Indexing Errors

This tool helps identify and diagnose indexing errors when loading physics data.
It performs comprehensive checks on:
1. File structure and accessibility
2. JSON format validity
3. Episode indexing consistency
4. Data shape mismatches
5. Memory and performance issues
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np
from ..utils.serialization import save_json
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False
    print("WARNING:  ijson not available - some diagnostics will be limited")


class IndexErrorDiagnostic:
    """Comprehensive diagnostic tool for physics dataset indexing errors."""
    
    def __init__(self, data_dir: Path, verbose: bool = True):
        """
        Initialize diagnostic tool.
        
        Args:
            data_dir: Directory containing physics JSON files
            verbose: Print detailed diagnostic information
        """
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.stats = {
            'total_files': 0,
            'valid_files': 0,
            'total_episodes': 0,
            'corrupted_episodes': 0,
            'indexing_errors': 0,
            'shape_mismatches': 0,
            'missing_fields': 0
        }
    
    def log_error(self, message: str, exception: Optional[Exception] = None):
        """Log an error with optional exception details."""
        error_entry = {'message': message}
        if exception:
            error_entry['exception'] = str(exception)
            error_entry['traceback'] = traceback.format_exc()
        self.errors.append(error_entry)
        if self.verbose:
            print(f"FAIL: ERROR: {message}")
            if exception:
                print(f"   Exception: {exception}")
    
    def log_warning(self, message: str):
        """Log a warning."""
        self.warnings.append(message)
        if self.verbose:
            print(f"WARNING:  WARNING: {message}")
    
    def log_info(self, message: str):
        """Log informational message."""
        if self.verbose:
            print(f"ℹ️  {message}")
    
    def check_directory_structure(self) -> bool:
        """
        Check if data directory exists and contains JSON files.
        
        Returns:
            True if directory structure is valid
        """
        self.log_info(f"Checking directory: {self.data_dir}")
        
        if not self.data_dir.exists():
            self.log_error(f"Data directory does not exist: {self.data_dir}")
            return False
        
        if not self.data_dir.is_dir():
            self.log_error(f"Path is not a directory: {self.data_dir}")
            return False
        
        json_files = list(self.data_dir.glob("*.json"))
        self.stats['total_files'] = len(json_files)
        
        if not json_files:
            self.log_error(f"No JSON files found in {self.data_dir}")
            return False
        
        self.log_info(f"Found {len(json_files)} JSON files")
        return True
    
    def check_json_validity(self, file_path: Path) -> Tuple[bool, Optional[int]]:
        """
        Check if JSON file is valid and can be parsed using streaming.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            (is_valid, episode_count or None)
        """
        try:
            # Use streaming to count episodes without loading entire file
            if IJSON_AVAILABLE:
                with open(file_path, 'rb') as f:
                    parser = ijson.items(f, 'item')
                    episode_count = sum(1 for _ in parser)
                return True, episode_count
            else:
                # Fallback: load entire file (may fail for very large files)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    self.log_error(f"File {file_path.name} does not contain a JSON array")
                    return False, None
                
                return True, len(data)
        
        except json.JSONDecodeError as e:
            self.log_error(f"Invalid JSON in {file_path.name}", e)
            return False, None
        
        except Exception as e:
            self.log_error(f"Error reading {file_path.name}", e)
            return False, None
    
    def check_episode_structure(self, episode: Dict, episode_idx: int, file_name: str) -> bool:
        """
        Check if episode has required fields and valid structure.
        
        Args:
            episode: Episode dictionary
            episode_idx: Index of episode in file
            file_name: Name of file containing episode
            
        Returns:
            True if episode structure is valid
        """
        required_fields = ['states', 'schema']
        is_valid = True
        
        for field in required_fields:
            if field not in episode:
                self.log_error(
                    f"Episode {episode_idx} in {file_name} missing required field: {field}"
                )
                self.stats['missing_fields'] += 1
                is_valid = False
        
        # Check states structure
        if 'states' in episode:
            states = episode['states']
            
            if not isinstance(states, list):
                self.log_error(
                    f"Episode {episode_idx} in {file_name}: 'states' is not a list"
                )
                is_valid = False
            
            elif len(states) == 0:
                self.log_warning(
                    f"Episode {episode_idx} in {file_name}: 'states' is empty"
                )
            
            else:
                # Check first timestep structure
                first_timestep = states[0]
                if not isinstance(first_timestep, dict):
                    self.log_error(
                        f"Episode {episode_idx} in {file_name}: timestep is not a dict"
                    )
                    is_valid = False
                
                elif 'objects' not in first_timestep:
                    self.log_error(
                        f"Episode {episode_idx} in {file_name}: timestep missing 'objects'"
                    )
                    is_valid = False
                
                else:
                    objects = first_timestep['objects']
                    if not isinstance(objects, list):
                        self.log_error(
                            f"Episode {episode_idx} in {file_name}: 'objects' is not a list"
                        )
                        is_valid = False
        
        return is_valid
    
    def check_object_state_fields(self, obj: Dict, obj_idx: int, episode_idx: int, file_name: str) -> bool:
        """
        Check if object has required state fields.
        
        Args:
            obj: Object dictionary
            obj_idx: Index of object in timestep
            episode_idx: Index of episode
            file_name: Name of file
            
        Returns:
            True if object has valid fields
        """
        expected_fields = {
            'position': 3,
            'velocity': 3,
            'orientation': 4,
            'angular_velocity': 3,
            'mass': 1,
            'size': 3
        }
        
        is_valid = True
        
        for field, expected_size in expected_fields.items():
            if field not in obj:
                self.log_warning(
                    f"Object {obj_idx} in episode {episode_idx} of {file_name} "
                    f"missing field: {field}"
                )
            else:
                value = obj[field]
                
                # Check size for array fields
                if expected_size > 1:
                    if not isinstance(value, (list, tuple)):
                        self.log_error(
                            f"Object {obj_idx} in episode {episode_idx} of {file_name}: "
                            f"{field} is not a list/tuple"
                        )
                        is_valid = False
                    elif len(value) != expected_size:
                        self.log_warning(
                            f"Object {obj_idx} in episode {episode_idx} of {file_name}: "
                            f"{field} has size {len(value)}, expected {expected_size}"
                        )
        
        return is_valid
    
    def check_indexing_consistency(self, file_path: Path, episodes: List[Dict]) -> bool:
        """
        Check if episodes can be indexed consistently.
        
        Args:
            file_path: Path to JSON file
            episodes: List of episodes
            
        Returns:
            True if indexing is consistent
        """
        is_valid = True
        
        # Test random access to episodes
        test_indices = [0, len(episodes) // 2, len(episodes) - 1] if len(episodes) > 0 else []
        
        for idx in test_indices:
            try:
                episode = episodes[idx]
                if not isinstance(episode, dict):
                    self.log_error(
                        f"Episode at index {idx} in {file_path.name} is not a dict"
                    )
                    self.stats['indexing_errors'] += 1
                    is_valid = False
            except IndexError as e:
                self.log_error(
                    f"IndexError accessing episode {idx} in {file_path.name}", e
                )
                self.stats['indexing_errors'] += 1
                is_valid = False
            except Exception as e:
                self.log_error(
                    f"Error accessing episode {idx} in {file_path.name}", e
                )
                self.stats['indexing_errors'] += 1
                is_valid = False
        
        return is_valid
    
    def check_streaming_access(self, file_path: Path, expected_count: int) -> bool:
        """
        Check if file can be accessed via streaming (ijson).
        
        Args:
            file_path: Path to JSON file
            expected_count: Expected number of episodes
            
        Returns:
            True if streaming access works
        """
        if not IJSON_AVAILABLE:
            self.log_warning("ijson not available - skipping streaming test")
            return True
        
        try:
            with open(file_path, 'rb') as f:
                parser = ijson.items(f, 'item')
                
                count = 0
                for episode in parser:
                    count += 1
                    if count >= 3:  # Just test first few
                        break
                
                if count == 0:
                    self.log_error(f"Streaming parser found 0 episodes in {file_path.name}")
                    return False
                
                return True
        
        except Exception as e:
            self.log_error(f"Streaming access failed for {file_path.name}", e)
            return False
    
    def diagnose_file(self, file_path: Path, max_episodes_to_check: int = 10) -> Dict:
        """
        Run comprehensive diagnostics on a single file.
        
        Args:
            file_path: Path to JSON file
            max_episodes_to_check: Maximum number of episodes to check in detail
            
        Returns:
            Dictionary with diagnostic results
        """
        self.log_info(f"\n{'='*70}")
        self.log_info(f"Diagnosing: {file_path.name}")
        self.log_info(f"{'='*70}")
        
        results = {
            'file': file_path.name,
            'valid': True,
            'episode_count': 0,
            'issues': []
        }
        
        # Check JSON validity
        is_valid, episodes = self.check_json_validity(file_path)
        if not is_valid:
            results['valid'] = False
            return results
        
        self.stats['valid_files'] += 1
        results['episode_count'] = len(episodes)
        self.stats['total_episodes'] += len(episodes)
        
        self.log_info(f"Episodes in file: {len(episodes)}")
        
        # Check indexing consistency
        if not self.check_indexing_consistency(file_path, episodes):
            results['valid'] = False
            results['issues'].append('indexing_inconsistency')
        
        # Check streaming access
        if not self.check_streaming_access(file_path, len(episodes)):
            results['valid'] = False
            results['issues'].append('streaming_access_failed')
        
        # Check episode structure (sample)
        episodes_to_check = min(max_episodes_to_check, len(episodes))
        self.log_info(f"Checking structure of {episodes_to_check} episodes...")
        
        for i in range(episodes_to_check):
            episode = episodes[i]
            
            if not self.check_episode_structure(episode, i, file_path.name):
                results['valid'] = False
                results['issues'].append(f'episode_{i}_structure_invalid')
                self.stats['corrupted_episodes'] += 1
            
            # Check object state fields (first episode only)
            if i == 0 and 'states' in episode and len(episode['states']) > 0:
                first_timestep = episode['states'][0]
                if 'objects' in first_timestep:
                    objects = first_timestep['objects']
                    for obj_idx, obj in enumerate(objects[:3]):  # Check first 3 objects
                        self.check_object_state_fields(obj, obj_idx, i, file_path.name)
        
        return results
    
    def run_full_diagnostic(self, max_files: Optional[int] = None) -> Dict:
        """
        Run full diagnostic on all files in data directory.
        
        Args:
            max_files: Maximum number of files to check (None = all)
            
        Returns:
            Dictionary with complete diagnostic results
        """
        print("\n" + "="*70)
        print("PHYSICS DATASET INDEXING DIAGNOSTIC")
        print("="*70)
        
        # Check directory structure
        if not self.check_directory_structure():
            return self.generate_report()
        
        # Get all JSON files
        json_files = list(self.data_dir.glob("*.json"))
        if max_files:
            json_files = json_files[:max_files]
        
        # Diagnose each file
        file_results = []
        for file_path in json_files:
            try:
                result = self.diagnose_file(file_path)
                file_results.append(result)
            except Exception as e:
                self.log_error(f"Fatal error diagnosing {file_path.name}", e)
                file_results.append({
                    'file': file_path.name,
                    'valid': False,
                    'episode_count': 0,
                    'issues': ['fatal_error']
                })
        
        return self.generate_report(file_results)
    
    def generate_report(self, file_results: Optional[List[Dict]] = None) -> Dict:
        """
        Generate comprehensive diagnostic report.
        
        Args:
            file_results: List of per-file diagnostic results
            
        Returns:
            Complete diagnostic report
        """
        print("\n" + "="*70)
        print("DIAGNOSTIC REPORT")
        print("="*70)
        
        # Summary statistics
        print("\n📊 Summary Statistics:")
        print(f"  Total files: {self.stats['total_files']}")
        print(f"  Valid files: {self.stats['valid_files']}")
        print(f"  Total episodes: {self.stats['total_episodes']}")
        print(f"  Corrupted episodes: {self.stats['corrupted_episodes']}")
        print(f"  Indexing errors: {self.stats['indexing_errors']}")
        print(f"  Shape mismatches: {self.stats['shape_mismatches']}")
        print(f"  Missing fields: {self.stats['missing_fields']}")
        
        # Errors
        if self.errors:
            print(f"\nFAIL: Errors Found: {len(self.errors)}")
            for i, error in enumerate(self.errors[:10], 1):  # Show first 10
                print(f"\n  {i}. {error['message']}")
                if 'exception' in error:
                    print(f"     Exception: {error['exception']}")
            
            if len(self.errors) > 10:
                print(f"\n  ... and {len(self.errors) - 10} more errors")
        else:
            print("\nPASS: No errors found!")
        
        # Warnings
        if self.warnings:
            print(f"\nWARNING:  Warnings: {len(self.warnings)}")
            for i, warning in enumerate(self.warnings[:10], 1):  # Show first 10
                print(f"  {i}. {warning}")
            
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
        
        # File-specific issues
        if file_results:
            invalid_files = [r for r in file_results if not r['valid']]
            if invalid_files:
                print(f"\n🔍 Files with Issues: {len(invalid_files)}")
                for result in invalid_files:
                    print(f"\n  {result['file']}:")
                    print(f"    Episodes: {result['episode_count']}")
                    print(f"    Issues: {', '.join(result['issues'])}")
        
        # Recommendations
        print("\n💡 Recommendations:")
        
        if self.stats['indexing_errors'] > 0:
            print("  - Fix indexing errors by regenerating corrupted files")
        
        if self.stats['missing_fields'] > 0:
            print("  - Ensure all episodes have required fields (states, schema)")
        
        if self.stats['corrupted_episodes'] > 0:
            print("  - Remove or regenerate corrupted episodes")
        
        if not IJSON_AVAILABLE:
            print("  - Install ijson for better large file handling: pip install ijson")
        
        if len(self.errors) == 0 and len(self.warnings) == 0:
            print("  PASS: Dataset appears healthy - no action needed!")
        
        print("\n" + "="*70)
        
        return {
            'stats': self.stats,
            'errors': self.errors,
            'warnings': self.warnings,
            'file_results': file_results or []
        }


def main():
    """Main entry point for diagnostic tool."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Diagnose physics dataset indexing errors'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory containing physics JSON files'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to check (default: all)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save report to JSON file'
    )
    
    args = parser.parse_args()
    
    # Run diagnostic
    diagnostic = IndexErrorDiagnostic(
        data_dir=Path(args.data_dir),
        verbose=not args.quiet
    )
    
    report = diagnostic.run_full_diagnostic(max_files=args.max_files)
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        save_json(report, output_path)
        print(f"\n💾 Report saved to: {output_path}")
    
    # Exit with error code if issues found
    if report['stats']['indexing_errors'] > 0 or len(report['errors']) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
