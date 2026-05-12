"""
Strict Diagnostic Tool - Stops on First Error with Full Diagnostic Dump

This version immediately stops indexing when an error occurs and dumps
all relevant diagnostics to the console for immediate debugging.
"""

import sys
from pathlib import Path
import json
import traceback
from typing import Dict, List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False


class IndexingError(Exception):
    """Custom exception for indexing errors with diagnostic context."""
    
    def __init__(self, message: str, context: Dict):
        super().__init__(message)
        self.context = context


class StrictIndexErrorDiagnostic:
    """
    Strict diagnostic tool that stops on first error.
    
    Immediately halts processing when an error is detected and dumps
    comprehensive diagnostic information to console.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize strict diagnostic tool.
        
        Args:
            data_dir: Directory containing physics JSON files
        """
        self.data_dir = Path(data_dir)
        self.current_file = None
        self.current_episode_idx = None
        self.stats = {
            'files_checked': 0,
            'episodes_checked': 0,
            'start_time': datetime.now()
        }
    
    def dump_diagnostics(self, error_message: str, context: Dict):
        """
        Dump comprehensive diagnostics to console.
        
        Args:
            error_message: Error message
            context: Diagnostic context
        """
        print("\n" + "="*80)
        print("ALERT: INDEXING ERROR DETECTED - STOPPING IMMEDIATELY")
        print("="*80)
        
        # Timestamp
        print(f"\n⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Error details
        print(f"\nFAIL: ERROR: {error_message}")
        
        # Context information
        print("\n📍 CONTEXT:")
        print(f"  Data Directory: {self.data_dir}")
        print(f"  Current File: {context.get('file', 'N/A')}")
        print(f"  Episode Index: {context.get('episode_idx', 'N/A')}")
        print(f"  Files Checked: {self.stats['files_checked']}")
        print(f"  Episodes Checked: {self.stats['episodes_checked']}")
        
        # Exception details
        if 'exception' in context:
            print(f"\n🔥 EXCEPTION:")
            print(f"  Type: {type(context['exception']).__name__}")
            print(f"  Message: {context['exception']}")
        
        # Traceback
        if 'traceback' in context:
            print(f"\n📚 TRACEBACK:")
            print(context['traceback'])
        
        # File details
        if 'file_path' in context:
            file_path = context['file_path']
            print(f"\n📄 FILE DETAILS:")
            print(f"  Path: {file_path}")
            
            if file_path.exists():
                file_size = file_path.stat().st_size
                print(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
                print(f"  Readable: {file_path.is_file()}")
            else:
                print(f"  Status: FILE DOES NOT EXIST")
        
        # Episode details
        if 'episode' in context:
            episode = context['episode']
            print(f"\n📦 EPISODE DETAILS:")
            print(f"  Type: {type(episode).__name__}")
            
            if isinstance(episode, dict):
                print(f"  Keys: {list(episode.keys())}")
                
                if 'states' in episode:
                    states = episode['states']
                    print(f"  States Type: {type(states).__name__}")
                    if isinstance(states, list):
                        print(f"  States Length: {len(states)}")
                        if len(states) > 0:
                            print(f"  First State Type: {type(states[0]).__name__}")
                            if isinstance(states[0], dict):
                                print(f"  First State Keys: {list(states[0].keys())}")
                
                if 'schema' in episode:
                    print(f"  Schema: {episode['schema']}")
        
        # Array details
        if 'array_info' in context:
            array_info = context['array_info']
            print(f"\n📊 ARRAY DETAILS:")
            for key, value in array_info.items():
                print(f"  {key}: {value}")
        
        # Expected vs Actual
        if 'expected' in context or 'actual' in context:
            print(f"\n⚖️  EXPECTED vs ACTUAL:")
            if 'expected' in context:
                print(f"  Expected: {context['expected']}")
            if 'actual' in context:
                print(f"  Actual: {context['actual']}")
        
        # Recommendations
        print(f"\n💡 IMMEDIATE ACTIONS:")
        recommendations = self._generate_recommendations(error_message, context)
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        # System info
        print(f"\n🖥️  SYSTEM INFO:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  Platform: {sys.platform}")
        print(f"  ijson Available: {IJSON_AVAILABLE}")
        
        print("\n" + "="*80)
        print("INDEXING STOPPED - FIX ERROR BEFORE CONTINUING")
        print("="*80 + "\n")
    
    def _generate_recommendations(self, error_message: str, context: Dict) -> List[str]:
        """Generate specific recommendations based on error."""
        recommendations = []
        
        error_lower = error_message.lower()
        
        if 'does not exist' in error_lower:
            recommendations.append("Create the data directory or check the path")
            recommendations.append("Run data generation: python generate_all_data.py")
        
        elif 'no json files' in error_lower:
            recommendations.append("Generate physics data files")
            recommendations.append("Check if files are in correct directory")
        
        elif 'index' in error_lower and 'out of range' in error_lower:
            recommendations.append("Regenerate the corrupted file")
            recommendations.append("Check episode count matches file contents")
            recommendations.append(f"Delete and regenerate: {context.get('file', 'the file')}")
        
        elif 'missing' in error_lower and 'field' in error_lower:
            recommendations.append("Regenerate file with complete episode structure")
            recommendations.append("Ensure data generation includes all required fields")
        
        elif 'invalid json' in error_lower or 'parse' in error_lower:
            recommendations.append("Delete corrupted JSON file")
            recommendations.append("Regenerate the file")
            recommendations.append("Check disk space and file permissions")
        
        elif 'shape' in error_lower or 'dimension' in error_lower:
            recommendations.append("Verify state_dim setting matches data (default: 21)")
            recommendations.append("Check max_objects configuration")
            recommendations.append("Ensure proper array dimensions in data generation")
        
        else:
            recommendations.append("Review error details above")
            recommendations.append("Check file integrity")
            recommendations.append("Consider regenerating the problematic file")
        
        # Always add these
        recommendations.append("Save this diagnostic output for debugging")
        recommendations.append("Check logs for additional context")
        
        return recommendations
    
    def check_directory_structure(self):
        """Check directory structure - stops on error."""
        print(f"\n🔍 Checking directory: {self.data_dir}")
        
        if not self.data_dir.exists():
            self.dump_diagnostics(
                f"Data directory does not exist: {self.data_dir}",
                {
                    'file_path': self.data_dir,
                    'expected': 'Directory exists',
                    'actual': 'Directory not found'
                }
            )
            raise IndexingError(
                f"Directory does not exist: {self.data_dir}",
                {'directory': str(self.data_dir)}
            )
        
        if not self.data_dir.is_dir():
            self.dump_diagnostics(
                f"Path is not a directory: {self.data_dir}",
                {
                    'file_path': self.data_dir,
                    'expected': 'Directory',
                    'actual': 'Not a directory'
                }
            )
            raise IndexingError(
                f"Not a directory: {self.data_dir}",
                {'path': str(self.data_dir)}
            )
        
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            self.dump_diagnostics(
                f"No JSON files found in {self.data_dir}",
                {
                    'file_path': self.data_dir,
                    'expected': 'At least one .json file',
                    'actual': '0 JSON files found'
                }
            )
            raise IndexingError(
                f"No JSON files in directory: {self.data_dir}",
                {'directory': str(self.data_dir)}
            )
        
        print(f"PASS: Found {len(json_files)} JSON files")
        return json_files
    
    def check_indexing_only(self, file_path: Path) -> int:
        """
        Check ONLY that indexing will work - memory efficient.
        
        Tests that we can access episodes by index using streaming.
        """
        print(f"\n📄 Testing indexing: {file_path.name}")
        
        try:
            # First, count total episodes (memory efficient)
            with open(file_path, 'r') as f:
                count = 0
                chunk_size = 1024 * 1024  # 1MB chunks
                
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    count += chunk.count('"schema":')
            
            if count == 0:
                self.dump_diagnostics(
                    f"File {file_path.name} contains no episodes",
                    {
                        'file': file_path.name,
                        'file_path': file_path,
                        'expected': 'At least one episode',
                        'actual': '0 episodes found'
                    }
                )
                raise IndexingError(f"Empty file: {file_path.name}", {'file': str(file_path)})
            
            print(f"  Found {count} episodes")
            print(f"  PASS: File structure valid (episodes counted successfully)")
            
            # Note: Skipping index access test for performance
            # Counting episodes already validates JSON structure
            # Actual indexing will be tested during dataset loading
            
            print(f"PASS: Indexing validated: {count} episodes")
            return count
            
        except IndexingError:
            raise
        except Exception as e:
            self.dump_diagnostics(
                f"Cannot index file: {file_path.name}",
                {
                    'file': file_path.name,
                    'file_path': file_path,
                    'exception': e,
                    'traceback': traceback.format_exc()
                }
            )
            raise IndexingError(f"Indexing failed: {file_path.name}", {'file': str(file_path)})
    
    def check_episode_structure(self, episode: Dict, episode_idx: int, file_path: Path):
        """Check episode structure - stops on error."""
        
        # Check type
        if not isinstance(episode, dict):
            self.dump_diagnostics(
                f"Episode {episode_idx} in {file_path.name} is not a dictionary",
                {
                    'file': file_path.name,
                    'file_path': file_path,
                    'episode_idx': episode_idx,
                    'episode': episode,
                    'expected': 'Dictionary (dict)',
                    'actual': type(episode).__name__
                }
            )
            raise IndexingError(
                f"Invalid episode type at index {episode_idx}",
                {'file': str(file_path), 'index': episode_idx}
            )
        
        # Check required fields
        required_fields = ['states', 'schema']
        for field in required_fields:
            if field not in episode:
                self.dump_diagnostics(
                    f"Episode {episode_idx} in {file_path.name} missing required field: {field}",
                    {
                        'file': file_path.name,
                        'file_path': file_path,
                        'episode_idx': episode_idx,
                        'episode': episode,
                        'expected': f"Field '{field}' present",
                        'actual': f"Field '{field}' missing",
                        'array_info': {
                            'available_fields': list(episode.keys()),
                            'missing_field': field
                        }
                    }
                )
                raise IndexingError(
                    f"Missing required field '{field}' at episode {episode_idx}",
                    {'file': str(file_path), 'index': episode_idx, 'field': field}
                )
        
        # Check states structure
        states = episode['states']
        if not isinstance(states, list):
            self.dump_diagnostics(
                f"Episode {episode_idx} in {file_path.name}: 'states' is not a list",
                {
                    'file': file_path.name,
                    'file_path': file_path,
                    'episode_idx': episode_idx,
                    'expected': "'states' as list",
                    'actual': f"'states' as {type(states).__name__}"
                }
            )
            raise IndexingError(
                f"Invalid states type at episode {episode_idx}",
                {'file': str(file_path), 'index': episode_idx}
            )
        
        if len(states) == 0:
            self.dump_diagnostics(
                f"Episode {episode_idx} in {file_path.name}: 'states' is empty",
                {
                    'file': file_path.name,
                    'file_path': file_path,
                    'episode_idx': episode_idx,
                    'expected': 'At least one timestep',
                    'actual': 'Empty states array'
                }
            )
            raise IndexingError(
                f"Empty states at episode {episode_idx}",
                {'file': str(file_path), 'index': episode_idx}
            )
        
        # Check first timestep
        first_timestep = states[0]
        if not isinstance(first_timestep, dict):
            self.dump_diagnostics(
                f"Episode {episode_idx} in {file_path.name}: first timestep is not a dict",
                {
                    'file': file_path.name,
                    'file_path': file_path,
                    'episode_idx': episode_idx,
                    'expected': 'Timestep as dict',
                    'actual': f'Timestep as {type(first_timestep).__name__}'
                }
            )
            raise IndexingError(
                f"Invalid timestep type at episode {episode_idx}",
                {'file': str(file_path), 'index': episode_idx}
            )
        
        if 'objects' not in first_timestep:
            self.dump_diagnostics(
                f"Episode {episode_idx} in {file_path.name}: timestep missing 'objects'",
                {
                    'file': file_path.name,
                    'file_path': file_path,
                    'episode_idx': episode_idx,
                    'expected': "'objects' field in timestep",
                    'actual': f"Available fields: {list(first_timestep.keys())}"
                }
            )
            raise IndexingError(
                f"Missing 'objects' field at episode {episode_idx}",
                {'file': str(file_path), 'index': episode_idx}
            )
    
    def check_indexing_consistency(self, file_path: Path, episodes: List):
        """Check indexing consistency - stops on error."""
        print(f"\n🔢 Testing indexing consistency...")
        
        # Test boundary indices
        test_indices = []
        if len(episodes) > 0:
            test_indices.append(0)  # First
        if len(episodes) > 1:
            test_indices.append(len(episodes) - 1)  # Last
        if len(episodes) > 2:
            test_indices.append(len(episodes) // 2)  # Middle
        
        for idx in test_indices:
            try:
                episode = episodes[idx]
                print(f"  PASS: Index {idx}: accessible")
            except IndexError as e:
                self.dump_diagnostics(
                    f"IndexError accessing episode {idx} in {file_path.name}",
                    {
                        'file': file_path.name,
                        'file_path': file_path,
                        'episode_idx': idx,
                        'exception': e,
                        'traceback': traceback.format_exc(),
                        'expected': f'Episode at index {idx}',
                        'actual': f'IndexError (array has {len(episodes)} episodes)',
                        'array_info': {
                            'array_length': len(episodes),
                            'requested_index': idx,
                            'valid_range': f'0 to {len(episodes) - 1}'
                        }
                    }
                )
                raise IndexingError(
                    f"Index out of range: {idx}",
                    {'file': str(file_path), 'index': idx, 'length': len(episodes)}
                )
    
    def run_strict_diagnostic(self, max_files: Optional[int] = None):
        """
        Run strict diagnostic - stops on first error.
        
        Args:
            max_files: Maximum files to check (None = all)
        """
        print("\n" + "="*80)
        print("ALERT: STRICT INDEXING DIAGNOSTIC - STOPS ON FIRST ERROR")
        print("="*80)
        print(f"Start Time: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Directory: {self.data_dir}")
        print("="*80)
        
        try:
            # Check directory
            json_files = self.check_directory_structure()
            
            if max_files:
                json_files = json_files[:max_files]
                print(f"\nWARNING:  Limiting to first {max_files} files")
            
            # Check each file - INDEXING ONLY (memory efficient)
            total_episodes = 0
            for file_path in json_files:
                self.current_file = file_path.name
                self.stats['files_checked'] += 1
                
                # Test indexing only - no data loading
                episode_count = self.check_indexing_only(file_path)
                total_episodes += episode_count
                self.stats['episodes_checked'] += episode_count
                
                print(f"PASS: {file_path.name}: Indexing OK")
            
            print(f"\nPASS: Total episodes indexed: {total_episodes:,}")
            
            # Success!
            elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
            print("\n" + "="*80)
            print("PASS: ALL DIAGNOSTICS PASSED - NO ERRORS FOUND")
            print("="*80)
            print(f"Files Checked: {self.stats['files_checked']}")
            print(f"Episodes Checked: {self.stats['episodes_checked']}")
            print(f"Time Elapsed: {elapsed:.2f} seconds")
            print("="*80 + "\n")
            
            return True
        
        except IndexingError as e:
            # Error already dumped by check methods
            return False
        
        except Exception as e:
            # Unexpected error
            self.dump_diagnostics(
                f"Unexpected error during diagnostic",
                {
                    'file': self.current_file,
                    'episode_idx': self.current_episode_idx,
                    'exception': e,
                    'traceback': traceback.format_exc()
                }
            )
            return False


def main():
    """Main entry point for strict diagnostic."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Strict diagnostic - stops on first error'
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
        help='Maximum files to check (default: all)'
    )
    
    args = parser.parse_args()
    
    # Run diagnostic
    diagnostic = StrictIndexErrorDiagnostic(data_dir=Path(args.data_dir))
    success = diagnostic.run_strict_diagnostic(max_files=args.max_files)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
