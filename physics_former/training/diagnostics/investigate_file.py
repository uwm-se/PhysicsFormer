"""Investigate a specific JSON file for corruption or format issues"""
import json
import sys
from pathlib import Path

def investigate_file(file_path):
    """Deep investigation of a JSON file"""
    file_path = Path(file_path)
    
    print(f"\n{'='*80}")
    print(f"INVESTIGATING: {file_path.name}")
    print(f"{'='*80}\n")
    
    # 1. File size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    size_gb = size_mb / 1024
    print(f"📊 File Size: {size_mb:.2f} MB ({size_gb:.2f} GB)")
    
    # 2. Check if it's valid JSON at all
    print(f"\n🔍 Testing JSON validity...")
    try:
        with open(file_path, 'r') as f:
            # Read first 10KB
            sample = f.read(10 * 1024)
            print(f"   First 10KB preview:")
            print(f"   {sample[:500]}...")
            
            # Check structure
            if sample.strip().startswith('['):
                print(f"   PASS: Starts with '[' (JSON array)")
            elif sample.strip().startswith('{'):
                print(f"   PASS: Starts with '{{' (JSON object)")
            else:
                print(f"   FAIL: Does not start with '[' or '{{'")
                print(f"   First char: {repr(sample[0])}")
    except Exception as e:
        print(f"   FAIL: Error reading file: {e}")
        return
    
    # 3. Try to parse as JSON (streaming)
    print(f"\n🔍 Attempting to parse JSON structure...")
    try:
        with open(file_path, 'r') as f:
            # Try to load the entire thing (risky for large files)
            print(f"   WARNING:  This may take a while for large files...")
            
            # Read in chunks and look for structure
            chunk_size = 1024 * 1024  # 1MB chunks
            chunk_count = 0
            schema_count = 0
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                chunk_count += 1
                schema_count += chunk.count('"schema":')
                
                if chunk_count == 1:
                    # Analyze first chunk
                    print("\n   First chunk analysis:")
                    print(f'   - Contains \'schema\': {chr(34) + "schema" + chr(34) + ":" in chunk}')
                    print(f'   - Contains \'object_states\': {chr(34) + "object_states" + chr(34) + ":" in chunk}')
                    print(f'   - Contains \'episode_id\': {chr(34) + "episode_id" + chr(34) + ":" in chunk}')
                    print(f'   - Schema count in first chunk: {chunk.count(chr(34) + "schema" + chr(34) + ":")} ')
                
                if chunk_count % 100 == 0:
                    print(f"   Processed {chunk_count} MB, found {schema_count} episodes so far...")
                
                # Stop after 500MB to avoid hanging
                if chunk_count >= 500:
                    print(f"   WARNING:  Stopping after 500MB to avoid hanging")
                    break
            
            print(f"\n   PASS: Processed {chunk_count} MB")
            print(f"   PASS: Found {schema_count} episodes (estimated)")
            
    except json.JSONDecodeError as e:
        print(f"   FAIL: JSON parsing error: {e}")
        print(f"   Location: line {e.lineno}, column {e.colno}")
    except Exception as e:
        print(f"   FAIL: Error: {e}")
    
    # 4. Check for expected format
    print(f"\n🔍 Checking expected format...")
    try:
        with open(file_path, 'r') as f:
            # Read first episode
            content = f.read(50000)  # 50KB should contain first episode
            
            # Try to find first complete episode
            if '"schema":' in content:
                # Find the schema value
                import re
                schema_match = re.search(r'"schema":\s*"([^"]+)"', content)
                if schema_match:
                    schema_value = schema_match.group(1)
                    print(f"   PASS: Found schema: '{schema_value}'")
                else:
                    print(f"   FAIL: Schema field exists but couldn't extract value")
            else:
                print(f"   FAIL: No 'schema' field found in first 50KB")
            
            if '"object_states":' in content:
                print(f"   PASS: Found 'object_states' field")
            else:
                print(f"   FAIL: No 'object_states' field found")
            
            if '"episode_id":' in content:
                print(f"   PASS: Found 'episode_id' field")
            else:
                print(f"   WARNING:  No 'episode_id' field found (may be optional)")
                
    except Exception as e:
        print(f"   FAIL: Error checking format: {e}")
    
    # 5. Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"File: {file_path.name}")
    print(f"Size: {size_gb:.2f} GB")
    print(f"Estimated episodes: {schema_count}")
    print(f"\nRecommendation:")
    if schema_count == 0:
        print(f"  FAIL: File appears CORRUPTED or INCOMPLETE")
        print(f"  -> Regenerate this schema")
    elif schema_count < 10000:
        print(f"  WARNING:  File has fewer episodes than expected (target: 50,000)")
        print(f"  -> May need to regenerate")
    else:
        print(f"  PASS: File appears valid")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python investigate_file.py <path_to_json_file>")
        sys.exit(1)
    
    investigate_file(sys.argv[1])
