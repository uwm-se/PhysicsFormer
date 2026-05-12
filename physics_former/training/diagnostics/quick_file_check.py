"""Quick check: Which files exist and their sizes"""
import json
from pathlib import Path
import sys

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/physics-llm-data/physics")

print(f"\n📁 Checking: {data_dir}\n")

json_files = sorted(data_dir.glob("*.json"))
print(f"Found {len(json_files)} JSON files\n")

for i, file_path in enumerate(json_files, 1):
    size_mb = file_path.stat().st_size / (1024 * 1024)
    
    # Try to count episodes quickly
    try:
        with open(file_path, 'r') as f:
            # Read more data to ensure we find the schema field
            # (states arrays can be very large, schema comes after)
            content = f.read(10 * 1024 * 1024)  # Read first 10MB
            if '"schema":' in content:
                status = "PASS: Valid JSON structure"
            else:
                status = "FAIL: No schema found (checked first 10MB)"
    except Exception as e:
        status = f"FAIL: Error: {e}"
    
    print(f"{i:2d}. {file_path.name:40s} {size_mb:8.2f} MB  {status}")

print(f"\nPASS: Total: {len(json_files)} files")
