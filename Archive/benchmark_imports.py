"""Benchmark import times for all modules in the shelterbelts package."""

import os
import sys
import time
import importlib
import subprocess
from pathlib import Path
import pandas as pd

def time_import(module_path, item_to_import=None):
    """Time how long it takes to import a module or item.
    
    Uses subprocess to get clean import without caching.
    """
    if item_to_import:
        cmd = f"from {module_path} import {item_to_import}"
    else:
        cmd = f"import {module_path}"
    
    python_cmd = [
        sys.executable,
        "-c",
        f"import time; start=time.perf_counter(); {cmd}; print(time.perf_counter()-start)"
    ]
    
    try:
        result = subprocess.run(python_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            return None
    except (subprocess.TimeoutExpired, ValueError):
        return None


def find_all_modules(base_path):
    """Find all Python modules in the package."""
    modules = []
    base_path = Path(base_path)
    
    for py_file in base_path.rglob("*.py"):
        if py_file.name == "__init__.py" or py_file.name.startswith("test_"):
            continue
        
        # Convert file path to module path
        rel_path = py_file.relative_to(base_path.parent)
        module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        module_path = ".".join(module_parts)
        
        modules.append({
            'file': str(py_file.relative_to(base_path.parent.parent)),
            'module': module_path
        })
    
    return modules


def main():
    """Run the benchmark and save results."""
    repo_root = Path(__file__).parent.parent
    src_path = repo_root / "src" / "shelterbelts"
    outdir = repo_root / "outdir"
    outdir.mkdir(exist_ok=True)
    
    print("Finding all modules...")
    modules = find_all_modules(src_path)
    print(f"Found {len(modules)} modules")
    
    results = []
    
    for i, mod_info in enumerate(modules, 1):
        module_path = mod_info['module']
        file_path = mod_info['file']
        
        print(f"[{i}/{len(modules)}] Timing {module_path}...", end=" ", flush=True)
        
        import_time = time_import(module_path)
        
        if import_time is not None:
            print(f"{import_time:.4f}s")
            results.append({
                'module': module_path,
                'file': file_path,
                'import_time_seconds': import_time,
                'status': 'success'
            })
        else:
            print("FAILED")
            results.append({
                'module': module_path,
                'file': file_path,
                'import_time_seconds': None,
                'status': 'failed'
            })
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    df = df.sort_values('import_time_seconds', ascending=False, na_position='last')
    
    output_file = outdir / "import_benchmark_results.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\nTop 10 slowest imports:")
    print(df.head(10)[['module', 'import_time_seconds']].to_string(index=False))
    
    print(f"\nSummary:")
    print(f"  Total modules: {len(results)}")
    print(f"  Successful: {df['status'].eq('success').sum()}")
    print(f"  Failed: {df['status'].eq('failed').sum()}")
    print(f"  Mean import time: {df['import_time_seconds'].mean():.4f}s")
    print(f"  Median import time: {df['import_time_seconds'].median():.4f}s")
    print(f"  Max import time: {df['import_time_seconds'].max():.4f}s")


if __name__ == "__main__":
    main()
