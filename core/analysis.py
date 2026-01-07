#!/usr/bin/env python3
"""
PRIORART Redundancy Analyzer

Analyzes exhaustive run results to find redundant or near-identical
method combinations. Helps identify which methods produce unique outputs.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json
import argparse
from tqdm import tqdm
import warnings


def parse_checksums(checksum_file: Path) -> Dict[str, str]:
    """Parse checksums file into {filename: hash} dict."""
    checksums = {}
    with open(checksum_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('  ', 1)
            if len(parts) == 2:
                checksums[parts[1]] = parts[0]
    return checksums


def find_exact_duplicates(checksums: Dict[str, str]) -> Dict[str, List[str]]:
    """Find files with identical hashes."""
    hash_to_files = defaultdict(list)
    for filename, file_hash in checksums.items():
        hash_to_files[file_hash].append(filename)
    
    return {h: files for h, files in hash_to_files.items() if len(files) > 1}


def compute_fingerprint(arr: np.ndarray) -> np.ndarray:
    """Compute statistical fingerprint of array (~20 dimensions)."""
    flat = arr.flatten()
    
    stats = [
        np.mean(flat),
        np.std(flat),
        np.min(flat),
        np.max(flat),
        np.median(flat),
        np.percentile(flat, 25),
        np.percentile(flat, 75),
    ]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist, _ = np.histogram(flat, bins=10, density=True)
        if np.any(np.isnan(hist)):
            hist = np.zeros(10)
    
    if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1:
        gy, gx = np.gradient(arr)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_stats = [np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag)]
    else:
        grad_stats = [0, 0, 0]
    
    return np.array(stats + list(hist) + grad_stats)


def compute_all_fingerprints(
    results_dir: Path,
    file_list: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Compute fingerprints for result files."""
    if file_list is None:
        files = sorted(results_dir.glob('*.npy'))
    else:
        files = [results_dir / f for f in file_list]
    
    if limit:
        files = files[:limit]
    
    fingerprints = {}
    for fpath in tqdm(files, desc="Computing fingerprints"):
        try:
            arr = np.load(fpath, mmap_mode='r')
            fingerprints[fpath.name] = compute_fingerprint(arr)
        except Exception as e:
            print(f"Warning: Could not process {fpath.name}: {e}")
    
    return fingerprints


def find_near_duplicates(
    fingerprints: Dict[str, np.ndarray],
    threshold: float = 0.999
) -> List[Tuple[str, str, float]]:
    """Find pairs with similar fingerprints (cosine similarity)."""
    names = list(fingerprints.keys())
    fps = np.array([fingerprints[n] for n in names])
    
    # Normalize
    norms = np.linalg.norm(fps, axis=1, keepdims=True)
    norms[norms == 0] = 1
    fps_norm = fps / norms
    
    near_dupes = []
    chunk_size = 1000
    
    for i in tqdm(range(0, len(names), chunk_size), desc="Finding near-duplicates"):
        chunk_end = min(i + chunk_size, len(names))
        chunk = fps_norm[i:chunk_end]
        
        for j in range(i, len(names), chunk_size):
            j_end = min(j + chunk_size, len(names))
            other = fps_norm[j:j_end]
            
            sims = chunk @ other.T
            
            for ci in range(chunk.shape[0]):
                for cj in range(other.shape[0]):
                    global_i = i + ci
                    global_j = j + cj
                    if global_i >= global_j:
                        continue
                    if sims[ci, cj] >= threshold:
                        near_dupes.append((
                            names[global_i],
                            names[global_j],
                            float(sims[ci, cj])
                        ))
    
    return sorted(near_dupes, key=lambda x: -x[2])


def generate_report(
    exact_dupes: Dict[str, List[str]],
    near_dupes: List[Tuple[str, str, float]],
    output_path: Path
):
    """Generate markdown redundancy report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# PRIORART Redundancy Analysis\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        
        # Summary
        total_exact = sum(len(files) for files in exact_dupes.values())
        f.write("## Summary\n\n")
        f.write(f"- **Exact duplicate groups**: {len(exact_dupes)} ({total_exact} files)\n")
        f.write(f"- **Near-duplicate pairs**: {len(near_dupes)}\n\n")
        
        # Exact duplicates
        f.write("## Exact Duplicates\n\n")
        if exact_dupes:
            for i, (hash_val, files) in enumerate(list(exact_dupes.items())[:20]):
                f.write(f"### Group {i+1} (hash: `{hash_val[:12]}...`)\n\n")
                for fname in files[:10]:
                    f.write(f"- `{fname}`\n")
                if len(files) > 10:
                    f.write(f"- ... and {len(files) - 10} more\n")
                f.write("\n")
        else:
            f.write("*No exact duplicates found.*\n\n")
        
        # Near duplicates
        f.write("## Near-Duplicates\n\n")
        if near_dupes:
            f.write("| File 1 | File 2 | Similarity |\n")
            f.write("|--------|--------|------------|\n")
            for f1, f2, sim in near_dupes[:30]:
                f.write(f"| `{f1[:35]}...` | `{f2[:35]}...` | {sim:.4f} |\n")
        else:
            f.write("*No near-duplicates found at threshold.*\n")
    
    print(f"Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='PRIORART Redundancy Analyzer')
    parser.add_argument('--checksums', type=str, help='Checksums file')
    parser.add_argument('--results', type=str, required=True, help='Results directory')
    parser.add_argument('--output', '-o', type=str, default='REDUNDANCY_REPORT.md',
                        help='Output report path')
    parser.add_argument('--threshold', type=float, default=0.999,
                        help='Near-duplicate similarity threshold')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of files to analyze')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    output_path = Path(args.output)
    
    # Exact duplicates from checksums
    exact_dupes = {}
    if args.checksums:
        checksums = parse_checksums(Path(args.checksums))
        exact_dupes = find_exact_duplicates(checksums)
        print(f"Found {len(exact_dupes)} exact duplicate groups")
    
    # Fingerprint analysis
    fingerprints = compute_all_fingerprints(results_dir, limit=args.limit)
    print(f"Computed {len(fingerprints)} fingerprints")
    
    near_dupes = find_near_duplicates(fingerprints, threshold=args.threshold)
    print(f"Found {len(near_dupes)} near-duplicate pairs")
    
    # Generate report
    generate_report(exact_dupes, near_dupes, output_path)


if __name__ == '__main__':
    main()

