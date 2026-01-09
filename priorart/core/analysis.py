#!/usr/bin/env python3
"""
PRIORART Redundancy Analyzer

Analyzes exhaustive run results to find redundant or near-identical
method combinations. Helps prune the method space to truly distinct approaches.

Ported from DIVERGE's battle-tested analyze_redundancy.py (4.28 TB proven).

NOTE: This module is READ-ONLY on the results dataset. It only writes
analysis outputs to the specified output directory.
"""

import numpy as np
from pathlib import Path
import argparse
import json
import gc
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import warnings


def parse_checksums(checksum_file: Path) -> Dict[str, str]:
    """Parse CHECKSUMS.txt into {filename: hash} dict."""
    checksums = {}
    with open(checksum_file, 'r') as fp:
        for line in fp:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('  ', 1)
            if len(parts) == 2:
                checksums[parts[1]] = parts[0]
    return checksums


def find_exact_duplicates(checksums: Dict[str, str]) -> Dict[str, List[str]]:
    """Find files with identical hashes.
    
    Returns: {hash: [list of filenames]} for hashes with 2+ files
    """
    hash_to_files = defaultdict(list)
    for filename, file_hash in checksums.items():
        hash_to_files[file_hash].append(filename)
    
    # Keep only duplicates
    return {h: files for h, files in hash_to_files.items() if len(files) > 1}


def parse_filename(filename: str) -> Dict[str, str]:
    """Parse method info from filename.
    
    Format: {decomposition}___{upsampling}.npy
    Can be adapted for other domain-specific formats.
    """
    name = filename.replace('.npy', '')
    parts = name.split('___')
    if len(parts) == 2:
        return {'stage1': parts[0], 'stage2': parts[1], 'full': name}
    return {'stage1': name, 'stage2': 'unknown', 'full': name}


def compute_fingerprint(arr: np.ndarray, max_sample_size: int = 1_000_000) -> np.ndarray:
    """Compute lightweight statistical fingerprint of array.
    
    Returns ~20-dimensional feature vector.
    
    For large arrays, uses sampling to reduce memory usage.
    """
    total_elements = arr.size
    use_sampling = total_elements > max_sample_size
    
    if use_sampling:
        # Random sampling for statistics (memory efficient)
        rng = np.random.default_rng(42)  # Reproducible
        flat_indices = rng.choice(total_elements, size=max_sample_size, replace=False)
        flat_sample = arr.flat[sorted(flat_indices)]
    else:
        flat_sample = arr.flatten()
    
    # Basic statistics
    stats = [
        float(np.mean(flat_sample)),
        float(np.std(flat_sample)),
        float(np.min(flat_sample)),
        float(np.max(flat_sample)),
        float(np.median(flat_sample)),
        float(np.percentile(flat_sample, 25)),
        float(np.percentile(flat_sample, 75)),
    ]
    
    # Histogram (10 bins, normalized)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist, _ = np.histogram(flat_sample, bins=10, density=True)
        if np.any(np.isnan(hist)):
            hist = np.zeros(10)
    
    # Clean up sample
    del flat_sample
    
    # Gradient magnitude (edge density) - use subsampling for large arrays
    if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1:
        if use_sampling:
            step = max(1, int(np.sqrt(total_elements / max_sample_size)))
            arr_sub = arr[::step, ::step]
        else:
            arr_sub = arr
        
        try:
            gy, gx = np.gradient(arr_sub.astype(np.float32))
            grad_mag = np.sqrt(gx**2 + gy**2)
            grad_stats = [float(np.mean(grad_mag)), float(np.std(grad_mag)), float(np.max(grad_mag))]
            del gy, gx, grad_mag
        except MemoryError:
            grad_stats = [float(np.std(arr_sub)), 0.0, 0.0]
        
        if use_sampling:
            del arr_sub
    else:
        grad_stats = [0.0, 0.0, 0.0]
    
    return np.array(stats + list(hist) + grad_stats)


def save_fingerprint_checkpoint(fingerprints: Dict[str, np.ndarray], checkpoint_file: Path):
    """Save fingerprints to checkpoint file."""
    serializable = {k: v.tolist() for k, v in fingerprints.items()}
    with open(checkpoint_file, 'w') as f:
        json.dump(serializable, f)


def load_fingerprint_checkpoint(checkpoint_file: Path) -> Dict[str, np.ndarray]:
    """Load fingerprints from checkpoint file."""
    if not checkpoint_file.exists():
        return {}
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        return {k: np.array(v) for k, v in data.items()}
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        return {}


def compute_all_fingerprints(
    results_dir: Path,
    file_list: Optional[List[str]] = None,
    limit: Optional[int] = None,
    checkpoint_file: Optional[Path] = None,
    gc_interval: int = 100
) -> Dict[str, np.ndarray]:
    """Compute fingerprints for all result files.
    
    Args:
        results_dir: Directory containing result .npy files
        file_list: Optional list of specific files to process
        limit: Optional limit on number of files
        checkpoint_file: Optional path to save/load intermediate results
        gc_interval: Run garbage collection every N files
    
    Returns: {filename: fingerprint_array}
    """
    if file_list is None:
        file_list = sorted(results_dir.glob('*.npy'))
    else:
        file_list = [results_dir / f for f in file_list]
    
    if limit:
        file_list = file_list[:limit]
    
    # Load existing checkpoint if available
    fingerprints = {}
    if checkpoint_file:
        fingerprints = load_fingerprint_checkpoint(checkpoint_file)
        print(f"Loaded {len(fingerprints)} fingerprints from checkpoint")
    
    # Filter out already processed files
    remaining = [f for f in file_list if f.name not in fingerprints]
    print(f"Processing {len(remaining)} files ({len(fingerprints)} already done)")
    
    processed_since_checkpoint = 0
    checkpoint_interval = 1000
    
    for i, fpath in enumerate(tqdm(remaining, desc="Computing fingerprints")):
        try:
            arr = np.load(fpath, mmap_mode='r')
            fingerprints[fpath.name] = compute_fingerprint(arr)
            
            del arr
            processed_since_checkpoint += 1
            
            if (i + 1) % gc_interval == 0:
                gc.collect()
            
            if checkpoint_file and processed_since_checkpoint >= checkpoint_interval:
                save_fingerprint_checkpoint(fingerprints, checkpoint_file)
                processed_since_checkpoint = 0
                
        except Exception as e:
            print(f"Warning: Could not process {fpath.name}: {e}")
    
    if checkpoint_file and processed_since_checkpoint > 0:
        save_fingerprint_checkpoint(fingerprints, checkpoint_file)
    
    gc.collect()
    return fingerprints


def fingerprint_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute cosine similarity between fingerprints."""
    norm1 = np.linalg.norm(fp1)
    norm2 = np.linalg.norm(fp2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(fp1, fp2) / (norm1 * norm2))


def find_near_duplicates(
    fingerprints: Dict[str, np.ndarray],
    threshold: float = 0.999
) -> List[Tuple[str, str, float]]:
    """Find pairs with very similar fingerprints.
    
    Returns list of (file1, file2, similarity) tuples.
    """
    names = list(fingerprints.keys())
    fps = np.array([fingerprints[n] for n in names])
    
    # Normalize for cosine similarity
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


def sample_correlation_analysis(
    results_dir: Path,
    pairs: List[Tuple[str, str, float]],
    n_samples: int = 100
) -> List[Tuple[str, str, float, str]]:
    """Compute actual array correlation for sampled pairs.
    
    Returns list of (file1, file2, correlation, status) tuples.
    """
    correlations = []
    sample_pairs = pairs[:n_samples] if len(pairs) > n_samples else pairs
    
    for f1, f2, _ in tqdm(sample_pairs, desc="Computing correlations"):
        try:
            arr1 = np.load(results_dir / f1, mmap_mode='r').flatten()
            arr2 = np.load(results_dir / f2, mmap_mode='r').flatten()
            
            if len(arr1) != len(arr2):
                correlations.append((f1, f2, 0.0, "size_mismatch"))
                continue
            
            corr = np.corrcoef(arr1, arr2)[0, 1]
            correlations.append((f1, f2, float(corr), "ok"))
        except Exception as e:
            correlations.append((f1, f2, 0.0, str(e)))
    
    return correlations


def analyze_parameter_sensitivity(fingerprints: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Analyze how much parameters affect output.
    
    Groups methods by base name and measures variation.
    """
    stage1_groups = defaultdict(list)
    stage2_groups = defaultdict(list)
    
    for filename, fp in fingerprints.items():
        info = parse_filename(filename)
        stage1_groups[info['stage1']].append((filename, fp))
        stage2_groups[info['stage2']].append((filename, fp))
    
    # Measure within-group variation
    stage1_sensitivity = {}
    for name, items in stage1_groups.items():
        if len(items) < 2:
            continue
        fps = np.array([fp for _, fp in items])
        mean_fp = np.mean(fps, axis=0)
        std_fp = np.std(fps, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = np.mean(std_fp / (np.abs(mean_fp) + 1e-10))
        stage1_sensitivity[name] = float(cv)
    
    stage2_sensitivity = {}
    for name, items in stage2_groups.items():
        if len(items) < 2:
            continue
        fps = np.array([fp for _, fp in items])
        mean_fp = np.mean(fps, axis=0)
        std_fp = np.std(fps, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = np.mean(std_fp / (np.abs(mean_fp) + 1e-10))
        stage2_sensitivity[name] = float(cv)
    
    return {
        'stage1': stage1_sensitivity,
        'stage2': stage2_sensitivity
    }


def cluster_methods(fingerprints: Dict[str, np.ndarray], n_clusters: int = 20) -> Dict[str, Any]:
    """Cluster methods by fingerprint similarity.
    
    Returns cluster assignments and representatives.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Warning: sklearn not available, skipping clustering")
        return {}
    
    names = list(fingerprints.keys())
    fps = np.array([fingerprints[n] for n in names])
    
    scaler = StandardScaler()
    fps_scaled = scaler.fit_transform(fps)
    
    n_clusters = min(n_clusters, len(names) // 2)
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(fps_scaled)
    
    clusters = defaultdict(list)
    for name, label in zip(names, labels):
        clusters[int(label)].append(name)
    
    representatives = {}
    for label, members in clusters.items():
        member_fps = np.array([fingerprints[m] for m in members])
        centroid = np.mean(member_fps, axis=0)
        distances = np.linalg.norm(member_fps - centroid, axis=1)
        rep_idx = np.argmin(distances)
        representatives[label] = members[rep_idx]
    
    return {
        'clusters': dict(clusters),
        'representatives': representatives,
        'n_clusters': n_clusters
    }


def generate_report(
    exact_dupes: Dict[str, List[str]],
    near_dupes: List[Tuple[str, str, float]],
    correlations: List[Tuple[str, str, float, str]],
    sensitivity: Dict[str, Dict[str, float]],
    clusters: Dict[str, Any],
    output_path: Path
):
    """Generate markdown redundancy report."""
    
    with open(output_path, 'w', encoding='utf-8') as fp:
        fp.write("# PRIORART Redundancy Analysis\n\n")
        fp.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        
        # Summary
        fp.write("## Summary\n\n")
        total_exact = sum(len(files) for files in exact_dupes.values())
        fp.write(f"- **Exact duplicate groups**: {len(exact_dupes)} ")
        fp.write(f"({total_exact} files)\n")
        fp.write(f"- **Near-duplicate pairs** (fingerprint similarity > 0.999): ")
        fp.write(f"{len(near_dupes)}\n")
        if clusters:
            fp.write(f"- **Method clusters**: {clusters.get('n_clusters', 'N/A')}\n")
        fp.write("\n")
        
        # Exact duplicates
        fp.write("## Exact Duplicate Groups\n\n")
        fp.write("These method combinations produce byte-for-byte identical outputs:\n\n")
        if exact_dupes:
            for i, (hash_val, files) in enumerate(list(exact_dupes.items())[:20]):
                fp.write(f"### Group {i+1} (hash: `{hash_val[:12]}...`)\n\n")
                for f in files[:10]:
                    info = parse_filename(f)
                    fp.write(f"- `{info['stage1']}` + `{info['stage2']}`\n")
                if len(files) > 10:
                    fp.write(f"- ... and {len(files) - 10} more\n")
                fp.write("\n")
            if len(exact_dupes) > 20:
                fp.write(f"*... and {len(exact_dupes) - 20} more groups*\n\n")
        else:
            fp.write("*No exact duplicates found.*\n\n")
        
        # Near duplicates
        fp.write("## Near-Duplicate Pairs\n\n")
        fp.write("These pairs have extremely similar statistical fingerprints:\n\n")
        if near_dupes:
            fp.write("| File 1 | File 2 | Similarity |\n")
            fp.write("|--------|--------|------------|\n")
            for f1, f2, sim in near_dupes[:30]:
                fp.write(f"| `{f1[:40]}...` | `{f2[:40]}...` | {sim:.4f} |\n")
            if len(near_dupes) > 30:
                fp.write(f"\n*... and {len(near_dupes) - 30} more pairs*\n")
        else:
            fp.write("*No near-duplicates found at threshold.*\n")
        fp.write("\n")
        
        # Correlation verification
        if correlations:
            fp.write("## Correlation Verification\n\n")
            fp.write("Actual array correlations for sampled near-duplicate pairs:\n\n")
            high_corr = [(f1, f2, c) for f1, f2, c, s in correlations if c > 0.99]
            fp.write(f"- **Pairs with r > 0.99**: {len(high_corr)}\n")
            med_corr = [(f1, f2, c) for f1, f2, c, s in correlations if 0.95 < c <= 0.99]
            fp.write(f"- **Pairs with 0.95 < r <= 0.99**: {len(med_corr)}\n\n")
        
        # Sensitivity
        fp.write("## Parameter Sensitivity\n\n")
        fp.write("Methods with LOW sensitivity produce similar outputs regardless of ")
        fp.write("other parameters (candidates for pruning).\n\n")
        
        if sensitivity.get('stage2'):
            fp.write("### Stage 2 Methods (by variation)\n\n")
            sorted_stage2 = sorted(sensitivity['stage2'].items(), key=lambda x: x[1])
            fp.write("| Method | Variation |\n")
            fp.write("|--------|----------|\n")
            for method, var in sorted_stage2[:15]:
                fp.write(f"| `{method}` | {var:.4f} |\n")
            fp.write("\n")
        
        # Clusters
        if clusters and clusters.get('clusters'):
            fp.write("## Method Clusters\n\n")
            fp.write("Methods grouped by fingerprint similarity:\n\n")
            for label, members in list(clusters['clusters'].items())[:10]:
                rep = clusters['representatives'].get(label, members[0])
                fp.write(f"### Cluster {label} ({len(members)} methods)\n")
                fp.write(f"- **Representative**: `{rep}`\n")
                fp.write(f"- **Members**: {', '.join(f'`{m[:30]}...`' for m in members[:5])}")
                if len(members) > 5:
                    fp.write(f" + {len(members) - 5} more")
                fp.write("\n\n")
        
        # Recommendations
        fp.write("## Recommendations\n\n")
        if exact_dupes:
            fp.write(f"1. **{total_exact - len(exact_dupes)} files can be pruned** ")
            fp.write("(exact duplicates - keep one per group)\n")
        if near_dupes:
            fp.write(f"2. **Review {len(near_dupes)} near-duplicate pairs** ")
            fp.write("for potential consolidation\n")
        if clusters and clusters.get('representatives'):
            fp.write(f"3. **{len(clusters['representatives'])} representative methods** ")
            fp.write("may capture most variation\n")
    
    print(f"Report saved to: {output_path}")


def save_json_results(
    exact_dupes: Dict[str, List[str]],
    near_dupes: List[Tuple[str, str, float]],
    correlations: List[Tuple[str, str, float, str]],
    sensitivity: Dict[str, Dict[str, float]],
    clusters: Dict[str, Any],
    output_path: Path
):
    """Save detailed results as JSON for further analysis."""
    
    results = {
        'generated': datetime.now().isoformat(),
        'exact_duplicates': {h: files for h, files in exact_dupes.items()},
        'near_duplicates': [
            {'file1': f1, 'file2': f2, 'similarity': sim}
            for f1, f2, sim in near_dupes
        ],
        'correlations': [
            {'file1': f1, 'file2': f2, 'correlation': c, 'status': s}
            for f1, f2, c, s in correlations
        ],
        'sensitivity': sensitivity,
        'clusters': {
            'assignments': clusters.get('clusters', {}),
            'representatives': {str(k): v for k, v in clusters.get('representatives', {}).items()}
        } if clusters else {}
    }
    
    with open(output_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    
    print(f"JSON results saved to: {output_path}")


def run_analysis(
    results_dir: Path,
    checksums_file: Optional[Path] = None,
    output_path: Optional[Path] = None,
    checkpoint_file: Optional[Path] = None,
    threshold: float = 0.999,
    fingerprint_limit: Optional[int] = None,
    correlation_samples: int = 100,
    skip_clustering: bool = False,
    fast: bool = False,
    gc_interval: int = 100
) -> Dict[str, Any]:
    """Run full redundancy analysis.
    
    This is the main entry point for programmatic use.
    
    Args:
        results_dir: Directory containing result .npy files (READ ONLY)
        checksums_file: Optional path to checksums file
        output_path: Optional path for markdown report
        checkpoint_file: Optional path for fingerprint checkpoint
        threshold: Near-duplicate similarity threshold
        fingerprint_limit: Limit fingerprint computation (for testing)
        correlation_samples: Number of pairs to correlate
        skip_clustering: Skip clustering analysis
        fast: Skip correlation sampling
        gc_interval: Garbage collection interval
        
    Returns:
        Dictionary with analysis results
    """
    print("=" * 60)
    print("PRIORART Redundancy Analyzer")
    print("=" * 60)
    print(f"Results directory: {results_dir} (READ ONLY)")
    if output_path:
        print(f"Output: {output_path}")
    print()
    
    # Stage 1: Exact duplicates from checksums
    checksums = {}
    exact_dupes = {}
    if checksums_file and checksums_file.exists():
        print("Stage 1: Analyzing checksums for exact duplicates...")
        checksums = parse_checksums(checksums_file)
        print(f"  Loaded {len(checksums)} checksums")
        exact_dupes = find_exact_duplicates(checksums)
        print(f"  Found {len(exact_dupes)} duplicate groups")
    else:
        print("Stage 1: Skipping (no checksums file)")
    
    # Stage 2: Statistical fingerprints
    print("\nStage 2: Computing statistical fingerprints...")
    file_list = list(checksums.keys()) if checksums else None
    fingerprints = compute_all_fingerprints(
        results_dir,
        file_list=file_list,
        limit=fingerprint_limit,
        checkpoint_file=checkpoint_file,
        gc_interval=gc_interval
    )
    print(f"  Computed {len(fingerprints)} fingerprints")
    
    # Stage 3: Find near-duplicates by fingerprint
    print("\nStage 3: Finding near-duplicates by fingerprint similarity...")
    near_dupes = find_near_duplicates(fingerprints, threshold=threshold)
    print(f"  Found {len(near_dupes)} near-duplicate pairs")
    
    # Stage 4: Correlation sampling (optional)
    correlations = []
    if not fast and near_dupes:
        print("\nStage 4: Sampling correlations for verification...")
        correlations = sample_correlation_analysis(
            results_dir,
            near_dupes,
            n_samples=correlation_samples
        )
        high_corr = sum(1 for _, _, c, _ in correlations if c > 0.99)
        print(f"  {high_corr}/{len(correlations)} pairs have r > 0.99")
    
    # Stage 5: Parameter sensitivity
    print("\nStage 5: Analyzing parameter sensitivity...")
    sensitivity = analyze_parameter_sensitivity(fingerprints)
    print(f"  Analyzed {len(sensitivity.get('stage1', {}))} stage1 methods")
    print(f"  Analyzed {len(sensitivity.get('stage2', {}))} stage2 methods")
    
    # Stage 6: Clustering (optional)
    clusters = {}
    if not skip_clustering and len(fingerprints) > 20:
        print("\nStage 6: Clustering methods...")
        clusters = cluster_methods(fingerprints)
        if clusters:
            print(f"  Created {clusters.get('n_clusters', 0)} clusters")
    
    # Generate outputs
    if output_path:
        print("\nGenerating reports...")
        generate_report(exact_dupes, near_dupes, correlations,
                        sensitivity, clusters, output_path)
        
        json_path = output_path.with_suffix('.json')
        save_json_results(exact_dupes, near_dupes, correlations,
                          sensitivity, clusters, json_path)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    return {
        'exact_duplicates': exact_dupes,
        'near_duplicates': near_dupes,
        'correlations': correlations,
        'sensitivity': sensitivity,
        'clusters': clusters,
        'fingerprints': fingerprints
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PRIORART Redundancy Analyzer'
    )
    parser.add_argument('--checksums', type=str,
                        help='Path to checksums file')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results directory (READ ONLY)')
    parser.add_argument('--output', '-o', type=str, default='REDUNDANCY_REPORT.md',
                        help='Output report path')
    parser.add_argument('--fast', action='store_true',
                        help='Skip correlation sampling (faster)')
    parser.add_argument('--fingerprint-limit', type=int, default=None,
                        help='Limit fingerprint computation (for testing)')
    parser.add_argument('--correlation-samples', type=int, default=100,
                        help='Number of pairs to correlate')
    parser.add_argument('--skip-clustering', action='store_true',
                        help='Skip clustering analysis')
    parser.add_argument('--threshold', type=float, default=0.999,
                        help='Fingerprint similarity threshold for near-duplicates')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint file for resumable fingerprinting')
    parser.add_argument('--gc-interval', type=int, default=100,
                        help='Run garbage collection every N files')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    checksums_file = Path(args.checksums) if args.checksums else None
    checkpoint_file = Path(args.checkpoint) if args.checkpoint else None
    
    run_analysis(
        results_dir=results_dir,
        checksums_file=checksums_file,
        output_path=output_path,
        checkpoint_file=checkpoint_file,
        threshold=args.threshold,
        fingerprint_limit=args.fingerprint_limit,
        correlation_samples=args.correlation_samples,
        skip_clustering=args.skip_clustering,
        fast=args.fast,
        gc_interval=args.gc_interval
    )


if __name__ == '__main__':
    main()
