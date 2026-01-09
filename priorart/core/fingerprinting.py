#!/usr/bin/env python3
"""
PRIORART Parallel Fingerprint Generator

Uses ProcessPoolExecutor with configurable workers to fingerprint
all output files for redundancy analysis.

Supports checkpointing - can be interrupted and resumed.

Ported from DIVERGE's battle-tested fingerprint_all_parallel.py (39,731 files proven).
"""

import numpy as np
from pathlib import Path
import json
import warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import argparse


# Default configuration
DEFAULT_NUM_WORKERS = 2  # Optimized for HDD
DEFAULT_CHECKPOINT_INTERVAL = 500  # Save every N files


def compute_fingerprint(arr: np.ndarray, max_sample_size: int = 1_000_000) -> np.ndarray:
    """Compute lightweight statistical fingerprint of array.
    
    Returns ~20-dimensional feature vector.
    For large arrays, uses sampling to reduce memory usage.
    
    Args:
        arr: Input array (typically 2D)
        max_sample_size: Maximum elements to sample for statistics
        
    Returns:
        numpy array with 20 features:
        - 7 basic stats (mean, std, min, max, median, p25, p75)
        - 10 histogram bins (normalized density)
        - 3 gradient stats (mean, std, max of gradient magnitude)
    """
    total_elements = arr.size
    use_sampling = total_elements > max_sample_size
    
    if use_sampling:
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
    
    del flat_sample
    
    # Gradient magnitude (edge density)
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


def process_file(filepath: Path) -> Tuple[str, Optional[List[float]], Optional[str]]:
    """Process a single file and return (filename, fingerprint, error).
    
    This function runs in a worker process.
    
    Returns:
        Tuple of (filename, fingerprint_list, error_message)
        fingerprint_list is None if error, error_message is None if success
    """
    try:
        arr = np.load(filepath, mmap_mode='r')
        fp = compute_fingerprint(arr)
        del arr
        return (filepath.name, fp.tolist(), None)
    except Exception as e:
        return (filepath.name, None, str(e))


def load_checkpoint(checkpoint_path: Path) -> Dict[str, List[float]]:
    """Load existing fingerprints from checkpoint.
    
    Args:
        checkpoint_path: Path to JSON checkpoint file
        
    Returns:
        Dict mapping filename to fingerprint list
    """
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return {}


def save_checkpoint(fingerprints: Dict[str, List[float]], checkpoint_path: Path):
    """Save fingerprints to checkpoint file.
    
    Args:
        fingerprints: Dict mapping filename to fingerprint list
        checkpoint_path: Path to save JSON checkpoint
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(fingerprints, f)


def run_parallel_fingerprinting(
    results_dir: Path,
    checkpoint_path: Path,
    num_workers: int = DEFAULT_NUM_WORKERS,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    file_list: Optional[List[Path]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run parallel fingerprinting on all .npy files in a directory.
    
    This is the main entry point for programmatic use.
    
    Args:
        results_dir: Directory containing .npy files
        checkpoint_path: Path for checkpoint file
        num_workers: Number of parallel workers
        checkpoint_interval: Save checkpoint every N files
        file_list: Optional list of specific files to process
        verbose: Print progress information
        
    Returns:
        Dict with 'fingerprints', 'success_count', 'failed_count'
    """
    if verbose:
        print(f"{'='*60}")
        print(f"PRIORART Parallel Fingerprint Generator")
        print(f"{'='*60}")
        print(f"Source: {results_dir}")
        print(f"Workers: {num_workers}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Started: {datetime.now().isoformat()}")
        print()
    
    # Load existing fingerprints
    fingerprints = load_checkpoint(checkpoint_path)
    if verbose:
        print(f"Loaded {len(fingerprints):,} existing fingerprints from checkpoint")
    
    # Get all files
    if file_list is None:
        all_files = sorted(results_dir.glob('*.npy'))
    else:
        all_files = file_list
    
    if verbose:
        print(f"Total files in source: {len(all_files):,}")
    
    # Filter to only unprocessed files
    files_to_process = [f for f in all_files if f.name not in fingerprints]
    
    if verbose:
        print(f"Files needing fingerprints: {len(files_to_process):,}")
    
    if not files_to_process:
        if verbose:
            print("\nAll files already fingerprinted!")
        return {
            'fingerprints': fingerprints,
            'success_count': 0,
            'failed_count': 0,
            'already_done': len(fingerprints)
        }
    
    if verbose:
        print(f"\nProcessing {len(files_to_process):,} files with {num_workers} workers...")
        print()
    
    success = 0
    failed = 0
    processed_since_checkpoint = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file, f): f for f in files_to_process}
        
        # Process as they complete
        pbar = tqdm(total=len(files_to_process), desc="Fingerprinting", disable=not verbose)
        
        for future in as_completed(future_to_file):
            filename, fp, error = future.result()
            
            if error:
                failed += 1
                if verbose:
                    tqdm.write(f"Failed: {filename}: {error}")
            else:
                fingerprints[filename] = fp
                success += 1
            
            processed_since_checkpoint += 1
            pbar.update(1)
            
            # Periodic checkpoint
            if processed_since_checkpoint >= checkpoint_interval:
                save_checkpoint(fingerprints, checkpoint_path)
                if verbose:
                    tqdm.write(f"Checkpoint: {len(fingerprints):,} fingerprints saved")
                processed_since_checkpoint = 0
        
        pbar.close()
    
    # Final save
    save_checkpoint(fingerprints, checkpoint_path)
    
    if verbose:
        print()
        print(f"{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Success: {success:,}")
        print(f"Failed:  {failed}")
        print(f"Total fingerprints: {len(fingerprints):,}")
        print(f"Saved to: {checkpoint_path}")
        print(f"Finished: {datetime.now().isoformat()}")
    
    return {
        'fingerprints': fingerprints,
        'success_count': success,
        'failed_count': failed,
        'already_done': len(fingerprints) - success
    }


def fingerprints_to_numpy(fingerprints: Dict[str, List[float]]) -> Tuple[List[str], np.ndarray]:
    """Convert fingerprints dict to numpy array.
    
    Args:
        fingerprints: Dict mapping filename to fingerprint list
        
    Returns:
        Tuple of (list of filenames, 2D numpy array of fingerprints)
    """
    names = list(fingerprints.keys())
    fps = np.array([fingerprints[n] for n in names])
    return names, fps


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='PRIORART Parallel Fingerprint Generator')
    parser.add_argument('--workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help=f'Number of parallel workers (default: {DEFAULT_NUM_WORKERS})')
    parser.add_argument('--source', type=str, required=True,
                        help='Source directory with .npy files')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file path')
    parser.add_argument('--interval', type=int, default=DEFAULT_CHECKPOINT_INTERVAL,
                        help=f'Checkpoint save interval (default: {DEFAULT_CHECKPOINT_INTERVAL})')
    args = parser.parse_args()
    
    results_dir = Path(args.source)
    checkpoint_path = Path(args.checkpoint)
    
    run_parallel_fingerprinting(
        results_dir=results_dir,
        checkpoint_path=checkpoint_path,
        num_workers=args.workers,
        checkpoint_interval=args.interval
    )


if __name__ == '__main__':
    main()
