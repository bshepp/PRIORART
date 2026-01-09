#!/usr/bin/env python3
"""
PRIORART Checksum Generator

Generates SHA256 checksums for all output files, providing
cryptographic proof of existence for prior art dating.
"""

import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import argparse
from tqdm import tqdm


def compute_file_checksum(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_existing_checksums(checksum_file: Path) -> Dict[str, str]:
    """Load existing checksums from file."""
    checksums = {}
    if checksum_file.exists():
        with open(checksum_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('  ', 1)
                if len(parts) == 2:
                    checksums[parts[1]] = parts[0]
    return checksums


def generate_checksums(
    results_dir: Path,
    output_file: Path,
    skip_existing: bool = True,
    pattern: str = "*.npy"
) -> Dict[str, int]:
    """
    Generate checksums for all result files.
    
    Args:
        results_dir: Directory containing result files
        output_file: Output checksums file
        skip_existing: Skip files already in checksums
        pattern: Glob pattern for files to checksum
        
    Returns:
        Statistics dict
    """
    results_dir = Path(results_dir)
    output_file = Path(output_file)
    
    # Find all files
    files = sorted(results_dir.glob(pattern))
    
    # Load existing
    existing = {}
    if skip_existing and output_file.exists():
        existing = load_existing_checksums(output_file)
    
    stats = {'total': len(files), 'computed': 0, 'skipped': 0}
    
    # Open for append or write
    mode = 'a' if skip_existing and existing else 'w'
    
    with open(output_file, mode) as f:
        if mode == 'w':
            f.write(f"# PRIORART Checksums\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Format: SHA256  filename\n\n")
        
        for filepath in tqdm(files, desc="Computing checksums"):
            filename = filepath.name
            
            if skip_existing and filename in existing:
                stats['skipped'] += 1
                continue
            
            checksum = compute_file_checksum(filepath)
            f.write(f"{checksum}  {filename}\n")
            stats['computed'] += 1
    
    return stats


def verify_checksums(
    results_dir: Path,
    checksum_file: Path
) -> Dict[str, list]:
    """
    Verify files against stored checksums.
    
    Returns:
        Dict with 'valid', 'invalid', 'missing' lists
    """
    results_dir = Path(results_dir)
    checksums = load_existing_checksums(checksum_file)
    
    results = {'valid': [], 'invalid': [], 'missing': []}
    
    for filename, expected_hash in tqdm(checksums.items(), desc="Verifying"):
        filepath = results_dir / filename
        
        if not filepath.exists():
            results['missing'].append(filename)
            continue
        
        actual_hash = compute_file_checksum(filepath)
        
        if actual_hash == expected_hash:
            results['valid'].append(filename)
        else:
            results['invalid'].append(filename)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='PRIORART Checksum Generator')
    parser.add_argument('results_dir', type=str, help='Results directory')
    parser.add_argument('--output', '-o', type=str, default='CHECKSUMS.txt',
                        help='Output checksums file')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip files already checksummed')
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing checksums instead of generating')
    parser.add_argument('--pattern', type=str, default='*.npy',
                        help='File pattern to match')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_file = Path(args.output)
    
    if args.verify:
        results = verify_checksums(results_dir, output_file)
        print(f"Valid: {len(results['valid'])}")
        print(f"Invalid: {len(results['invalid'])}")
        print(f"Missing: {len(results['missing'])}")
    else:
        stats = generate_checksums(
            results_dir, output_file,
            skip_existing=args.skip_existing,
            pattern=args.pattern
        )
        print(f"Total files: {stats['total']}")
        print(f"Computed: {stats['computed']}")
        print(f"Skipped: {stats['skipped']}")


if __name__ == '__main__':
    main()

