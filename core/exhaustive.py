#!/usr/bin/env python3
"""
PRIORART Exhaustive Runner

Systematically explores all combinations of decomposition and upsampling
methods with their parameter ranges, generating prior art documentation.

Domain-agnostic: works on any 2D scalar field.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import Dict, List, Any, Optional, Tuple
import json
import argparse
from tqdm import tqdm

from .decomposition import (
    DECOMPOSITION_REGISTRY,
    run_decomposition,
    get_all_methods_info as get_decomp_info
)
from .upsampling import (
    UPSAMPLING_REGISTRY,
    run_upsampling,
    get_all_methods_info as get_upsamp_info
)


def generate_param_combinations(param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from ranges."""
    if not param_ranges:
        return [{}]
    
    keys = list(param_ranges.keys())
    values = [param_ranges[k] for k in keys]
    
    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def generate_all_combinations() -> List[Dict[str, Any]]:
    """Generate all method + parameter combinations."""
    combinations = []
    
    for decomp_name, decomp_method in DECOMPOSITION_REGISTRY.items():
        decomp_params_list = generate_param_combinations(decomp_method.param_ranges)
        
        for upsamp_name, upsamp_method in UPSAMPLING_REGISTRY.items():
            upsamp_params_list = generate_param_combinations(upsamp_method.param_ranges)
            
            for decomp_params in decomp_params_list:
                for upsamp_params in upsamp_params_list:
                    combo_id = format_combo_id(
                        decomp_name, decomp_params,
                        upsamp_name, upsamp_params
                    )
                    combinations.append({
                        'id': combo_id,
                        'decomposition': {
                            'name': decomp_name,
                            'params': decomp_params
                        },
                        'upsampling': {
                            'name': upsamp_name,
                            'params': upsamp_params
                        }
                    })
    
    return combinations


def format_combo_id(
    decomp_name: str,
    decomp_params: Dict[str, Any],
    upsamp_name: str,
    upsamp_params: Dict[str, Any]
) -> str:
    """Format a unique identifier for a method combination."""
    decomp_str = decomp_name
    if decomp_params:
        param_str = '_'.join(f"{k}{v}" for k, v in sorted(decomp_params.items()))
        decomp_str = f"{decomp_name}_{param_str}"
    
    upsamp_str = upsamp_name
    if upsamp_params:
        param_str = '_'.join(f"{k}{v}" for k, v in sorted(upsamp_params.items()))
        upsamp_str = f"{upsamp_name}_{param_str}"
    
    return f"{decomp_str}___{upsamp_str}"


def run_combination(
    data: np.ndarray,
    decomp_name: str,
    decomp_params: Dict[str, Any],
    upsamp_name: str,
    upsamp_params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a single decomposition + upsampling combination.
    
    Returns:
        (upsampled_trend, upsampled_residual) tuple
    """
    # Decompose
    trend, residual = run_decomposition(decomp_name, data, decomp_params)
    
    # Get scale from upsampling params
    scale = upsamp_params.get('scale', 2)
    
    # Upsample both components
    trend_up = run_upsampling(upsamp_name, trend, scale, upsamp_params)
    residual_up = run_upsampling(upsamp_name, residual, scale, upsamp_params)
    
    return trend_up, residual_up


def run_exhaustive(
    data: np.ndarray,
    output_dir: Path,
    skip_existing: bool = True,
    save_trend: bool = False,
    combinations: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Run exhaustive exploration of all method combinations.
    
    Args:
        data: Input 2D array
        output_dir: Directory to save results
        skip_existing: Skip combinations that already have output files
        save_trend: Also save trend component (doubles storage)
        combinations: Specific combinations to run (None = all)
        
    Returns:
        Summary statistics
    """
    output_dir = Path(output_dir)
    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if combinations is None:
        combinations = generate_all_combinations()
    
    stats = {
        'total': len(combinations),
        'completed': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }
    
    pbar = tqdm(combinations, desc="Running combinations")
    
    for combo in pbar:
        combo_id = combo['id']
        result_path = results_dir / f"{combo_id}.npy"
        
        # Skip if exists
        if skip_existing and result_path.exists():
            stats['skipped'] += 1
            pbar.set_postfix(done=stats['completed'], skip=stats['skipped'])
            continue
        
        try:
            trend_up, residual_up = run_combination(
                data,
                combo['decomposition']['name'],
                combo['decomposition']['params'],
                combo['upsampling']['name'],
                combo['upsampling']['params']
            )
            
            # Save residual (primary output)
            np.save(result_path, residual_up.astype(np.float32))
            
            # Optionally save trend
            if save_trend:
                trend_path = results_dir / f"{combo_id}_trend.npy"
                np.save(trend_path, trend_up.astype(np.float32))
            
            stats['completed'] += 1
            
        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append({
                'combo_id': combo_id,
                'error': str(e)
            })
        
        pbar.set_postfix(done=stats['completed'], skip=stats['skipped'], fail=stats['failed'])
    
    return stats


def generate_prior_art_doc(
    output_dir: Path,
    domain: str = "2D Scalar Fields"
) -> Path:
    """Generate prior art documentation markdown."""
    output_dir = Path(output_dir)
    doc_dir = output_dir / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_path = doc_dir / f"PRIOR_ART_{timestamp}.md"
    
    combinations = generate_all_combinations()
    decomp_info = get_decomp_info()
    upsamp_info = get_upsamp_info()
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(f"# PRIORART: {domain} Prior Art Documentation\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        f.write("## Purpose\n\n")
        f.write("This document establishes prior art for combinations of signal\n")
        f.write("decomposition and upsampling methods applied to 2D scalar fields.\n")
        f.write("Published under Apache 2.0 license with explicit patent grant.\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Combinations**: {len(combinations)}\n")
        f.write(f"- **Decomposition Methods**: {len(decomp_info)}\n")
        f.write(f"- **Upsampling Methods**: {len(upsamp_info)}\n\n")
        
        # Decomposition methods
        f.write("## Decomposition Methods\n\n")
        for name, info in sorted(decomp_info.items()):
            f.write(f"### {name}\n\n")
            f.write(f"- **Category**: {info['category']}\n")
            f.write(f"- **Preserves**: {info['preserves']}\n")
            f.write(f"- **Destroys**: {info['destroys']}\n")
            if info['param_ranges']:
                f.write(f"- **Parameters**: {info['param_ranges']}\n")
            f.write("\n")
        
        # Upsampling methods
        f.write("## Upsampling Methods\n\n")
        for name, info in sorted(upsamp_info.items()):
            f.write(f"### {name}\n\n")
            f.write(f"- **Category**: {info['category']}\n")
            f.write(f"- **Preserves**: {info['preserves']}\n")
            f.write(f"- **Introduces**: {info['introduces']}\n")
            if info['param_ranges']:
                f.write(f"- **Parameters**: {info['param_ranges']}\n")
            f.write("\n")
        
        # All combinations
        f.write("## All Combinations\n\n")
        f.write("Each combination below represents a distinct processing pipeline:\n\n")
        
        for i, combo in enumerate(combinations, 1):
            f.write(f"{i}. `{combo['id']}`\n")
            if i % 100 == 0:
                f.write("\n")
        
        f.write("\n---\n")
        f.write(f"\n*Document generated by PRIORART v0.1.0*\n")
    
    return doc_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PRIORART Exhaustive Runner - Generate prior art'
    )
    parser.add_argument('input', type=str, help='Input .npy file (2D array)')
    parser.add_argument('--output', '-o', type=str, default='./priorart_results',
                        help='Output directory')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip combinations with existing output')
    parser.add_argument('--save-trend', action='store_true',
                        help='Also save trend components')
    parser.add_argument('--doc-only', action='store_true',
                        help='Only generate documentation, no processing')
    parser.add_argument('--domain', type=str, default='2D Scalar Fields',
                        help='Domain name for documentation')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate documentation
    doc_path = generate_prior_art_doc(output_dir, args.domain)
    print(f"Documentation: {doc_path}")
    
    if args.doc_only:
        return
    
    # Load input data
    data = np.load(args.input)
    print(f"Input shape: {data.shape}")
    
    # Count combinations
    combinations = generate_all_combinations()
    print(f"Total combinations: {len(combinations)}")
    
    # Run exhaustive
    stats = run_exhaustive(
        data,
        output_dir,
        skip_existing=args.skip_existing,
        save_trend=args.save_trend
    )
    
    print(f"\nCompleted: {stats['completed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    
    # Save stats
    stats_path = output_dir / 'run_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == '__main__':
    main()

