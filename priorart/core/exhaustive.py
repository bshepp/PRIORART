#!/usr/bin/env python3
"""
PRIORART Exhaustive Runner

Systematically explores all combinations of methods from any registered
domain pipeline, generating prior art documentation.

Refactored to use plugin architecture - works with any domain.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import Dict, List, Any, Optional, Tuple, Callable
import json
import argparse
from tqdm import tqdm

from .plugin import Pipeline, MethodCategory, Method, DomainPlugin
from .registry import get_registry, auto_discover_domains


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


def format_combo_id(stage_specs: List[Tuple[str, Dict[str, Any]]]) -> str:
    """Format a unique identifier for a method combination.
    
    Args:
        stage_specs: List of (method_name, params) for each stage
        
    Returns:
        String ID in format: stage1_params___stage2_params___...
    """
    parts = []
    for method_name, params in stage_specs:
        part = method_name
        if params:
            param_str = '_'.join(f"{k}{v}" for k, v in sorted(params.items()))
            part = f"{method_name}_{param_str}"
        parts.append(part)
    
    return '___'.join(parts)


def generate_pipeline_combinations(pipeline: Pipeline) -> List[Dict[str, Any]]:
    """Generate all method + parameter combinations for a pipeline.
    
    Args:
        pipeline: Pipeline with stages
        
    Returns:
        List of combination dictionaries with 'id' and 'stages' keys
    """
    if not pipeline.stages:
        return []
    
    # Generate all combinations for each stage
    stage_combos = []
    for stage in pipeline.stages:
        stage_methods = []
        for method_name, method in stage.methods.items():
            for params in method.get_param_combinations():
                stage_methods.append((method_name, params))
        stage_combos.append(stage_methods)
    
    # Cartesian product across all stages
    combinations = []
    for combo in product(*stage_combos):
        combo_id = format_combo_id(combo)
        combinations.append({
            'id': combo_id,
            'stages': [
                {'name': method_name, 'params': params}
                for method_name, params in combo
            ]
        })
    
    return combinations


def run_pipeline_combination(
    pipeline: Pipeline,
    data: np.ndarray,
    stage_specs: List[Dict[str, Any]]
) -> Any:
    """Run a single pipeline with specific methods and parameters.
    
    Args:
        pipeline: The pipeline to run
        data: Input data
        stage_specs: List of {'name': method_name, 'params': params} for each stage
        
    Returns:
        Pipeline output
    """
    if len(stage_specs) != len(pipeline.stages):
        raise ValueError(f"Expected {len(pipeline.stages)} stage specs, got {len(stage_specs)}")
    
    result = data
    for stage, spec in zip(pipeline.stages, stage_specs):
        method = stage.get(spec['name'])
        if method is None:
            raise ValueError(f"Unknown method '{spec['name']}' in stage '{stage.name}'")
        result = method.run(result, spec['params'])
    
    return result


def run_exhaustive(
    data: np.ndarray,
    pipeline: Pipeline,
    output_dir: Path,
    skip_existing: bool = True,
    combinations: Optional[List[Dict]] = None,
    save_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """Run exhaustive exploration of all method combinations for a pipeline.
    
    Args:
        data: Input 2D array
        pipeline: Pipeline to explore
        output_dir: Directory to save results
        skip_existing: Skip combinations that already have output files
        combinations: Specific combinations to run (None = all)
        save_fn: Custom save function (result, path) -> None
        
    Returns:
        Summary statistics
    """
    output_dir = Path(output_dir)
    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if combinations is None:
        combinations = generate_pipeline_combinations(pipeline)
    
    if save_fn is None:
        def save_fn(result, path):
            # Default: save as float32 numpy array
            if isinstance(result, tuple):
                # For decomposition pipelines, save the residual (second element)
                result = result[1] if len(result) > 1 else result[0]
            np.save(path, np.asarray(result).astype(np.float32))
    
    stats = {
        'total': len(combinations),
        'completed': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }
    
    pbar = tqdm(combinations, desc=f"Running {pipeline.name}")
    
    for combo in pbar:
        combo_id = combo['id']
        result_path = results_dir / f"{combo_id}.npy"
        
        if skip_existing and result_path.exists():
            stats['skipped'] += 1
            pbar.set_postfix(done=stats['completed'], skip=stats['skipped'])
            continue
        
        try:
            result = run_pipeline_combination(pipeline, data, combo['stages'])
            save_fn(result, result_path)
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
    pipeline: Pipeline,
    output_dir: Path,
    domain_name: str = "Generic"
) -> Path:
    """Generate prior art documentation markdown for a pipeline.
    
    Args:
        pipeline: Pipeline to document
        output_dir: Output directory
        domain_name: Name of the domain
        
    Returns:
        Path to generated document
    """
    output_dir = Path(output_dir)
    doc_dir = output_dir / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_path = doc_dir / f"PRIOR_ART_{domain_name}_{timestamp}.md"
    
    combinations = generate_pipeline_combinations(pipeline)
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(f"# PRIORART: {domain_name} - {pipeline.name} Prior Art Documentation\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        f.write("## Purpose\n\n")
        f.write("This document establishes prior art for combinations of processing methods.\n")
        f.write("Published under Apache 2.0 license with explicit patent grant.\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Domain**: {domain_name}\n")
        f.write(f"- **Pipeline**: {pipeline.name}\n")
        f.write(f"- **Total Combinations**: {len(combinations):,}\n\n")
        
        # Document each stage
        for i, stage in enumerate(pipeline.stages, 1):
            f.write(f"## Stage {i}: {stage.name}\n\n")
            if stage.description:
                f.write(f"{stage.description}\n\n")
            
            f.write(f"**Methods**: {len(stage.methods)}\n\n")
            
            for method_name, method in sorted(stage.methods.items()):
                f.write(f"### {method_name}\n\n")
                if method.description:
                    f.write(f"{method.description}\n\n")
                if method.preserves:
                    f.write(f"- **Preserves**: {method.preserves}\n")
                if method.destroys:
                    f.write(f"- **Destroys**: {method.destroys}\n")
                if method.param_ranges:
                    f.write(f"- **Parameters**: `{method.param_ranges}`\n")
                    f.write(f"- **Combinations**: {len(method.get_param_combinations())}\n")
                f.write("\n")
        
        # List all combinations
        f.write("## All Combinations\n\n")
        f.write("Each combination below represents a distinct processing pipeline:\n\n")
        
        for i, combo in enumerate(combinations, 1):
            f.write(f"{i}. `{combo['id']}`\n")
            if i % 100 == 0:
                f.write("\n")
        
        f.write("\n---\n")
        f.write(f"\n*Document generated by PRIORART v0.2.0*\n")
    
    return doc_path


def run_domain_exhaustive(
    domain_name: str,
    pipeline_name: str,
    input_data: np.ndarray,
    output_dir: Path,
    skip_existing: bool = True,
    doc_only: bool = False
) -> Dict[str, Any]:
    """Run exhaustive exploration for a specific domain and pipeline.
    
    This is a high-level convenience function.
    
    Args:
        domain_name: Name of the domain (e.g., 'terrain', 'astronomy')
        pipeline_name: Name of the pipeline within the domain
        input_data: Input data array
        output_dir: Output directory
        skip_existing: Skip existing outputs
        doc_only: Only generate documentation
        
    Returns:
        Statistics dictionary
    """
    # Get the domain
    registry = get_registry()
    domain = registry.get(domain_name)
    
    if domain is None:
        # Try auto-discovery
        auto_discover_domains()
        domain = registry.get(domain_name)
        
    if domain is None:
        raise ValueError(f"Unknown domain: {domain_name}. Available: {registry.list_domains()}")
    
    # Get the pipeline
    pipeline = domain.get_pipeline(pipeline_name)
    if pipeline is None:
        raise ValueError(f"Unknown pipeline: {pipeline_name}. Available: {domain.list_pipelines()}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate documentation
    doc_path = generate_prior_art_doc(pipeline, output_dir, domain_name)
    print(f"Documentation: {doc_path}")
    
    if doc_only:
        return {'doc_path': str(doc_path)}
    
    # Run exhaustive
    print(f"\nRunning exhaustive exploration for {domain_name}/{pipeline_name}")
    print(f"Input shape: {input_data.shape}")
    
    combinations = generate_pipeline_combinations(pipeline)
    print(f"Total combinations: {len(combinations):,}")
    
    stats = run_exhaustive(
        input_data,
        pipeline,
        output_dir,
        skip_existing=skip_existing
    )
    
    print(f"\nCompleted: {stats['completed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    
    # Save stats
    stats_path = output_dir / 'run_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PRIORART Exhaustive Runner - Generate prior art'
    )
    parser.add_argument('input', type=str, help='Input .npy file (2D array)')
    parser.add_argument('--domain', '-d', type=str, required=True,
                        help='Domain name (e.g., terrain, astronomy)')
    parser.add_argument('--pipeline', '-p', type=str, default=None,
                        help='Pipeline name (default: first pipeline in domain)')
    parser.add_argument('--output', '-o', type=str, default='./priorart_results',
                        help='Output directory')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip combinations with existing output')
    parser.add_argument('--doc-only', action='store_true',
                        help='Only generate documentation, no processing')
    parser.add_argument('--list-domains', action='store_true',
                        help='List available domains and exit')
    
    args = parser.parse_args()
    
    # Auto-discover domains
    auto_discover_domains()
    registry = get_registry()
    
    if args.list_domains:
        print(registry.summary())
        return
    
    # Get domain
    domain = registry.get(args.domain)
    if domain is None:
        print(f"Error: Unknown domain '{args.domain}'")
        print(f"Available domains: {registry.list_domains()}")
        return
    
    # Get pipeline
    pipeline_name = args.pipeline
    if pipeline_name is None:
        pipelines = domain.list_pipelines()
        if not pipelines:
            print(f"Error: Domain '{args.domain}' has no pipelines")
            return
        pipeline_name = pipelines[0]
        print(f"Using default pipeline: {pipeline_name}")
    
    # Load input data
    data = np.load(args.input)
    
    # Run
    run_domain_exhaustive(
        domain_name=args.domain,
        pipeline_name=pipeline_name,
        input_data=data,
        output_dir=Path(args.output),
        skip_existing=args.skip_existing,
        doc_only=args.doc_only
    )


if __name__ == '__main__':
    main()
