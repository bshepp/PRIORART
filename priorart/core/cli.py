#!/usr/bin/env python3
"""
PRIORART Unified CLI

Provides commands for:
- domains: List available domain plugins
- run: Run exhaustive exploration
- checksums: Generate/verify checksums
- analyze: Run redundancy analysis
- fingerprint: Generate fingerprints for results
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from .registry import get_registry, auto_discover_domains
from .exhaustive import run_domain_exhaustive, generate_pipeline_combinations
from .checksums import generate_checksums, verify_checksums
from .analysis import run_analysis
from .fingerprinting import run_parallel_fingerprinting


def cmd_domains(args):
    """List available domains and pipelines."""
    auto_discover_domains()
    registry = get_registry()
    
    print(registry.summary())
    
    if args.verbose:
        print("\nDetailed combinations:")
        for domain_name, count in registry.count_total_combinations().items():
            print(f"  {domain_name}: {count:,} total combinations")


def cmd_run(args):
    """Run exhaustive exploration."""
    auto_discover_domains()
    
    # Load input data
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    data = np.load(args.input)
    print(f"Loaded input: {data.shape}")
    
    # Run exhaustive exploration
    stats = run_domain_exhaustive(
        domain_name=args.domain,
        pipeline_name=args.pipeline,
        input_data=data,
        output_dir=Path(args.output),
        skip_existing=args.skip_existing,
        doc_only=args.doc_only
    )
    
    return 0


def cmd_checksums(args):
    """Generate or verify checksums."""
    source_dir = Path(args.source)
    
    if args.verify:
        # Verify mode
        manifest_path = Path(args.manifest) if args.manifest else source_dir / 'CHECKSUMS.txt'
        if not manifest_path.exists():
            print(f"Error: Manifest not found: {manifest_path}")
            return 1
        
        results = verify_checksums(source_dir, manifest_path)
        print(f"Valid: {len(results['valid'])}")
        print(f"Invalid: {len(results['invalid'])}")
        print(f"Missing: {len(results['missing'])}")
        return 0 if len(results['invalid']) == 0 else 1
    else:
        # Generate mode
        manifest_path = Path(args.output) if args.output else source_dir / 'CHECKSUMS.txt'
        stats = generate_checksums(source_dir, manifest_path, skip_existing=not args.force)
        print(f"Total files: {stats['total']}")
        print(f"Computed: {stats['computed']}")
        print(f"Skipped: {stats['skipped']}")
        return 0


def cmd_analyze(args):
    """Run redundancy analysis."""
    results_dir = Path(args.results)
    
    checksums_file = Path(args.checksums) if args.checksums else None
    output_path = Path(args.output) if args.output else results_dir / 'REDUNDANCY_REPORT.md'
    checkpoint_file = Path(args.checkpoint) if args.checkpoint else None
    
    run_analysis(
        results_dir=results_dir,
        checksums_file=checksums_file,
        output_path=output_path,
        checkpoint_file=checkpoint_file,
        threshold=args.threshold,
        fast=args.fast,
        skip_clustering=args.skip_clustering
    )
    
    return 0


def cmd_fingerprint(args):
    """Generate fingerprints for results."""
    results_dir = Path(args.source)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else results_dir / 'fingerprints.json'
    
    run_parallel_fingerprinting(
        results_dir=results_dir,
        checkpoint_path=checkpoint_path,
        num_workers=args.workers,
        checkpoint_interval=args.interval
    )
    
    return 0


def cmd_info(args):
    """Show information about a domain."""
    auto_discover_domains()
    registry = get_registry()
    
    domain = registry.get(args.domain)
    if domain is None:
        print(f"Error: Unknown domain '{args.domain}'")
        print(f"Available: {registry.list_domains()}")
        return 1
    
    print(f"Domain: {domain.name}")
    print(f"Description: {domain.description}")
    print()
    print("Pipelines:")
    
    for pipeline in domain.get_pipelines():
        print(f"\n  {pipeline.name}:")
        if pipeline.description:
            print(f"    {pipeline.description}")
        print(f"    Stages: {[s.name for s in pipeline.stages]}")
        print(f"    Combinations: {pipeline.count_combinations():,}")
        
        if args.verbose:
            for stage in pipeline.stages:
                print(f"\n    Stage: {stage.name}")
                for method_name, method in stage.methods.items():
                    combos = len(method.get_param_combinations())
                    print(f"      - {method_name}: {combos} combinations")
    
    generators = domain.get_generators()
    if generators:
        print(f"\nGenerators: {list(generators.keys())}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PRIORART - Systematic Prior Art Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  priorart domains                    List available domains
  priorart info terrain               Show terrain domain details
  priorart run input.npy -d terrain   Run exhaustive for terrain
  priorart checksums ./results        Generate checksums
  priorart analyze ./results          Run redundancy analysis
  priorart fingerprint ./results      Generate fingerprints
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # domains command
    p_domains = subparsers.add_parser('domains', help='List available domains')
    p_domains.add_argument('-v', '--verbose', action='store_true',
                           help='Show detailed information')
    
    # info command
    p_info = subparsers.add_parser('info', help='Show domain information')
    p_info.add_argument('domain', type=str, help='Domain name')
    p_info.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed methods')
    
    # run command
    p_run = subparsers.add_parser('run', help='Run exhaustive exploration')
    p_run.add_argument('input', type=str, help='Input .npy file')
    p_run.add_argument('-d', '--domain', type=str, required=True,
                       help='Domain name')
    p_run.add_argument('-p', '--pipeline', type=str, default=None,
                       help='Pipeline name (default: first)')
    p_run.add_argument('-o', '--output', type=str, default='./priorart_results',
                       help='Output directory')
    p_run.add_argument('--skip-existing', action='store_true',
                       help='Skip existing outputs')
    p_run.add_argument('--doc-only', action='store_true',
                       help='Only generate documentation')
    
    # checksums command
    p_check = subparsers.add_parser('checksums', help='Generate/verify checksums')
    p_check.add_argument('source', type=str, help='Source directory')
    p_check.add_argument('-o', '--output', type=str, default=None,
                         help='Output manifest path')
    p_check.add_argument('--verify', action='store_true',
                         help='Verify instead of generate')
    p_check.add_argument('--manifest', type=str, default=None,
                         help='Manifest file for verification')
    p_check.add_argument('--force', action='store_true',
                         help='Recompute all checksums')
    
    # analyze command
    p_analyze = subparsers.add_parser('analyze', help='Run redundancy analysis')
    p_analyze.add_argument('results', type=str, help='Results directory')
    p_analyze.add_argument('-o', '--output', type=str, default=None,
                           help='Output report path')
    p_analyze.add_argument('--checksums', type=str, default=None,
                           help='Checksums file for exact duplicate detection')
    p_analyze.add_argument('--checkpoint', type=str, default=None,
                           help='Fingerprint checkpoint file')
    p_analyze.add_argument('--threshold', type=float, default=0.999,
                           help='Near-duplicate threshold')
    p_analyze.add_argument('--fast', action='store_true',
                           help='Skip correlation verification')
    p_analyze.add_argument('--skip-clustering', action='store_true',
                           help='Skip clustering analysis')
    
    # fingerprint command
    p_fp = subparsers.add_parser('fingerprint', help='Generate fingerprints')
    p_fp.add_argument('source', type=str, help='Source directory')
    p_fp.add_argument('--checkpoint', type=str, default=None,
                      help='Checkpoint file path')
    p_fp.add_argument('--workers', type=int, default=2,
                      help='Number of parallel workers')
    p_fp.add_argument('--interval', type=int, default=500,
                      help='Checkpoint interval')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Dispatch to command handlers
    commands = {
        'domains': cmd_domains,
        'info': cmd_info,
        'run': cmd_run,
        'checksums': cmd_checksums,
        'analyze': cmd_analyze,
        'fingerprint': cmd_fingerprint,
    }
    
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
