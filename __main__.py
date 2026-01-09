#!/usr/bin/env python3
"""
PRIORART Package Entry Point

Allows running PRIORART as a module:
    python -m priorart domains
    python -m priorart run input.npy -d terrain
    python -m priorart analyze ./results
"""

from priorart.core.cli import main

if __name__ == '__main__':
    main()
