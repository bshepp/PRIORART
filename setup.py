#!/usr/bin/env python3
"""
PRIORART Setup Script

Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="priorart",
    version="0.2.0",
    description="Systematic Prior Art Generation for Mathematical Method Combinations",
    author="PRIORART Contributors",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests", "experiments"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-image>=0.18.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "full": [
            "PyWavelets>=1.1.0",
            "opencv-python>=4.5.0",
            "scikit-learn>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "priorart=priorart.core.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
)
