#!/usr/bin/env python3
"""
PRIORART Plugin Interface

Defines the base classes for domain plugins. Each domain (terrain, astronomy, etc.)
implements these interfaces to register their feature extraction methods.

Architecture:
    DomainPlugin
    └── pipelines: List[Pipeline]
        └── stages: List[MethodCategory]
            └── methods: Dict[str, Method]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional, Tuple
import numpy as np


@dataclass
class Method:
    """A single processing method with parameters.
    
    Attributes:
        name: Unique identifier for the method
        func: Callable that performs the processing
        category: Category name (e.g., 'decomposition', 'morphology')
        default_params: Default parameter values
        param_ranges: Dict mapping param names to lists of values to explore
        preserves: Description of what features are preserved
        destroys: Description of what features are removed/destroyed
        description: Human-readable description
    """
    name: str
    func: Callable
    category: str
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_ranges: Dict[str, List[Any]] = field(default_factory=dict)
    preserves: str = ""
    destroys: str = ""
    description: str = ""
    
    def run(self, data: np.ndarray, params: Optional[Dict[str, Any]] = None) -> Any:
        """Run the method with given parameters.
        
        Args:
            data: Input data array
            params: Optional parameter overrides
            
        Returns:
            Result of the method (typically np.ndarray or tuple of arrays)
        """
        effective_params = {**self.default_params}
        if params:
            effective_params.update(params)
        return self.func(data, **effective_params)
    
    def get_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from param_ranges.
        
        Returns:
            List of parameter dictionaries
        """
        if not self.param_ranges:
            return [self.default_params.copy()]
        
        from itertools import product
        keys = list(self.param_ranges.keys())
        values = [self.param_ranges[k] for k in keys]
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        return combinations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for documentation/serialization."""
        return {
            'name': self.name,
            'category': self.category,
            'default_params': self.default_params,
            'param_ranges': self.param_ranges,
            'preserves': self.preserves,
            'destroys': self.destroys,
            'description': self.description
        }


@dataclass
class MethodCategory:
    """A category of related methods (e.g., all decomposition methods).
    
    Attributes:
        name: Category name (e.g., 'decomposition', 'upsampling', 'morphology')
        description: Human-readable description
        methods: Dict mapping method names to Method objects
    """
    name: str
    description: str = ""
    methods: Dict[str, Method] = field(default_factory=dict)
    
    def register(self, method: Method):
        """Register a method in this category."""
        self.methods[method.name] = method
    
    def get(self, name: str) -> Optional[Method]:
        """Get a method by name."""
        return self.methods.get(name)
    
    def list_methods(self) -> List[str]:
        """List all method names in this category."""
        return list(self.methods.keys())
    
    def count_combinations(self) -> int:
        """Count total parameter combinations across all methods."""
        return sum(len(m.get_param_combinations()) for m in self.methods.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for documentation/serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'methods': {k: v.to_dict() for k, v in self.methods.items()},
            'total_combinations': self.count_combinations()
        }


@dataclass
class Pipeline:
    """A processing pipeline consisting of ordered stages.
    
    For terrain: decomposition → upsampling
    For astronomy: morphology extraction → radial profile
    
    Each stage is a MethodCategory. The pipeline defines how stages
    are composed together.
    
    Attributes:
        name: Pipeline name (e.g., 'decomp_upsample', 'morphology')
        description: Human-readable description
        stages: Ordered list of MethodCategories
        compose_fn: Function to compose stage outputs
    """
    name: str
    description: str = ""
    stages: List[MethodCategory] = field(default_factory=list)
    compose_fn: Optional[Callable] = None
    
    def add_stage(self, category: MethodCategory):
        """Add a processing stage."""
        self.stages.append(category)
    
    def run(self, data: np.ndarray, stage_methods: List[Tuple[str, Dict[str, Any]]]) -> Any:
        """Run the pipeline with specified methods for each stage.
        
        Args:
            data: Input data
            stage_methods: List of (method_name, params) for each stage
            
        Returns:
            Pipeline output
        """
        if len(stage_methods) != len(self.stages):
            raise ValueError(f"Expected {len(self.stages)} stage methods, got {len(stage_methods)}")
        
        result = data
        intermediate_results = []
        
        for stage, (method_name, params) in zip(self.stages, stage_methods):
            method = stage.get(method_name)
            if method is None:
                raise ValueError(f"Unknown method '{method_name}' in stage '{stage.name}'")
            result = method.run(result, params)
            intermediate_results.append(result)
        
        if self.compose_fn:
            return self.compose_fn(intermediate_results)
        return result
    
    def count_combinations(self) -> int:
        """Count total combinations across all stages."""
        total = 1
        for stage in self.stages:
            total *= stage.count_combinations()
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for documentation/serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'stages': [s.to_dict() for s in self.stages],
            'total_combinations': self.count_combinations()
        }


class DomainPlugin(ABC):
    """Base class for domain plugins.
    
    Each domain (terrain, astronomy, cmb, satellite, etc.) implements
    this interface to register its feature extraction methods and pipelines.
    
    Example:
        class TerrainPlugin(DomainPlugin):
            name = "terrain"
            description = "Digital Elevation Model processing"
            
            def get_pipelines(self) -> List[Pipeline]:
                decomp = MethodCategory("decomposition", ...)
                upsamp = MethodCategory("upsampling", ...)
                pipeline = Pipeline("decomp_upsample", stages=[decomp, upsamp])
                return [pipeline]
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique domain identifier."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable domain description."""
        pass
    
    @abstractmethod
    def get_pipelines(self) -> List[Pipeline]:
        """Return list of processing pipelines for this domain."""
        pass
    
    def get_generators(self) -> Dict[str, Callable]:
        """Return synthetic test data generators.
        
        Optional. Override to provide test data generation functions.
        
        Returns:
            Dict mapping generator names to callables that produce test data.
        """
        return {}
    
    def get_pipeline(self, name: str) -> Optional[Pipeline]:
        """Get a pipeline by name."""
        for p in self.get_pipelines():
            if p.name == name:
                return p
        return None
    
    def list_pipelines(self) -> List[str]:
        """List all pipeline names."""
        return [p.name for p in self.get_pipelines()]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for documentation/serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'pipelines': [p.to_dict() for p in self.get_pipelines()],
            'generators': list(self.get_generators().keys())
        }


def method_decorator(
    name: str,
    category: str,
    default_params: Optional[Dict[str, Any]] = None,
    param_ranges: Optional[Dict[str, List[Any]]] = None,
    preserves: str = "",
    destroys: str = "",
    description: str = ""
) -> Callable:
    """Decorator to register a function as a Method.
    
    Usage:
        @method_decorator(
            name='gaussian',
            category='decomposition',
            default_params={'sigma': 10},
            param_ranges={'sigma': [2, 5, 10, 20, 50, 100]},
            preserves='smooth regions',
            destroys='high-frequency detail'
        )
        def gaussian_decomposition(data, sigma=10):
            ...
    """
    def decorator(func: Callable) -> Method:
        return Method(
            name=name,
            func=func,
            category=category,
            default_params=default_params or {},
            param_ranges=param_ranges or {},
            preserves=preserves,
            destroys=destroys,
            description=description
        )
    return decorator
