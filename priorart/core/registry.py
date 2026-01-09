#!/usr/bin/env python3
"""
PRIORART Domain Registry

Handles plugin discovery, registration, and access.
Provides a central point to find all available domains and their pipelines.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import importlib
import pkgutil

from .plugin import DomainPlugin, Pipeline


class DomainRegistry:
    """Registry for domain plugins.
    
    Provides methods to:
    - Register plugins manually
    - Auto-discover plugins from the domains package
    - Access plugins and their pipelines
    
    Usage:
        registry = DomainRegistry()
        registry.auto_discover()  # Finds all plugins in priorart.domains
        
        # Or register manually
        registry.register(TerrainPlugin())
        
        # Access
        terrain = registry.get("terrain")
        pipeline = terrain.get_pipeline("decomp_upsample")
    """
    
    def __init__(self):
        self._plugins: Dict[str, DomainPlugin] = {}
    
    def register(self, plugin: DomainPlugin) -> None:
        """Register a domain plugin.
        
        Args:
            plugin: DomainPlugin instance
        """
        if plugin.name in self._plugins:
            print(f"Warning: Overwriting existing plugin '{plugin.name}'")
        self._plugins[plugin.name] = plugin
    
    def unregister(self, name: str) -> bool:
        """Unregister a domain plugin.
        
        Args:
            name: Domain name
            
        Returns:
            True if plugin was removed, False if not found
        """
        if name in self._plugins:
            del self._plugins[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[DomainPlugin]:
        """Get a domain plugin by name.
        
        Args:
            name: Domain name (e.g., 'terrain', 'astronomy')
            
        Returns:
            DomainPlugin instance or None
        """
        return self._plugins.get(name)
    
    def list_domains(self) -> List[str]:
        """List all registered domain names."""
        return list(self._plugins.keys())
    
    def get_all(self) -> Dict[str, DomainPlugin]:
        """Get all registered plugins."""
        return dict(self._plugins)
    
    def get_pipeline(self, domain: str, pipeline: str) -> Optional[Pipeline]:
        """Get a specific pipeline from a domain.
        
        Args:
            domain: Domain name
            pipeline: Pipeline name
            
        Returns:
            Pipeline instance or None
        """
        plugin = self.get(domain)
        if plugin:
            return plugin.get_pipeline(pipeline)
        return None
    
    def list_all_pipelines(self) -> Dict[str, List[str]]:
        """List all pipelines organized by domain.
        
        Returns:
            Dict mapping domain names to lists of pipeline names
        """
        return {
            name: plugin.list_pipelines()
            for name, plugin in self._plugins.items()
        }
    
    def count_total_combinations(self) -> Dict[str, int]:
        """Count total combinations per domain.
        
        Returns:
            Dict mapping domain names to total combination counts
        """
        counts = {}
        for name, plugin in self._plugins.items():
            total = sum(p.count_combinations() for p in plugin.get_pipelines())
            counts[name] = total
        return counts
    
    def auto_discover(self, package_name: str = "priorart.domains") -> int:
        """Auto-discover and register plugins from a package.
        
        Scans the specified package for modules that define a 
        `get_plugin()` function or a `PLUGIN` variable.
        
        Args:
            package_name: Package to scan for plugins
            
        Returns:
            Number of plugins discovered
        """
        discovered = 0
        
        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            print(f"Warning: Could not import {package_name}: {e}")
            return 0
        
        # Get package path
        if hasattr(package, '__path__'):
            package_path = package.__path__
        else:
            return 0
        
        # Iterate through submodules
        for importer, modname, ispkg in pkgutil.iter_modules(package_path):
            full_modname = f"{package_name}.{modname}"
            
            try:
                module = importlib.import_module(full_modname)
                
                # Look for get_plugin() function
                if hasattr(module, 'get_plugin'):
                    plugin = module.get_plugin()
                    if isinstance(plugin, DomainPlugin):
                        self.register(plugin)
                        discovered += 1
                        continue
                
                # Look for PLUGIN variable
                if hasattr(module, 'PLUGIN'):
                    plugin = module.PLUGIN
                    if isinstance(plugin, DomainPlugin):
                        self.register(plugin)
                        discovered += 1
                        continue
                
            except Exception as e:
                print(f"Warning: Could not load plugin from {full_modname}: {e}")
        
        return discovered
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary for documentation/serialization."""
        return {
            'domains': {name: p.to_dict() for name, p in self._plugins.items()},
            'total_domains': len(self._plugins),
            'total_combinations': self.count_total_combinations()
        }
    
    def summary(self) -> str:
        """Generate a text summary of registered domains."""
        lines = ["PRIORART Domain Registry", "=" * 40, ""]
        
        if not self._plugins:
            lines.append("No domains registered.")
            return "\n".join(lines)
        
        for name, plugin in sorted(self._plugins.items()):
            lines.append(f"Domain: {name}")
            lines.append(f"  Description: {plugin.description}")
            lines.append(f"  Pipelines: {', '.join(plugin.list_pipelines())}")
            
            for pipeline in plugin.get_pipelines():
                lines.append(f"    - {pipeline.name}: {pipeline.count_combinations():,} combinations")
            lines.append("")
        
        return "\n".join(lines)


# Global registry instance
_global_registry: Optional[DomainRegistry] = None


def get_registry() -> DomainRegistry:
    """Get or create the global domain registry.
    
    Returns:
        The global DomainRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = DomainRegistry()
    return _global_registry


def register_domain(plugin: DomainPlugin) -> None:
    """Register a plugin with the global registry.
    
    Convenience function for:
        get_registry().register(plugin)
    """
    get_registry().register(plugin)


def get_domain(name: str) -> Optional[DomainPlugin]:
    """Get a domain from the global registry.
    
    Convenience function for:
        get_registry().get(name)
    """
    return get_registry().get(name)


def list_domains() -> List[str]:
    """List all domains in the global registry.
    
    Convenience function for:
        get_registry().list_domains()
    """
    return get_registry().list_domains()


def auto_discover_domains(package_name: str = "priorart.domains") -> int:
    """Auto-discover domains in the global registry.
    
    Convenience function for:
        get_registry().auto_discover(package_name)
    """
    return get_registry().auto_discover(package_name)
