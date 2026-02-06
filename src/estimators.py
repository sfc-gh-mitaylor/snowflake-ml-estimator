"""
Estimator Factory Module

Creates ML estimators by name using configuration from YAML.
Decouples model creation from usage - callers don't need to know
about imports, parameters, or instantiation details.

Usage:
    from src.estimators import EstimatorFactory
    
    factory = EstimatorFactory()
    
    # Create single estimator
    model = factory.create("XGBClassifier")
    
    # Create all enabled estimators
    models = factory.create_all()
    
    # List available estimators
    names = factory.list_available()
"""
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class EstimatorFactory:
    """
    Factory for creating ML estimators from YAML configuration.
    
    The factory pattern decouples WHAT to create from HOW to create it.
    Configuration lives in YAML, this class handles the mechanics.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize factory with configuration.
        
        Args:
            config_path: Path to YAML config. Defaults to estimators.yaml
                        in the same directory as this module.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "estimators.yaml"
        
        self.config_path = Path(config_path)
        self._config = None
    
    @property
    def config(self) -> Dict:
        """Lazy-load configuration on first access."""
        if self._config is None:
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f)
        return self._config
    
    def list_available(self, include_disabled: bool = False) -> List[str]:
        """
        List all available estimator names.
        
        Args:
            include_disabled: If True, include estimators with enabled=false
            
        Returns:
            List of estimator names
        """
        names = []
        for name, spec in self.config["estimators"].items():
            if include_disabled or spec.get("enabled", True):
                names.append(name)
        return names
    
    def get_spec(self, name: str) -> Dict:
        """
        Get the full specification for an estimator.
        
        Args:
            name: Estimator name (e.g., "XGBClassifier")
            
        Returns:
            Dict with module, class, params, row_limit, etc.
            
        Raises:
            KeyError: If estimator not found in config
        """
        if name not in self.config["estimators"]:
            available = self.list_available(include_disabled=True)
            raise KeyError(
                f"Estimator '{name}' not found. "
                f"Available: {available}"
            )
        return self.config["estimators"][name]
    
    def get_row_limit(self, name: str) -> Optional[int]:
        """Get row limit for an estimator, or None if unlimited."""
        spec = self.get_spec(name)
        return spec.get("row_limit")
    
    def create(self, name: str, **override_params) -> Any:
        """
        Create an estimator instance by name.
        
        Args:
            name: Estimator name (e.g., "XGBClassifier")
            **override_params: Override any default parameters
            
        Returns:
            Instantiated estimator object
            
        Example:
            # Use defaults from config
            model = factory.create("XGBClassifier")
            
            # Override specific params
            model = factory.create("XGBClassifier", n_estimators=500)
        """
        spec = self.get_spec(name)
        
        module = importlib.import_module(spec["module"])
        cls = getattr(module, spec.get("class", name))
        
        params = spec.get("params", {}).copy()
        params.update(override_params)
        
        return cls(**params)
    
    def create_all(self, include_disabled: bool = False) -> Dict[str, Any]:
        """
        Create all available estimators.
        
        Args:
            include_disabled: If True, include estimators with enabled=false
            
        Returns:
            Dict mapping estimator names to instances
        """
        estimators = {}
        for name in self.list_available(include_disabled=include_disabled):
            try:
                estimators[name] = self.create(name)
            except ImportError as e:
                print(f"Warning: Could not import {name}: {e}")
        return estimators
    
    def create_list(self, names: List[str]) -> List[Any]:
        """
        Create a list of specific estimators.
        
        Args:
            names: List of estimator names to create
            
        Returns:
            List of instantiated estimators (same order as names)
        """
        return [self.create(name) for name in names]


DEFAULT_FACTORY = EstimatorFactory()
