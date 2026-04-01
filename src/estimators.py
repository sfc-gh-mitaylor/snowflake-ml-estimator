"""
Estimator factory — creates ML models by name from YAML config.
"""
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class EstimatorFactory:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "estimators.yaml"
        self.config_path = Path(config_path)
        self._config = None

    @property
    def config(self) -> Dict:
        if self._config is None:
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f)
        return self._config

    def list_available(self, include_disabled: bool = False) -> List[str]:
        return [
            name for name, spec in self.config["estimators"].items()
            if include_disabled or spec.get("enabled", True)
        ]

    def get_spec(self, name: str) -> Dict:
        if name not in self.config["estimators"]:
            raise KeyError(f"Estimator '{name}' not found. Available: {self.list_available(include_disabled=True)}")
        return self.config["estimators"][name]

    def get_row_limit(self, name: str) -> Optional[int]:
        return self.get_spec(name).get("row_limit")

    def get_task_type(self, name: str) -> str:
        return self.get_spec(name).get("task_type", "classification")

    def create(self, name: str, **override_params) -> Any:
        spec = self.get_spec(name)
        module = importlib.import_module(spec["module"])
        cls = getattr(module, spec.get("class", name))
        params = spec.get("params", {}).copy()
        params.update(override_params)
        return cls(**params)


DEFAULT_FACTORY = EstimatorFactory()
