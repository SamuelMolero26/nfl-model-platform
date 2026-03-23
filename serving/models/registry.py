import importlib
import json
from pathlib import Path
from typing import Optional

from .base import BaseModel

# Map MODEL_NAME → fully-qualified class path for dynamic import
_MODEL_CLASS_MAP: dict[str, str] = {
    "player_projection": "serving.models.player_projection.model.PlayerProjectionModel",
    "draft_optimizer": "serving.models.draft_optimizer.model.DraftOptimizerModel",
    "team_diagnosis": "serving.models.team_diagnosis.model.TeamDiagnosisModel",
    "career_simulator": "serving.models.career_simulator.model.CareerSimulatorModel",
    "roster_fit": "serving.models.roster_fit.model.RosterFitModel",
    "positional_flexibility": "serving.models.positional_flexibility.model.PositionalFlexibilityModel",
    "health_analyzer": "serving.models.health_analyzer.model.HealthAnalyzerModel",
}


class ModelRegistry:
    """
    Singleton registry for all NFL models.

    - Scans artifacts/ on init to discover available model versions.
    - Lazy-loads model objects from disk on first predict() call.
    - Caches loaded models in memory for subsequent calls.
    - Supports versioned artifacts; defaults to the latest version.

    Usage:
        registry = ModelRegistry.instance()
        result = registry.predict("player_projection", inputs)
        info = registry.list_models()
    """

    _instance: Optional["ModelRegistry"] = None

    def __init__(self, artifacts_root: Optional[Path] = None):
        self._artifacts_root = artifacts_root or (
            Path(__file__).parents[2] / "artifacts"
        )
        # { model_name: { version: BaseModel } }  — None means not yet loaded
        self._cache: dict[str, dict[str, Optional[BaseModel]]] = {}
        self._scan()

    # Singleton

    @classmethod
    def instance(cls, artifacts_root: Optional[Path] = None) -> "ModelRegistry":
        """Return (or create) the global singleton."""
        if cls._instance is None:
            cls._instance = cls(artifacts_root)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear singleton — useful in tests."""
        cls._instance = None

    # Discovery

    def _scan(self) -> None:
        """Scan artifacts/ and register all model + version directories."""
        if not self._artifacts_root.exists():
            return
        for model_dir in sorted(self._artifacts_root.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            self._cache.setdefault(model_name, {})
            for version_dir in sorted(model_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                if (version_dir / "model.pkl").exists():
                    # Register as None (not yet loaded)
                    self._cache[model_name][version_dir.name] = None

    def refresh(self) -> None:
        """Re-scan artifacts/ to pick up newly trained models."""
        self._scan()

    # Access
    def latest_version(self, model_name: str) -> str:
        """Return the latest version string for a model (lexicographic sort)."""
        versions = self._cache.get(model_name, {})
        if not versions:
            raise KeyError(f"No artifacts found for model '{model_name}'")
        return sorted(versions.keys())[-1]

    def get(self, model_name: str, version: Optional[str] = None) -> BaseModel:
        """
        Return a loaded model instance.
        Lazy-loads from disk on first call; subsequent calls return cached instance.
        """
        if model_name not in self._cache or not self._cache[model_name]:
            raise KeyError(
                f"Model '{model_name}' not found in artifacts. "
                f"Available: {list(self._cache.keys())}"
            )

        version = version or self.latest_version(model_name)
        if version not in self._cache[model_name]:
            raise KeyError(
                f"Version '{version}' not found for model '{model_name}'. "
                f"Available: {list(self._cache[model_name].keys())}"
            )

        # Return cached instance if already loaded
        if self._cache[model_name][version] is not None:
            return self._cache[model_name][version]

        # Lazy-load
        model = self._load(model_name, version)
        self._cache[model_name][version] = model
        return model

    def predict(
        self, model_name: str, inputs: dict, version: Optional[str] = None
    ) -> dict:
        """Convenience: get model and run predict in one call."""
        return self.get(model_name, version).predict(inputs)

    # Loading

    def _load(self, model_name: str, version: str) -> BaseModel:
        """Instantiate model class and load artifact from disk."""
        class_path = _MODEL_CLASS_MAP.get(model_name)
        if class_path is None:
            raise ValueError(
                f"No class mapping for model '{model_name}'. "
                f"Register it in _MODEL_CLASS_MAP."
            )
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        instance: BaseModel = cls()
        artifact_dir = self._artifacts_root / model_name / version
        instance.load(artifact_dir)
        return instance

    # Introspection
    def list_models(self) -> list[dict]:
        """
        List all registered models with their available versions and metadata.

        Returns a list of dicts:
            [{"model_name": ..., "versions": [...], "latest": ..., "metadata": {...}}, ...]
        """
        result = []
        for model_name, versions in self._cache.items():
            entry: dict = {
                "model_name": model_name,
                "versions": sorted(versions.keys()),
                "latest": sorted(versions.keys())[-1] if versions else None,
                "loaded": [v for v, m in versions.items() if m is not None],
                "metadata": {},
            }
            # Read metadata.json for the latest version (no need to load pkl)
            latest = entry["latest"]
            if latest:
                meta_path = self._artifacts_root / model_name / latest / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        entry["metadata"] = json.load(f)
            result.append(entry)
        return result

    def is_available(self, model_name: str, version: Optional[str] = None) -> bool:
        """Return True if the model artifact exists on disk."""
        if model_name not in self._cache or not self._cache[model_name]:
            return False
        v = version or self.latest_version(model_name)
        return v in self._cache[model_name]

    def __repr__(self) -> str:
        summary = {k: list(v.keys()) for k, v in self._cache.items()}
        return f"<ModelRegistry artifacts={self._artifacts_root} models={summary}>"
