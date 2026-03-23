import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for all NFL model platform models.

    Subclasses must implement:
        - train(X, y)
        - predict(inputs) -> dict
        - evaluate(X, y) -> dict
        - feature_names (property) -> list[str]

    Serialization (save/load) is handled here via pickle + metadata.json.
    """

    # Subclasses set this to identify themselves in the registry and artifacts/
    MODEL_NAME: str = ""
    MODEL_VERSION: str = "v1"

    def __init__(self):
        self._model: Any = None
        self._is_trained: bool = False
        self._metadata: dict = {}

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model. Must set self._is_trained = True when done."""
        ...

    @abstractmethod
    def predict(self, inputs: dict) -> dict:
        """
        Run inference given a dict of raw inputs.

        Returns a dict with at minimum:
            - "prediction": the primary output value(s)
            - "shap_values": dict mapping feature name → SHAP value (or None if not computed)
            - "metadata": dict with model_name, version, timestamp
        """
        ...

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Compute validation metrics on a held-out set.

        Returns a dict of metric_name → value, e.g.:
            {"rmse": 1.23, "mae": 0.98, "r2": 0.71}
        """
        ...

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Ordered list of feature column names the model was trained on."""
        ...

    # Prediction response helpers
    def _base_response(self) -> dict:
        """Standard metadata block included in every predict() response."""
        return {
            "model_name": self.MODEL_NAME,
            "version": self.MODEL_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Save / Load
    def save(self, artifact_dir: Optional[Path] = None) -> Path:
        """
        Persist model to disk.

        Creates:
            artifacts/{MODEL_NAME}/{MODEL_VERSION}/model.pkl
            artifacts/{MODEL_NAME}/{MODEL_VERSION}/metadata.json

        Returns the artifact directory path.
        """
        if not self._is_trained:
            raise RuntimeError(
                f"Cannot save {self.MODEL_NAME}: model has not been trained."
            )

        if artifact_dir is None:
            artifact_dir = (
                Path(__file__).parents[2]
                / "artifacts"
                / self.MODEL_NAME
                / self.MODEL_VERSION
            )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifact_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self._model, f)

        metadata = {
            "model_name": self.MODEL_NAME,
            "version": self.MODEL_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "feature_names": self.feature_names,
            **self._metadata,
        }
        with open(artifact_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return artifact_dir

    def load(self, artifact_dir: Optional[Path] = None) -> "BaseModel":
        """
        Load model from disk.

        Looks for:
            artifacts/{MODEL_NAME}/{MODEL_VERSION}/model.pkl
            artifacts/{MODEL_NAME}/{MODEL_VERSION}/metadata.json
        """
        if artifact_dir is None:
            artifact_dir = (
                Path(__file__).parents[2]
                / "artifacts"
                / self.MODEL_NAME
                / self.MODEL_VERSION
            )

        model_path = artifact_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"No model artifact at {model_path}")

        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

        metadata_path = artifact_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)

        self._is_trained = True
        return self

    # for ease of confirmation
    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return f"<{self.__class__.__name__} name={self.MODEL_NAME!r} version={self.MODEL_VERSION!r} {status}>"
