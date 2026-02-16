"""Learned router: decision tree or MLP trained on benchmark data."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from kvshuttle.router.features import RouterInput

logger = logging.getLogger(__name__)


@dataclass
class LearnedRouter:
    """Learned routing model.

    Wraps a scikit-learn classifier trained to predict the best compressor
    given request features.
    """

    model: object  # sklearn classifier
    label_names: list[str]  # compressor names in label order
    model_type: str = "decision_tree"

    def predict(self, router_input: RouterInput) -> str:
        """Predict the best compressor for a request.

        Args:
            router_input: Features for the current request.

        Returns:
            Name of the recommended compressor.
        """
        features = router_input.to_feature_vector().reshape(1, -1)
        pred_idx = self.model.predict(features)[0]
        return self.label_names[pred_idx]

    def predict_batch(self, features: np.ndarray) -> list[str]:
        """Predict compressors for a batch of feature vectors.

        Args:
            features: Array of shape [N, num_features].

        Returns:
            List of compressor names.
        """
        pred_indices = self.model.predict(features)
        return [self.label_names[i] for i in pred_indices]

    @classmethod
    def train(
        cls,
        features: np.ndarray,
        best_compressor_indices: np.ndarray,
        label_names: list[str],
        model_type: str = "decision_tree",
    ) -> LearnedRouter:
        """Train a routing model.

        Args:
            features: Training features, shape [N, num_features].
            best_compressor_indices: Target labels (integer indices into label_names).
            label_names: Compressor names corresponding to label indices.
            model_type: "decision_tree", "mlp", or "gradient_boosting".

        Returns:
            Trained LearnedRouter.
        """
        if model_type == "decision_tree":
            model = DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=5, random_state=42
            )
        elif model_type == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=42,
                early_stopping=True,
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(features, best_compressor_indices)
        accuracy = model.score(features, best_compressor_indices)
        logger.info("Trained %s router: training accuracy = %.3f", model_type, accuracy)

        return cls(model=model, label_names=label_names, model_type=model_type)
