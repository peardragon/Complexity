"""The single DNN objective used by reference search and shell sampling.

There are deliberately no numerical fallbacks in this module.  A caller must
load the canonical JSON contract and must use every trainable parameter,
including biases, in the L2 term.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .provenance import canonical_json_bytes, sha256_bytes


EXPECTED_OBJECTIVE_ID = (
    "dnn_binary_logistic_ce_mean_l2_all_parameters_gamma_100_lambda_1_dense_extension_v1"
)
EXPECTED_L2_FORMULA = (
    "lambda_reg * squared_l2_norm / (2 * parameter_count)"
)
EXPECTED_EXACT_CHECKPOINT_POLICY = (
    "first_full_data_exact_post_optimizer_update"
)
REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "objective_id",
    "loss",
    "label_encoding",
    "ce_reduction",
    "gamma_ce",
    "regularizer",
    "regularized_parameters",
    "lambda_reg",
    "l2_formula",
    "inverse_temperature",
    "exact_reference",
    "first_radial_derivative",
)


@dataclass(frozen=True)
class ObjectiveContract:
    schema_version: int
    objective_id: str
    loss: str
    label_encoding: str
    ce_reduction: str
    gamma_ce: float
    regularizer: str
    regularized_parameters: str
    lambda_reg: float
    l2_formula: str
    inverse_temperature: float
    exact_reference: Mapping[str, Any]
    first_radial_derivative: Mapping[str, Any]
    fingerprint: str

    @classmethod
    def from_json(cls, path: str | Path) -> "ObjectiveContract":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("DNN objective JSON must contain one object")
        missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in payload]
        if missing:
            raise ValueError(
                f"invalid DNN objective contract: missing keys {missing!r}"
            )
        if not isinstance(payload["exact_reference"], dict):
            raise ValueError("exact_reference must be an object")
        if not isinstance(payload["first_radial_derivative"], dict):
            raise ValueError("first_radial_derivative must be an object")
        fingerprint = sha256_bytes(canonical_json_bytes(payload))
        contract = cls(
            schema_version=int(payload["schema_version"]),
            objective_id=str(payload["objective_id"]),
            loss=str(payload["loss"]),
            label_encoding=str(payload["label_encoding"]),
            ce_reduction=str(payload["ce_reduction"]),
            gamma_ce=float(payload["gamma_ce"]),
            regularizer=str(payload["regularizer"]),
            regularized_parameters=str(payload["regularized_parameters"]),
            lambda_reg=float(payload["lambda_reg"]),
            l2_formula=str(payload["l2_formula"]),
            inverse_temperature=float(payload["inverse_temperature"]),
            exact_reference=dict(payload["exact_reference"]),
            first_radial_derivative=dict(payload["first_radial_derivative"]),
            fingerprint=fingerprint,
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append(f"schema_version={self.schema_version}, expected 1")
        if self.objective_id != EXPECTED_OBJECTIVE_ID:
            errors.append(
                f"objective_id={self.objective_id!r}, "
                f"expected {EXPECTED_OBJECTIVE_ID!r}"
            )
        if self.loss != "binary_logistic":
            errors.append("loss must be binary_logistic")
        if self.label_encoding != "minus_one_plus_one":
            errors.append("labels must be encoded as -1/+1")
        if self.ce_reduction != "mean_over_training_examples":
            errors.append("cross entropy reduction must be the training mean")
        if self.gamma_ce != 100.0:
            errors.append("gamma_ce must equal 100.0")
        if self.regularizer != "l2":
            errors.append("regularizer must be l2")
        if (
            self.regularized_parameters
            != "all_trainable_parameters_including_biases"
        ):
            errors.append("the L2 norm must include every trainable bias")
        if self.lambda_reg != 1.0:
            errors.append("lambda_reg must equal 1.0")
        if self.l2_formula != EXPECTED_L2_FORMULA:
            errors.append(
                f"l2_formula must equal {EXPECTED_L2_FORMULA!r}"
            )
        if self.inverse_temperature != 1.0:
            errors.append("inverse_temperature must equal 1.0")
        if (
            self.exact_reference.get("classification_rule")
            != "signed_margin_strictly_positive"
        ):
            errors.append("exact references require strictly positive margins")
        if self.exact_reference.get("required_wrong_count") != 0:
            errors.append("exact references require zero wrong examples")
        if (
            self.exact_reference.get(
                "required_min_signed_margin_strictly_greater_than"
            )
            != 0.0
        ):
            errors.append("exact references require min signed margin > 0")
        if (
            self.exact_reference.get("checkpoint_policy")
            != EXPECTED_EXACT_CHECKPOINT_POLICY
        ):
            errors.append(
                "exact references must use the first full-data exact "
                "post-optimizer-update checkpoint"
            )
        if (
            self.exact_reference.get("first_exact_is_production_endpoint")
            is not True
        ):
            errors.append("first_exact must be the production endpoint")
        if (
            self.exact_reference.get(
                "optimizer_convergence_is_diagnostic_only"
            )
            is not True
        ):
            errors.append(
                "optimizer convergence must be diagnostic only"
            )
        if "first_exact_is_diagnostic_only" in self.exact_reference:
            errors.append(
                "the contradictory first_exact_is_diagnostic_only key is forbidden"
            )
        if (
            self.first_radial_derivative.get("method")
            != "direct_autograd_radial_score"
        ):
            errors.append("first radial derivative must use the direct score")
        if bool(
            self.first_radial_derivative.get(
                "finite_difference_fallback", True
            )
        ):
            errors.append("finite-difference first-derivative fallback is forbidden")
        if errors:
            raise ValueError("invalid DNN objective contract: " + "; ".join(errors))

    def ce_mean_numpy(
        self, logits: np.ndarray, labels_pm1: np.ndarray
    ) -> float:
        logits_array = np.asarray(logits, dtype=np.float64).reshape(-1)
        labels_array = np.asarray(labels_pm1, dtype=np.float64).reshape(-1)
        if logits_array.shape != labels_array.shape:
            raise ValueError(
                f"logit/label shape mismatch: "
                f"{logits_array.shape} vs {labels_array.shape}"
            )
        if logits_array.size == 0:
            raise ValueError("cross entropy cannot be evaluated on an empty set")
        if not np.all(np.isin(labels_array, (-1.0, 1.0))):
            raise ValueError("labels must contain only -1 and +1")
        if not np.all(np.isfinite(logits_array)):
            raise ValueError("logits contain non-finite values")
        signed_logits = labels_array * logits_array
        return float(np.logaddexp(0.0, -signed_logits).mean())

    def l2_penalty_numpy(self, theta: np.ndarray) -> float:
        theta_array = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta_array.size == 0:
            raise ValueError("theta cannot be empty")
        if not np.all(np.isfinite(theta_array)):
            raise ValueError("theta contains non-finite values")
        return float(
            self.lambda_reg
            * np.dot(theta_array, theta_array)
            / (2.0 * theta_array.size)
        )

    def total_numpy(
        self,
        logits: np.ndarray,
        labels_pm1: np.ndarray,
        theta: np.ndarray,
    ) -> tuple[float, float, float]:
        ce = self.ce_mean_numpy(logits, labels_pm1)
        l2 = self.l2_penalty_numpy(theta)
        total = self.gamma_ce * ce + l2
        return float(total), float(ce), float(l2)

    def l2_gradient_numpy(self, theta: np.ndarray) -> np.ndarray:
        theta_array = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta_array.size == 0:
            raise ValueError("theta cannot be empty")
        if not np.all(np.isfinite(theta_array)):
            raise ValueError("theta contains non-finite values")
        return self.lambda_reg * theta_array / float(theta_array.size)

    def exact_reference_metrics_numpy(
        self,
        logits: np.ndarray,
        labels_pm1: np.ndarray,
    ) -> dict[str, float | int | bool]:
        """Replay the strict, post-optimization exactness criterion."""

        logits_array = np.asarray(logits, dtype=np.float64).reshape(-1)
        labels_array = np.asarray(labels_pm1, dtype=np.float64).reshape(-1)
        if logits_array.shape != labels_array.shape or logits_array.size == 0:
            raise ValueError("logits and labels must share a non-empty shape")
        if not np.all(np.isin(labels_array, (-1.0, 1.0))):
            raise ValueError("labels must contain only -1 and +1")
        if not np.all(np.isfinite(logits_array)):
            raise ValueError("logits contain non-finite values")
        signed_margin = labels_array * logits_array
        wrong_count = int(np.count_nonzero(signed_margin <= 0.0))
        minimum = float(np.min(signed_margin))
        return {
            "wrong_count": wrong_count,
            "min_signed_margin": minimum,
            "exact": bool(wrong_count == 0 and minimum > 0.0),
        }

    def torch_terms(
        self,
        logits: Any,
        labels_pm1: Any,
        parameters: Iterable[Any],
    ) -> tuple[Any, Any, Any]:
        """Return ``(total, ce_mean, l2)`` without optimizer weight decay."""

        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("torch is required for torch_terms") from exc

        labels = labels_pm1.reshape_as(logits)
        if labels.numel() == 0:
            raise ValueError("cross entropy cannot be evaluated on an empty set")
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("logits contain non-finite values")
        if not bool(torch.isfinite(labels).all().item()) or not bool(
            torch.all((labels == -1) | (labels == 1)).item()
        ):
            raise ValueError("labels must contain only finite -1 and +1 values")
        ce_mean = functional.softplus(-labels * logits).mean()
        parameter_list = list(parameters)
        if not parameter_list:
            raise ValueError("the complete trainable parameter list is required")
        if any(
            not bool(torch.isfinite(parameter).all().item())
            for parameter in parameter_list
        ):
            raise ValueError("trainable parameters contain non-finite values")
        count = int(sum(int(parameter.numel()) for parameter in parameter_list))
        if count <= 0:
            raise ValueError("parameter count must be positive")
        squared_norm = sum(
            torch.sum(parameter.reshape(-1) ** 2)
            for parameter in parameter_list
        )
        l2 = float(self.lambda_reg) * squared_norm / (2.0 * float(count))
        total = float(self.gamma_ce) * ce_mean + l2
        return total, ce_mean, l2

    def metadata(self, *, parameter_count: int) -> dict[str, Any]:
        if int(parameter_count) <= 0:
            raise ValueError("parameter_count must be positive")
        return {
            "objective_id": self.objective_id,
            "objective_fingerprint": self.fingerprint,
            "loss": self.loss,
            "ce_reduction": self.ce_reduction,
            "gamma_ce": self.gamma_ce,
            "lambda_reg": self.lambda_reg,
            "parameter_count": int(parameter_count),
            "l2_denominator": int(2 * int(parameter_count)),
            "regularized_parameters": self.regularized_parameters,
            "first_radial_derivative_method": (
                self.first_radial_derivative["method"]
            ),
        }
