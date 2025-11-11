import numpy as np
import shap
from shap_wrapper import make_model_predict_fn


class SHAPExplainer:
    def __init__(self, model, background_data_size=100, device="cpu"):
        self.model = model
        self.background_data_size = background_data_size
        self.background_data = None
        self.device = device
        self.explainers = {}  # cache explainers per task

    def prepare_background_data(self, sample_features):
        """Prepare a background dataset for SHAP using representative samples (numpy array)."""
        sample_features = np.asarray(sample_features)
        n_samples, n_feats = sample_features.shape
        if n_samples <= self.background_data_size:
            self.background_data = sample_features.copy()
        else:
            # sample without replacement
            idx = np.random.choice(n_samples, self.background_data_size, replace=False)
            self.background_data = sample_features[idx]

    def explain_prediction(self, features, task_id, feature_names, apply_sigmoid=True, tol=1e-6):
        """
        Explain prediction(s) using shap.Explainer with a numpy predict function.

        - features: numpy array shape (n_samples, n_features)
        - task_id: 'transaction' or 'account'
        - feature_names: list of feature names
        Returns dict with expected_value, shap_values (array), and feature importance list.
        """
        X = np.asarray(features, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Prepare background if missing
        if self.background_data is None:
            # Use the provided features as background fallback
            self.prepare_background_data(X)

        # Create predict function for this task
        predict_fn = make_model_predict_fn(self.model, device=self.device, task_id=task_id, apply_sigmoid=apply_sigmoid)

        # Cache explainer per task/settings key
        cache_key = (task_id, apply_sigmoid)
        if cache_key not in self.explainers:
            # Use shap.Explainer which will pick an appropriate algorithm
            try:
                self.explainers[cache_key] = shap.Explainer(predict_fn, self.background_data, feature_names=feature_names)
            except Exception:
                # Fallback: let explainer infer feature names
                self.explainers[cache_key] = shap.Explainer(predict_fn, self.background_data)

        explainer = self.explainers[cache_key]

        # Compute SHAP values
        shap_values = explainer(X)

        # Extract arrays and expected/base value robustly
        values = np.asarray(shap_values.values)

        # shap.Explainer may return different explainer types; expected/base value
        # can be on the explainer or on the returned Explanation as base_values.
        expected_value = None
        if hasattr(explainer, 'expected_value'):
            expected_value = np.asarray(explainer.expected_value)
        else:
            # try to get base_values from the returned explanation
            if hasattr(shap_values, 'base_values'):
                expected_value = np.asarray(shap_values.base_values)
            elif hasattr(shap_values, 'expected_value'):
                expected_value = np.asarray(shap_values.expected_value)

        # Normalize values shape for common cases:
        # - (n_samples, n_features)
        # - (n_samples, 1, n_features) -> squeeze middle dim
        # - (n_samples, n_outputs, n_features) -> if n_outputs==1 squeeze to (n_samples, n_features)
        if values.ndim == 3 and values.shape[1] == 1:
            values = values[:, 0, :]
        elif values.ndim == 3 and values.shape[1] > 1:
            # multi-output explanation: pick first output
            values = values[:, 0, :]
        elif values.ndim == 1:
            # single value per sample -> reshape to (n_samples, 1)
            values = values.reshape(-1, 1)

        # Reconstruct predictions from SHAP and compare
        preds = predict_fn(X)
        # Derive a scalar expected value for reconstruction.
        expected_scalar = 0.0
        if expected_value is None:
            # unknown expected value — leave as 0 and warn
            print("[SHAP WARNING] Không tìm được expected/base value trong explainer; sẽ dùng 0 làm baseline.")
            expected_scalar = 0.0
        else:
            # expected_value may be scalar, per-sample, or per-output
            try:
                ev = np.asarray(expected_value)
                if ev.size == 1:
                    expected_scalar = float(ev.ravel()[0])
                elif ev.ndim == 1 and ev.shape[0] == values.shape[1]:
                    # per-feature? fallback to mean
                    expected_scalar = float(ev.mean())
                elif ev.ndim == 1 and ev.shape[0] == X.shape[0]:
                    # per-sample base values — take first sample's base
                    expected_scalar = float(ev.ravel()[0])
                elif ev.ndim == 2:
                    # (n_samples, outputs) or (n_samples, 1) -> take first element
                    expected_scalar = float(ev.ravel()[0])
                else:
                    expected_scalar = float(ev.ravel()[0])
            except Exception:
                expected_scalar = float(np.asarray(expected_value).ravel()[0])

        recon = expected_scalar + np.sum(values, axis=1)
        max_diff = float(np.max(np.abs(preds - recon)))

        if max_diff > max(tol, 1e-3):
            # Include diagnostic info
            print(f"[SHAP WARNING] Additivity check failed. Max diff={max_diff:.6g}. This may be due to using logits vs probabilities or unsupported ops.")

        # Create feature importance list for first sample
        fi = []
        first_shap = values[0]
        for name, shap_v, feat_v in zip(feature_names, first_shap, X[0]):
            fi.append({
                "feature_name": name,
                "shap_value": float(shap_v),
                "feature_value": float(feat_v)
            })
        fi.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "expected_value": expected_scalar,
            "max_additivity_diff": max_diff,
            "preds": preds.tolist(),
            "feature_importance": fi,
            "raw_shap_values": values.tolist()
        }