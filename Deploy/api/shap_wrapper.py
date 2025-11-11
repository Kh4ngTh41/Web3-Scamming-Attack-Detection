import numpy as np
import torch


def make_model_predict_fn(model, device="cpu", task_id="transaction", apply_sigmoid=True):
    """Return a predict function that accepts a numpy array X (n_samples, n_features)
    and returns a 1D numpy array of model outputs (probabilities if apply_sigmoid=True).

    The returned function is suitable for shap.Explainer which expects a numpy callable.
    """
    model.to(device)
    model.eval()

    def predict_fn(x_numpy: np.ndarray):
        # Ensure 2D
        x_arr = np.asarray(x_numpy, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        with torch.no_grad():
            xt = torch.from_numpy(x_arr).to(device)
            out = model(xt, task_id=task_id)
            # out may be a tensor of shape (n,1) or (n,)
            if isinstance(out, tuple) or isinstance(out, list):
                out = out[0]
            out = out.squeeze()
            if apply_sigmoid:
                out = torch.sigmoid(out)
            out_np = out.cpu().numpy()
            # Ensure shape (n_samples,)
            return np.asarray(out_np).reshape(-1)

    return predict_fn
