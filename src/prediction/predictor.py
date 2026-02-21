import numpy as np

def predict_quantiles(load: np.ndarray, eta_min: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Predict arrival-time waiting quantiles (P50, P90) in minutes for each store.

    Parameters
    ----------
    load : np.ndarray
        Normalized load proxy per store in [0, 1], shape (N,).
    eta_min : float
        ETA/horizon in minutes (kept for interface; set by caller).

    Returns
    -------
    (W_p50, W_p90) : tuple[np.ndarray, np.ndarray]
        Predicted waiting time at arrival (P50, P90), shape (N,).
    """
    load = np.clip(load, 0.0, 1.0)
    # Simple, monotone placeholder. Replace with a trained quantile model.
    W_p50 = 2.0 + 8.0 * load
    W_p90 = W_p50 + (1.0 + 6.0 * load)
    return W_p50, W_p90
