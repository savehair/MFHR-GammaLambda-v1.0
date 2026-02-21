import numpy as np

def choose_model(W_p50: np.ndarray, W_p90: np.ndarray, gamma: float) -> tuple[int, np.ndarray]:
    """User risk-aware choice:
        j* = argmin_j ( W_p50[j] + gamma * (W_p90[j] - W_p50[j]) )

    gamma = 0 -> risk-neutral
    gamma > 0 -> risk-averse
    """
    W_p50 = np.asarray(W_p50, dtype=float)
    W_p90 = np.asarray(W_p90, dtype=float)
    delta = W_p90 - W_p50
    score = W_p50 + float(gamma) * delta
    j_star = int(np.argmin(score))
    return j_star, delta
