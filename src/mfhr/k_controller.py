def compute_K(K0: float, lambda_: float, delta_j: float, Kmax: float) -> float:
    """Map predictive uncertainty to MFHR displacement tolerance K (minutes).

    K_j = K0 + lambda * delta_j
    K_j = min(K_j, Kmax)
    """
    K = float(K0) + float(lambda_) * float(delta_j)
    return min(K, float(Kmax))
