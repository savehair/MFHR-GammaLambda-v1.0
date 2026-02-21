def online_update(theta: float, epsilon: float, eta: float) -> float:
    """Placeholder for online parameter update.

    theta_{t+1} = theta_t - eta * grad L(epsilon)

    Here we approximate grad L(epsilon) by epsilon (squared loss).
    Replace with your ML framework update if needed.
    """
    return float(theta) - float(eta) * float(epsilon)
