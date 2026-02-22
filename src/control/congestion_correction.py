def dynamic_lambda(lam, predicted_p90, threshold=10, alpha=0.5):
    if predicted_p90 > threshold:
        return lam * (1 + alpha)
    return lam