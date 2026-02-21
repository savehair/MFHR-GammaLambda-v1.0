import numpy as np

def min_sum_awt(waiting_times) -> float:
    """Min-Sum AWT objective: tau = mean(waiting_times)."""
    x = np.asarray(waiting_times, dtype=float)
    return float(np.mean(x))
