import numpy as np

def jain_index(waits):
    waits = np.array(waits)
    u = 1 / (1 + waits)
    return (np.sum(u) ** 2) / (len(u) * np.sum(u ** 2))