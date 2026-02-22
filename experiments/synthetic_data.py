import numpy as np

def generate_synthetic_dataset(T=2000, N=5, F=4, seed=42):
    rng = np.random.default_rng(seed)

    data = []
    base = np.sin(np.linspace(0, 20, T)) * 5 + 10

    for t in range(T):
        stores = []
        for j in range(N):
            noise = rng.normal(0, 1)
            wait = base[t] + j * 0.5 + noise
            features = np.array([
                wait,
                np.cos(t / 20),
                np.sin(t / 15),
                rng.uniform(0, 1)
            ])
            stores.append(features)
        data.append(stores)

    data = np.array(data)  # [T, N, F]

    # 时间顺序划分
    train_end = int(0.7 * T)
    val_end = int(0.85 * T)

    return (
        data[:train_end],
        data[train_end:val_end],
        data[val_end:]
    )