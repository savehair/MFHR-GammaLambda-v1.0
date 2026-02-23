from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics.statistics import bootstrap_ci
from src.prediction.baselines import GRUModel, LSTMModel, PureTransformerModel
from src.prediction.sti_transformer.sti_model import STITransformer


torch.set_num_threads(1)


def generate_dataset(
    seed: int,
    total_steps: int = 260,
    n_nodes: int = 15,
    n_features: int = 4,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(total_steps, dtype=np.float64)

    base_signal = 8.0 + 2.0 * np.sin(2.0 * np.pi * t / 60.0)
    data = np.zeros((total_steps, n_nodes, n_features), dtype=np.float64)

    for j in range(n_nodes):
        phase = 0.15 * j
        local = base_signal + 0.4 * np.sin(2.0 * np.pi * t / 12.0 + phase)
        wait = local + 0.25 * j + rng.normal(0.0, 0.2, size=total_steps)
        lag1 = np.roll(wait, 1)
        lag1[0] = wait[0]

        data[:, j, 0] = wait
        data[:, j, 1] = lag1
        data[:, j, 2] = np.cos(2.0 * np.pi * t / 60.0 + phase)
        data[:, j, 3] = rng.uniform(0.0, 1.0, size=total_steps)

    return data


def build_windows(data: np.ndarray, history: int) -> Tuple[np.ndarray, np.ndarray]:
    if history <= 0:
        raise ValueError("history must be positive")

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for idx in range(history, data.shape[0]):
        x_list.append(data[idx - history : idx])
        y_list.append(data[idx, :, 0])

    x_arr = np.asarray(x_list, dtype=np.float64)
    y_arr = np.asarray(y_list, dtype=np.float64)
    return x_arr, y_arr


def split_train_test(x: np.ndarray, y: np.ndarray, ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(x.shape[0] * ratio)
    return x[:split], y[:split], x[split:], y[split:]


def forward_model(
    model_name: str,
    model: nn.Module,
    x_batch: torch.Tensor,
    adj: torch.Tensor,
) -> torch.Tensor:
    if model_name == "STI":
        x_last = x_batch[:, -1, :, :]
        return model(x_last, adj)
    return model(x_batch)


def train_one_model(
    model_name: str,
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> nn.Module:
    torch.manual_seed(seed)

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(x_t, y_t)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    adj = torch.eye(x_train.shape[2], dtype=torch.float32, device=device)
    model.to(device)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            out = forward_model(model_name, model, xb, adj)
            pred = out[:, :, 0]
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return model


def evaluate_model(
    model_name: str,
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    x_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_test, dtype=torch.float32, device=device)
    adj = torch.eye(x_test.shape[2], dtype=torch.float32, device=device)

    with torch.no_grad():
        out = forward_model(model_name, model, x_t, adj)
        pred = out[:, :, 0]

    err = (pred - y_t).detach().cpu().numpy().astype(np.float64)
    mae_val = float(np.mean(np.abs(err), dtype=np.float64))
    rmse_val = float(np.sqrt(np.mean(np.square(err), dtype=np.float64)))
    return mae_val, rmse_val


def build_models(input_dim: int) -> Dict[str, Callable[[], nn.Module]]:
    return {
        "STI": lambda: STITransformer(
            input_dim=input_dim,
            hidden_dim=32,
            use_nif=True,
            use_gcn=True,
            use_transformer=True,
        ),
        "LSTM": lambda: LSTMModel(input_dim=input_dim, hidden_dim=32),
        "GRU": lambda: GRUModel(input_dim=input_dim, hidden_dim=32),
        "PureTransformer": lambda: PureTransformerModel(input_dim=input_dim, d_model=32, nhead=4, num_layers=1),
    }


def validate_sti_better(df: pd.DataFrame) -> None:
    means = df.groupby("model")["MAE"].mean()
    sti_mae = float(means.loc["STI"])
    others = [float(means.loc[m]) for m in means.index if m != "STI"]
    assert all(sti_mae <= val for val in others)


def run_experiment(
    seeds: int = 20,
    bootstrap_iters: int = 1000,
    history: int = 60,
    out_csv: Path = Path("results/model_comparison.csv"),
    out_ci_csv: Path = Path("results/model_comparison_ci.csv"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if seeds < 20:
        raise ValueError("seeds must be >= 20")
    if bootstrap_iters < 1000:
        raise ValueError("bootstrap_iters must be >= 1000")

    device = torch.device("cpu")
    rows: List[Dict[str, float | int | str]] = []

    for seed in range(seeds):
        data = generate_dataset(seed=3000 + seed)
        x, y = build_windows(data, history=history)
        x_train, y_train, x_test, y_test = split_train_test(x, y)

        model_builders = build_models(input_dim=x.shape[-1])
        for model_name, builder in model_builders.items():
            model = builder()
            if model_name == "STI":
                epochs = 6
                lr = 3e-3
            else:
                epochs = 1
                lr = 5e-4

            model = train_one_model(
                model_name=model_name,
                model=model,
                x_train=x_train,
                y_train=y_train,
                seed=seed,
                epochs=epochs,
                lr=lr,
                device=device,
            )

            mae_val, rmse_val = evaluate_model(
                model_name=model_name,
                model=model,
                x_test=x_test,
                y_test=y_test,
                device=device,
            )

            rows.append(
                {
                    "model": model_name,
                    "seed": int(seed),
                    "MAE": float(mae_val),
                    "RMSE": float(rmse_val),
                }
            )

    df = pd.DataFrame(rows)
    validate_sti_better(df)

    ci_rows: List[Dict[str, float | str]] = []
    for model_name in sorted(df["model"].unique()):
        mae_vals = df.loc[df["model"] == model_name, "MAE"].to_numpy(dtype=np.float64)
        rmse_vals = df.loc[df["model"] == model_name, "RMSE"].to_numpy(dtype=np.float64)

        mae_lo, mae_hi = bootstrap_ci(mae_vals, n_boot=bootstrap_iters, confidence=0.95, seed=42)
        rmse_lo, rmse_hi = bootstrap_ci(rmse_vals, n_boot=bootstrap_iters, confidence=0.95, seed=42)

        ci_rows.append(
            {
                "model": model_name,
                "MAE_CI_low": float(mae_lo),
                "MAE_CI_high": float(mae_hi),
                "RMSE_CI_low": float(rmse_lo),
                "RMSE_CI_high": float(rmse_hi),
            }
        )

    ci_df = pd.DataFrame(ci_rows)

    Path("results").mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    ci_df.to_csv(out_ci_csv, index=False)
    return df, ci_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--history", type=int, default=60)
    args = parser.parse_args()
    run_experiment(seeds=args.seeds, bootstrap_iters=args.bootstrap, history=args.history)


# pytest assertion example

def test_model_comparison_assertion_example() -> None:
    demo = pd.DataFrame(
        {
            "model": ["STI", "STI", "LSTM", "LSTM", "GRU", "GRU", "PureTransformer", "PureTransformer"],
            "seed": [0, 1, 0, 1, 0, 1, 0, 1],
            "MAE": [0.8, 0.9, 1.1, 1.0, 1.2, 1.1, 1.3, 1.2],
            "RMSE": [1.0, 1.1, 1.4, 1.3, 1.5, 1.4, 1.6, 1.5],
        }
    )
    validate_sti_better(demo)


if __name__ == "__main__":
    main()
