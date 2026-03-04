import pandas as pd
from src.metrics.statistics import paired_t_test, cohens_d


def main() -> None:
    df = pd.read_csv("results/ablation_results.csv")

    required = {"Group", "MAE"}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(f"ablation_results.csv missing columns: {missing}")

    if "seed" not in df.columns:
        raise ValueError(
            "ablation_results.csv must contain a 'seed' column for paired significance tests."
        )

    pivot = df.pivot_table(index="seed", columns="Group", values="MAE", aggfunc="mean")

    if "A_Full" not in pivot.columns:
        raise ValueError("Group 'A_Full' not found in ablation_results.csv")

    baseline = pivot["A_Full"]
    comparisons = [g for g in ["B_NoNIF", "C_NoGCN", "D_NoTransformer"] if g in pivot.columns]

    if not comparisons:
        raise ValueError("No comparison groups found among B_NoNIF/C_NoGCN/D_NoTransformer")

    rows = []
    for group in comparisons:
        pair = pd.concat([baseline, pivot[group]], axis=1).dropna()
        if pair.empty:
            continue

        a = pair.iloc[:, 0].to_numpy()
        b = pair.iloc[:, 1].to_numpy()

        stats = paired_t_test(a, b)
        effect = cohens_d(a, b)

        rows.append(
            {
                "baseline": "A_Full",
                "compare": group,
                "n_pairs": int(len(pair)),
                "mean_A": float(a.mean()),
                "mean_B": float(b.mean()),
                "mean_diff_A_minus_B": float(stats["mean_diff"]),
                "p_value": float(stats["p_value"]),
                "cohens_d": float(effect),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv("results/significance_results.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
