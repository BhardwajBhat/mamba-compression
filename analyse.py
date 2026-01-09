import pandas as pd


def analyze_quant_results(csv_path):
    df = pd.read_csv(csv_path)

    quant_comparison = df.groupby("quant")["ppl"].mean().reset_index()
    print("--- Average Perplexity by Quantization Type ---")
    print(quant_comparison)
    print("-" * 40)

    layer_sensitivity = (
        df.groupby("layer_mode")["ppl"]
        .agg(["mean", "max", "std"])
        .sort_values(by="mean", ascending=False)
    )
    print("\n--- Layer Mode Sensitivity ---")
    print(layer_sensitivity)


def compare_svd_rdt(csv_path):
    df = pd.read_csv(csv_path)

    filtered_df = df[(df["rb"] == 16) & (df["rc"] == 16)].copy()
    summary = filtered_df.groupby(["quant", "rdt"])["ppl"].mean().unstack(level=0)

    if "fake" in summary.columns and "hadamard" in summary.columns:
        summary["ppl_gap_%"] = (
            (summary["fake"] - summary["hadamard"]) / summary["hadamard"]
        ) * 100

    print("--- PPL Comparison (Varying RDT, rb=16, rc=16) ---")
    print(summary.sort_index(ascending=False))  # Higher RDT first

    print("\n--- Sensitivity Analysis ---")
    for q_type in ["fake", "hadamard"]:
        if q_type in df["quant"].unique():
            subset = filtered_df[filtered_df["quant"] == q_type].sort_values("rdt")
            subset["diff"] = subset["ppl"].diff().abs()
            print(
                f"Max sensitivity for {q_type} quantization at RDT: {subset.loc[subset['diff'].idxmax(), 'rdt']}"
            )


def big_model(csv_path):
    df = pd.read_csv(csv_path)

    print("-" * 10)
    key_cols = ["bits", "group", "rb", "rc", "rdt", "layer_mode"]
    pivot = df.pivot_table(index=key_cols, columns="quant", values="ppl").dropna()
    pivot["delta_fake_minus_hadamard"] = pivot["fake"] - pivot["hadamard"]
    print(pivot.describe())

    from scipy import stats

    # 1. Perform the Paired T-Test
    # This tests the null hypothesis that the mean difference is zero.
    t_stat, p_value = stats.ttest_rel(pivot["fake"], pivot["hadamard"])

    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4e}")
    print("-" * 10)
    key_cols = ["group", "rb", "rc", "rdt", "layer_mode", "quant"]
    pivot = df.pivot_table(index=key_cols, columns="bits", values="ppl").dropna()
    pivot["delta_4_minus_8"] = pivot[4] - pivot[8]
    print(pivot.describe())

    print("-" * 10)
    key_cols = ["group", "rb", "rc", "rdt", "quant", "bits"]
    pivot = df.pivot_table(index=key_cols, columns="layer_mode", values="ppl").dropna()
    print(pivot.describe())

    print("-" * 10)
    key_cols = ["layer_mode", "rb", "rc", "rdt", "quant", "bits"]
    pivot = df.pivot_table(index=key_cols, columns="group", values="ppl").dropna()
    print(pivot.describe())

    print("-" * 10)
    print("Effect of RDT when RC and RB are 16:")
    filtered_df = df[(df["rc"] == 16) & (df["rb"] == 16)]
    key_cols = ["bits", "group", "layer_mode", "quant"]
    pivot_rdt = filtered_df.pivot_table(
        index=key_cols, columns="rdt", values="ppl"
    ).dropna()
    print(pivot_rdt.describe())

    print("-" * 10)
    print("Effect of RC when RDT is 160 and RB is 16:")
    filtered_df = df[(df["rb"] == 16) & (df["rdt"] == 160)]
    key_cols = ["bits", "group", "layer_mode", "quant"]
    pivot_rdt = filtered_df.pivot_table(
        index=key_cols, columns="rc", values="ppl"
    ).dropna()
    print(pivot_rdt.describe())

    print("-" * 10)
    print("Effect of RB when RDT is 160 and RC is 16:")
    filtered_df = df[(df["rc"] == 16) & (df["rdt"] == 160)]
    key_cols = ["bits", "group", "layer_mode", "quant"]
    pivot_rdt = filtered_df.pivot_table(
        index=key_cols, columns="rb", values="ppl"
    ).dropna()
    print(pivot_rdt.describe())


big_model("./hadamard_all_sweep_2.8b_optimized.csv")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def big_model_with_viz(csv_path):
    # Load data
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 22))
    axes = axes.flatten()

    # 1. Quantization Comparison (Delta: Fake - Hadamard)
    key_cols = ["bits", "group", "rb", "rc", "rdt", "layer_mode"]
    pivot_q = df.pivot_table(index=key_cols, columns="quant", values="ppl").dropna()
    pivot_q["delta_fake_minus_hadamard"] = pivot_q["fake"] - pivot_q["hadamard"]
    sns.boxplot(
        data=pivot_q, y="delta_fake_minus_hadamard", ax=axes[0], color="skyblue"
    )
    axes[0].set_title(
        "PPL Difference: Fake vs Hadamard\n(Higher Delta = Hadamard is better)",
        fontweight="bold",
    )
    axes[0].axhline(0, color="red", linestyle="--")

    # 2. Bits Comparison (Delta: 4-bit - 8-bit)
    key_cols = ["group", "rb", "rc", "rdt", "layer_mode", "quant"]
    pivot_b = df.pivot_table(index=key_cols, columns="bits", values="ppl").dropna()
    if 4 in pivot_b.columns and 8 in pivot_b.columns:
        pivot_b["delta_4_8"] = pivot_b[4] - pivot_b[8]
        sns.boxplot(data=pivot_b, y="delta_4_8", ax=axes[1], color="salmon")
        axes[1].set_title(
            "PPL Difference: 4-bit vs 8-bit\n(Precision Penalty)", fontweight="bold"
        )
        axes[1].axhline(0, color="red", linestyle="--")

    # 3. Layer Mode Effect
    sns.boxplot(x="layer_mode", y="ppl", data=df, ax=axes[2])
    axes[2].set_title("Impact of Layer Mode on PPL", fontweight="bold")

    # 4. Group Size Effect
    sns.boxplot(x="group", y="ppl", data=df, ax=axes[3])
    axes[3].set_title("Impact of Group Size on PPL", fontweight="bold")

    # 5. RDT effect when RC=16 and RB=16
    f_rdt = df[(df["rc"] == 16) & (df["rb"] == 16)]
    sns.lineplot(
        x="rdt",
        y="ppl",
        hue="bits",
        style="quant",
        data=f_rdt,
        ax=axes[4],
        marker="o",
        markersize=8,
    )
    axes[4].set_title("Effect of RDT (Fixed RC=16, RB=16)", fontweight="bold")

    # 6. RC effect when RDT=160 and RB=16
    f_rc = df[(df["rb"] == 16) & (df["rdt"] == 160)]
    sns.lineplot(
        x="rc",
        y="ppl",
        hue="bits",
        style="quant",
        data=f_rc,
        ax=axes[5],
        marker="s",
        markersize=8,
    )
    axes[5].set_title("Effect of RC (Fixed RDT=160, RB=16)", fontweight="bold")

    # 7. RB effect when RDT=160 and RC=16
    f_rb = df[(df["rc"] == 16) & (df["rdt"] == 160)]
    sns.lineplot(
        x="rb",
        y="ppl",
        hue="bits",
        style="quant",
        data=f_rb,
        ax=axes[6],
        marker="D",
        markersize=8,
    )
    axes[6].set_title("Effect of RB (Fixed RDT=160, RC=16)", fontweight="bold")

    # 8. Overall Distribution
    sns.violinplot(x="bits", y="ppl", hue="quant", data=df, split=True, ax=axes[7])
    axes[7].set_title("PPL Distribution by Bits and Quant", fontweight="bold")

    plt.tight_layout()
    plt.savefig("hadamard_analysis_summary.png", dpi=300)
    print("Analysis complete. Image saved as 'hadamard_analysis_summary.png'.")


# Execute
big_model_with_viz("./hadamard_all_sweep_2.8b_optimized.csv")
