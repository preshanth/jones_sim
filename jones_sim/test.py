import pandas as pd
import matplotlib.pyplot as plt

# Your data
data = {
    "Ant": list(range(0, 27)),
    "Mean": [
        0.000,
        4.518,
        2.389,
        0.910,
        -3.482,
        -3.496,
        -4.459,
        3.720,
        1.063,
        2.093,
        -4.829,
        4.748,
        3.288,
        -2.870,
        -3.144,
        -3.277,
        -2.035,
        0.250,
        -0.605,
        -2.138,
        1.032,
        -3.625,
        -2.076,
        -1.350,
        -0.392,
        2.875,
        -3.008,
    ],
    "Std": [
        0.000,
        0.193,
        0.081,
        0.123,
        0.150,
        0.120,
        0.097,
        0.148,
        0.107,
        0.119,
        0.064,
        0.161,
        0.109,
        0.086,
        0.181,
        0.105,
        0.134,
        0.138,
        0.074,
        0.119,
        0.096,
        0.157,
        0.113,
        0.090,
        0.127,
        0.118,
        0.063,
    ],
    "CASA": [
        0.000,
        4.504,
        2.317,
        0.986,
        -3.443,
        -3.442,
        -4.417,
        3.664,
        1.011,
        2.080,
        -4.792,
        4.697,
        3.328,
        -2.874,
        -3.184,
        -3.168,
        -1.958,
        0.246,
        -0.677,
        -2.087,
        1.120,
        -3.607,
        -2.078,
        -1.339,
        -0.436,
        2.850,
        -3.003,
    ],
}

df = pd.DataFrame(data)
df["Diff"] = df["Mean"] - df["CASA"]
df["ci_lower"] = df["Mean"] - 1.96 * df["Std"]
df["ci_upper"] = df["Mean"] + 1.96 * df["Std"]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)

# --- Top panel: Solved + CASA delays ---
ax1.errorbar(
    df["Ant"],
    df["Mean"],
    yerr=[df["Mean"] - df["ci_lower"], df["ci_upper"] - df["Mean"]],
    fmt="o",
    capsize=4,
    label="Solved Delay (Mean ± 95% CI)",
    color="tab:blue",
)
ax1.plot(df["Ant"], df["CASA"], "s-", color="tab:orange", label="CASA Delay")
ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
ax1.set_ylabel("Delay (ns)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Bottom panel: Residuals ---
ax2.bar(df["Ant"], df["Diff"], color="tab:green")
ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
ax2.set_xlabel("Antenna")
ax2.set_ylabel("Residual\n(Solved - CASA)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
