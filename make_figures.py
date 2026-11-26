import argparse
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SURFACE = "#ffffff"
TEXT_PRIMARY = "#191919"
TEXT_SECONDARY = "#666666"
GRIDLINE = "#e7e7e7"
BASELINE = "#c2c2c2"

MODEL_COLORS = {
    "LLQAR": "#257de0",
    "tGARCH": "#f47230",
    "nGARCH": "#00b894",
    "HS": "#f2a600",
}
MODEL_LABELS = {
    "LLQAR": "LLQAR",
    "tGARCH": "t-GARCH",
    "nGARCH": "Normal GARCH",
    "HS": "Historical simulation",
}
ASSET_ORDER = ["BTC-USD", "ETH-USD", "SPX", "TSLA"]
ASSET_LABELS = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SPX": "S&P 500", "TSLA": "Tesla"}
ASSET_COLORS = dict(zip(ASSET_ORDER, ["#257de0", "#f47230", "#00b894", "#f2a600"]))
CONFIDENCE_LEVELS = [(0.05, "95% VaR (target 5%)"), (0.01, "99% VaR (target 1%)")]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "text.color": TEXT_PRIMARY,
})


def style_axis(axis):
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="y", color=GRIDLINE, linewidth=1)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", length=0)


def grouped_bars(rates, title, subtitle, output_path):
    """rates: DataFrame with columns asset, model, tau, violation_rate_percent."""
    models = list(MODEL_COLORS)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    bar_width = 0.19
    for axis, (tau, panel_title) in zip(axes, CONFIDENCE_LEVELS):
        panel = rates[rates["tau"] == tau]
        for model_index, model in enumerate(models):
            values = [
                panel[(panel["asset"] == asset) & (panel["model"] == model)]["violation_rate_percent"].iloc[0]
                for asset in ASSET_ORDER
            ]
            positions = [i + (model_index - 1.5) * bar_width for i in range(len(ASSET_ORDER))]
            axis.bar(positions, values, width=bar_width, color=MODEL_COLORS[model],
                     edgecolor=SURFACE, linewidth=1.5)
        axis.axhline(tau * 100, color=TEXT_PRIMARY, linewidth=1)
        axis.set_xticks(range(len(ASSET_ORDER)))
        axis.set_xticklabels([ASSET_LABELS[asset] for asset in ASSET_ORDER])
        axis.set_title(panel_title, loc="left", fontsize=11, color=TEXT_PRIMARY, pad=10)
        axis.set_ylabel("Violation rate (%)")
        style_axis(axis)
    handles = [Patch(color=MODEL_COLORS[model], label=MODEL_LABELS[model]) for model in models]
    handles.append(Line2D([0], [0], color=TEXT_PRIMARY, linewidth=1, label="Nominal target"))
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
                  bbox_to_anchor=(0.5, -0.01), labelcolor=TEXT_SECONDARY)
    figure.suptitle(title, x=0.01, ha="left", fontsize=13.5, fontweight="semibold", y=0.99)
    figure.text(0.01, 0.915, subtitle, ha="left", fontsize=9.5, color=TEXT_SECONDARY)
    figure.tight_layout(rect=(0, 0.07, 1, 0.9))
    figure.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def bandwidth_sensitivity(backtest, output_path):
    variants = [("LLQAR-bw0.5", "0.5x"), ("LLQAR", "1x (default)"), ("LLQAR-bw2", "2x")]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for axis, (tau, panel_title) in zip(axes, CONFIDENCE_LEVELS):
        for asset in ASSET_ORDER:
            values = [
                backtest[(backtest["asset"] == asset) & (backtest["model"] == model) & (backtest["tau"] == tau)
                         & (backtest["window"] == 250)]["violation_rate"].iloc[0] * 100
                for model, _ in variants
            ]
            axis.plot(range(len(variants)), values, color=ASSET_COLORS[asset], linewidth=2,
                      marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                      label=ASSET_LABELS[asset])
        axis.axhline(tau * 100, color=TEXT_PRIMARY, linewidth=1)
        axis.set_xticks(range(len(variants)))
        axis.set_xticklabels([label for _, label in variants])
        axis.set_xlim(-0.25, len(variants) - 0.75)
        axis.set_xlabel("Bandwidth multiplier")
        axis.set_ylabel("Violation rate (%)")
        axis.set_title(panel_title, loc="left", fontsize=11, color=TEXT_PRIMARY, pad=10)
        style_axis(axis)
    handles = [Line2D([0], [0], color=ASSET_COLORS[asset], linewidth=2, marker="o", markersize=8,
                      markeredgecolor=SURFACE, label=ASSET_LABELS[asset]) for asset in ASSET_ORDER]
    handles.append(Line2D([0], [0], color=TEXT_PRIMARY, linewidth=1, label="Nominal target"))
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
                  bbox_to_anchor=(0.5, -0.01), labelcolor=TEXT_SECONDARY)
    figure.suptitle("Wider LLQAR bandwidths bring coverage closer to target on every asset", x=0.01, ha="left",
                    fontsize=13.5, fontweight="semibold", y=0.99)
    figure.text(0.01, 0.915, "Full-sample violation rate, 250-day window; closer to the target line is better",
                ha="left", fontsize=9.5, color=TEXT_SECONDARY)
    figure.tight_layout(rect=(0, 0.07, 1, 0.9))
    figure.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-250", default="results_250", help="folder holding backtest_all.csv")
    parser.add_argument("--results-365", default="results_365", help="folder holding regimes_all.csv")
    parser.add_argument("--out", default="figures")
    arguments = parser.parse_args()
    os.makedirs(arguments.out, exist_ok=True)

    backtest = pd.read_csv(os.path.join(arguments.results_250, "backtest_all.csv"))
    backtest = backtest[backtest["window"] == 250]
    full_sample = backtest[backtest["model"].isin(MODEL_COLORS)].copy()
    full_sample["violation_rate_percent"] = full_sample["violation_rate"] * 100
    grouped_bars(
        full_sample,
        "Full-sample VaR violation rates",
        "Share of days the loss exceeded the forecast VaR; 250-day rolling window, forecasts from 2018 to November 2026",
        os.path.join(arguments.out, "violation_rates.png"),
    )

    regimes = pd.read_csv(os.path.join(arguments.results_365, "regimes_all.csv"))
    covid = regimes[(regimes["regime"] == "COVID crash") & regimes["model"].isin(MODEL_COLORS)].copy()
    covid["violation_rate_percent"] = covid["violation_rate"] * 100
    grouped_bars(
        covid,
        "VaR violation rates during the COVID crash (15 Feb to 30 Jun 2020)",
        "94 to 137 forecast days per asset; LLQAR has the highest rate at 99% on every asset, and at 95% on all but the S&P 500",
        os.path.join(arguments.out, "covid_regime.png"),
    )

    bandwidth_sensitivity(backtest, os.path.join(arguments.out, "bandwidth_sensitivity.png"))
    print("figures written to", arguments.out)


if __name__ == "__main__":
    main()
