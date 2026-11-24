import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog

warnings.filterwarnings("ignore")
from arch import arch_model  # noqa: E402

TAUS = (0.05, 0.01)
DEFAULT_WINDOW = 250
BENCHMARKS = ["HS", "nGARCH", "tGARCH"]
# model name -> (time-lag reading, bandwidth multiplier)
LLQAR_VARIANTS = {
    "LLQAR": ("inflate", 1.0),
    "LLQAR-literal": ("literal", 1.0),
    "LLQAR-bw0.5": ("inflate", 0.5),
    "LLQAR-bw2": ("inflate", 2.0),
}
MODELS = BENCHMARKS + list(LLQAR_VARIANTS)

# Calendar crisis windows (inclusive); every other day is labelled "Other"
REGIMES = {
    "2018 crypto crash": ("2018-01-01", "2018-12-31"),
    "COVID crash": ("2020-02-15", "2020-06-30"),
    "2021 bubble": ("2021-01-01", "2021-12-31"),
    "2022 collapse": ("2022-01-01", "2022-12-31"),
}


def load_losses(path):
    prices_table = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    prices = prices_table.set_index("Date")["Close"].astype(float).dropna()
    return -np.log(prices / prices.shift(1)).dropna()


def fit_garch(losses, distribution):
    model = arch_model(losses * 100, mean="Constant", vol="GARCH", p=1, q=1, dist=distribution)
    result = model.fit(disp="off", show_warning=False)
    next_day_variance = float(result.forecast(horizon=1, reindex=False).variance.values[-1, 0])
    mean = float(result.params["mu"]) / 100
    degrees_of_freedom = float(result.params["nu"]) if distribution == "t" else None
    return mean, np.sqrt(next_day_variance) / 100, degrees_of_freedom


def benchmark_forecasts(window_losses):
    forecasts = {("HS", tau): float(np.quantile(window_losses, 1 - tau)) for tau in TAUS}
    try:
        mean, sigma, _ = fit_garch(window_losses, "normal")
        for tau in TAUS:
            forecasts[("nGARCH", tau)] = mean + sigma * stats.norm.ppf(1 - tau)
    except Exception:
        for tau in TAUS:
            forecasts[("nGARCH", tau)] = np.nan
    try:
        mean, sigma, degrees_of_freedom = fit_garch(window_losses, "t")
        unit_variance_scale = np.sqrt((degrees_of_freedom - 2) / degrees_of_freedom)
        for tau in TAUS:
            forecasts[("tGARCH", tau)] = (
                mean + sigma * unit_variance_scale * stats.t.ppf(1 - tau, degrees_of_freedom)
            )
    except Exception:
        for tau in TAUS:
            forecasts[("tGARCH", tau)] = np.nan
    return forecasts


def weighted_quantile_regression(predictors, responses, weights, quantile_level):
    """Minimise sum of weights * check_loss(response - intercept - slope * predictor) as a linear program; return the intercept."""
    observation_count = len(responses)
    objective = np.concatenate([
        [0.0, 0.0],
        quantile_level * weights,
        (1 - quantile_level) * weights,
    ])
    equality_matrix = np.zeros((observation_count, 2 + 2 * observation_count))
    equality_matrix[:, 0] = 1.0
    equality_matrix[:, 1] = predictors
    equality_matrix[:, 2:2 + observation_count] = np.eye(observation_count)
    equality_matrix[:, 2 + observation_count:] = -np.eye(observation_count)
    bounds = [(None, None), (None, None)] + [(0, None)] * (2 * observation_count)
    solution = linprog(objective, A_eq=equality_matrix, b_eq=responses, bounds=bounds, method="highs")
    return solution.x[0] if solution.success else np.nan


def llqar_forecast(window_losses, weight_mode, bandwidth_multiplier):
    """Return {tau: VaR} for one window."""
    losses = np.asarray(window_losses)
    window_length = len(losses)
    current_loss = losses[-1]
    responses = losses[1:]
    predictors = losses[:-1]
    time_lag = np.arange(window_length - 1, 0, -1)
    centered_predictors = predictors - current_loss
    if weight_mode == "literal":
        time_factor = (time_lag + 1) / (2.0 * time_lag)
    else:
        time_factor = 1 + time_lag / (window_length - 1)
    distance = np.abs(centered_predictors) * time_factor
    interquartile_range = np.subtract(*np.percentile(distance, [75, 25]))
    spread = interquartile_range if interquartile_range > 0 else distance.std() + 1e-12
    bandwidth = bandwidth_multiplier * (4 / (3 * len(distance))) ** 0.2 * spread
    weights = np.exp(-0.5 * (distance / bandwidth) ** 2)
    weights = weights / weights.sum() * len(weights) + 1e-10
    return {
        tau: weighted_quantile_regression(centered_predictors, responses, weights, 1 - tau)
        for tau in TAUS
    }


def forecast_one_window(window_losses):
    forecasts = benchmark_forecasts(window_losses)
    for model, (weight_mode, bandwidth_multiplier) in LLQAR_VARIANTS.items():
        try:
            llqar_values = llqar_forecast(window_losses, weight_mode, bandwidth_multiplier)
        except Exception:
            llqar_values = {tau: np.nan for tau in TAUS}
        for tau in TAUS:
            forecasts[(model, tau)] = llqar_values[tau]
    return forecasts


def rolling_forecasts(losses, window, workers=None, max_steps=None):
    values, dates = losses.values, losses.index
    step_count = len(values) - window
    if max_steps:
        step_count = min(step_count, max_steps)
    windows = [values[step:step + window] for step in range(step_count)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(forecast_one_window, windows, chunksize=8))
    rows = []
    for step, result in enumerate(results):
        row = {"Date": dates[step + window], "Loss": values[step + window]}
        row.update({f"{model}|{tau}": result[(model, tau)] for model in MODELS for tau in TAUS})
        rows.append(row)
    return pd.DataFrame(rows).set_index("Date")


def kupiec_test(violations, tau):
    """Kupiec proportion-of-failures test; returns the p-value."""
    observation_count = len(violations)
    violation_count = int(violations.sum())
    if violation_count in (0, observation_count):
        alternative_log_likelihood = 0.0
    else:
        observed_rate = violation_count / observation_count
        alternative_log_likelihood = (
            violation_count * np.log(observed_rate)
            + (observation_count - violation_count) * np.log(1 - observed_rate)
        )
    null_log_likelihood = (
        violation_count * np.log(tau)
        + (observation_count - violation_count) * np.log(1 - tau)
    )
    statistic = -2 * (null_log_likelihood - alternative_log_likelihood)
    return float(1 - stats.chi2.cdf(statistic, 1))


def christoffersen_test(violations, tau):
    """Christoffersen conditional coverage test; returns the p-value."""
    sequence = violations.astype(int)
    previous, current = sequence[:-1], sequence[1:]
    no_to_no = int(((previous == 0) & (current == 0)).sum())
    no_to_yes = int(((previous == 0) & (current == 1)).sum())
    yes_to_no = int(((previous == 1) & (current == 0)).sum())
    yes_to_yes = int(((previous == 1) & (current == 1)).sum())

    def count_times_log(count, probability):
        return count * np.log(probability) if count > 0 else 0.0

    probability_after_no = no_to_yes / max(no_to_no + no_to_yes, 1)
    probability_after_yes = yes_to_yes / max(yes_to_no + yes_to_yes, 1)
    overall_probability = (no_to_yes + yes_to_yes) / max(
        no_to_no + no_to_yes + yes_to_no + yes_to_yes, 1
    )
    independence_alternative = (
        count_times_log(no_to_no, 1 - probability_after_no)
        + count_times_log(no_to_yes, probability_after_no)
        + count_times_log(yes_to_no, 1 - probability_after_yes)
        + count_times_log(yes_to_yes, probability_after_yes)
    )
    independence_null = (
        count_times_log(no_to_no + yes_to_no, 1 - overall_probability)
        + count_times_log(no_to_yes + yes_to_yes, overall_probability)
    )
    independence_statistic = -2 * (independence_null - independence_alternative)

    observation_count = len(sequence)
    violation_count = int(sequence.sum())
    if violation_count in (0, observation_count):
        coverage_alternative = 0.0
    else:
        observed_rate = violation_count / observation_count
        coverage_alternative = (
            violation_count * np.log(observed_rate)
            + (observation_count - violation_count) * np.log(1 - observed_rate)
        )
    coverage_null = (
        violation_count * np.log(tau)
        + (observation_count - violation_count) * np.log(1 - tau)
    )
    coverage_statistic = -2 * (coverage_null - coverage_alternative)
    return float(1 - stats.chi2.cdf(coverage_statistic + independence_statistic, 2))


def pinball_loss(losses, value_at_risk, tau):
    quantile_level = 1 - tau
    return float(np.mean((quantile_level - (losses < value_at_risk)) * (losses - value_at_risk)))


def backtest(forecasts):
    rows = []
    losses = forecasts["Loss"].values
    for model in MODELS:
        for tau in TAUS:
            value_at_risk = forecasts[f"{model}|{tau}"].values
            valid = np.isfinite(value_at_risk)
            valid_losses, valid_var = losses[valid], value_at_risk[valid]
            violations = valid_losses > valid_var
            rows.append({
                "model": model,
                "tau": tau,
                "observations": int(valid.sum()),
                "failed_fits": int((~valid).sum()),
                "violations": int(violations.sum()),
                "violation_rate": violations.mean(),
                "kupiec_p_value": kupiec_test(violations, tau),
                "christoffersen_p_value": christoffersen_test(violations, tau),
                "pinball_loss": pinball_loss(valid_losses, valid_var, tau),
            })
    return pd.DataFrame(rows)


def regime_of(date):
    for name, (start, end) in REGIMES.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return name
    return "Other"


def regime_table(forecasts):
    regime_labels = pd.Series([regime_of(date) for date in forecasts.index], index=forecasts.index)
    rows = []
    for regime in list(REGIMES) + ["Other"]:
        in_regime = (regime_labels == regime).values
        if in_regime.sum() == 0:
            continue
        for model in MODELS:
            for tau in TAUS:
                value_at_risk = forecasts[f"{model}|{tau}"].values[in_regime]
                losses = forecasts["Loss"].values[in_regime]
                valid = np.isfinite(value_at_risk)
                violations = losses[valid] > value_at_risk[valid]
                rows.append({
                    "regime": regime,
                    "model": model,
                    "tau": tau,
                    "observations": int(valid.sum()),
                    "violations": int(violations.sum()),
                    "violation_rate": violations.mean() if valid.any() else np.nan,
                    "pinball_loss": (
                        pinball_loss(losses[valid], value_at_risk[valid], tau) if valid.any() else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def make_plot(tag, forecasts, regimes, output_directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    tau = 0.05
    shown_models = ["HS", "tGARCH", "LLQAR", "LLQAR-literal"]
    colors = dict(zip(shown_models, ["#7f7f7f", "#1f77b4", "#d62728", "#ff9896"]))
    figure, axes = plt.subplots(2, 1, figsize=(13, 8))
    axes[0].plot(forecasts.index, forecasts["Loss"], color="0.82", lw=0.6, label="loss")
    for model in ("tGARCH", "LLQAR"):
        axes[0].plot(forecasts.index, forecasts[f"{model}|{tau}"], lw=1.1,
                     color=colors[model], label=f"{model} 95% VaR")
    axes[0].set_title(f"{tag}: losses and 95% VaR")
    axes[0].legend(fontsize=8, ncol=3, loc="upper left")

    regime_order = [name for name in list(REGIMES) + ["Other"] if name in set(regimes["regime"])]
    rates = regimes[regimes["tau"] == tau].pivot(index="regime", columns="model", values="violation_rate")
    rates = rates.loc[regime_order, shown_models]
    rates.plot.bar(ax=axes[1], width=0.8, color=[colors[model] for model in shown_models], legend=False)
    axes[1].axhline(tau, color="k", ls="--", lw=1)
    axes[1].set_ylabel("violation rate (target 5%)")
    axes[1].set_title("95% VaR violation rate by regime")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(handles=[Patch(color=colors[model], label=model) for model in shown_models],
                   fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    figure.tight_layout()
    figure.savefig(os.path.join(output_directory, f"{tag}_var.png"), dpi=130)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--window-crypto", type=int, default=None,
                        help="window override for BTC*/ETH* files, e.g. 365")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="limit forecast windows")
    arguments = parser.parse_args()

    os.makedirs(arguments.out, exist_ok=True)
    backtest_tables, regime_tables = [], []
    for filename in sorted(os.listdir(arguments.data)):
        if not filename.lower().endswith(".csv"):
            continue
        name = filename[:-4]
        is_crypto = name.upper().startswith(("BTC", "ETH"))
        window = arguments.window_crypto if (arguments.window_crypto and is_crypto) else arguments.window
        tag = f"{name}_w{window}"
        losses = load_losses(os.path.join(arguments.data, filename))
        print(f"== {tag}: {len(losses)} losses, kurtosis {stats.kurtosis(losses, fisher=False):.1f}")
        forecasts = rolling_forecasts(losses, window, arguments.workers, arguments.max_steps)
        forecasts.to_csv(os.path.join(arguments.out, f"{tag}_forecasts.csv"))
        backtest_table = backtest(forecasts)
        regime_summary = regime_table(forecasts)
        for table in (backtest_table, regime_summary):
            table.insert(0, "asset", name)
            table.insert(1, "window", window)
        backtest_table.to_csv(os.path.join(arguments.out, f"{tag}_backtest.csv"), index=False)
        regime_summary.to_csv(os.path.join(arguments.out, f"{tag}_regimes.csv"), index=False)
        make_plot(tag, forecasts, regime_summary, arguments.out)
        backtest_tables.append(backtest_table)
        regime_tables.append(regime_summary)
        print(backtest_table.round(4).to_string(index=False))
    if backtest_tables:
        pd.concat(backtest_tables).to_csv(os.path.join(arguments.out, "backtest_all.csv"), index=False)
        pd.concat(regime_tables).to_csv(os.path.join(arguments.out, "regimes_all.csv"), index=False)


if __name__ == "__main__":
    main()