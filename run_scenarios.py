"""
Run all 6 empirically coherent scenarios and compare core metrics.

v2 changelog (post code-review improvements):
- ADD: Multiprocessing-based parallel execution (25-30x speedup on 32-core systems)
- ADD: CLI arguments (--n-mc, --horizon, --workers, --quick, --output, --seed)
- ADD: Progress display (uses tqdm if available; falls back otherwise)
- ADD: 6-panel plot — including new v2 baseline dynamics (IPP exit rate, AI Lerner, regulator activation rate)
- ADD: MC 10-90 percentile confidence bands (HHI, AI share)
- ADD: Reproducibility metadata (run_config.json saved automatically)
- ADD: Per-proposition diagnostic proxy tests (P1, P3)
- FIX: sys.path handling (supports both flat and nested folder layouts)
- FIX: Migrated matplotlib colormap to new API (plt.colormaps[...])

Usage:
    python run_scenarios.py                            # default: 5 MC × 60 months
    python run_scenarios.py --quick                    # quick demo: 3 MC × 24 months
    python run_scenarios.py --n-mc 200 --horizon 120   # v3 recommended spec
    python run_scenarios.py --workers 8                # 8-core parallel
    python run_scenarios.py --workers -1               # use all cores
    python run_scenarios.py --output ./my_results      # specify output location
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

# Path handling: try local first (flat layout), then src/ (nested layout)
_HERE = Path(__file__).resolve().parent
for _candidate in [_HERE, _HERE.parent / "src", _HERE.parent]:
    if (_candidate / "baseline_model.py").exists():
        sys.path.insert(0, str(_candidate))
        break

warnings.filterwarnings("ignore", category=FutureWarning, module="mesa")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline_model import (
    ElectricityMarketABM, ModelParameters,
    SCENARIOS_6CELL,
)

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ============================================================================
#  PARALLEL WORKER
# ============================================================================


def _run_single(args: tuple) -> pd.DataFrame:
    """
    Worker function for parallel execution. Must be at module level (picklable).
    """
    scen, mc_index, horizon, seed_base = args
    # abs() to ensure non-negative seed
    seed = seed_base + abs(hash((scen.code, mc_index))) % 10_000
    model = ElectricityMarketABM(
        scenario=scen,
        params=ModelParameters(),
        horizon_months=horizon,
        seed=seed,
    )
    df = model.run()
    df["mc_index"] = mc_index
    return df


def run_sweep(
    n_mc: int,
    horizon: int,
    seed_base: int = 42,
    workers: int = 1,
) -> pd.DataFrame:
    """
    Run the 6-cell × n_mc sweep, optionally in parallel.

    workers=-1 → use all cores (os.cpu_count())
    """
    if workers == -1:
        workers = os.cpu_count() or 1

    tasks = [
        (scen, mc, horizon, seed_base)
        for scen in SCENARIOS_6CELL
        for mc in range(n_mc)
    ]
    total = len(tasks)

    results: list[pd.DataFrame] = []
    if workers > 1:
        with mp.Pool(workers) as pool:
            iterator = pool.imap_unordered(_run_single, tasks)
            if HAS_TQDM:
                iterator = tqdm(iterator, total=total, desc="MC runs", unit="run")
                results = list(iterator)
            else:
                for i, df in enumerate(iterator, 1):
                    results.append(df)
                    if i % max(1, total // 10) == 0 or i == total:
                        print(f"  Progress: {i}/{total} runs ({100*i/total:.0f}%)")
    else:
        # Single-threaded path
        iterator = tasks
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="MC runs", unit="run")
        for i, task in enumerate(iterator, 1):
            results.append(_run_single(task))
            if not HAS_TQDM and (i % max(1, total // 10) == 0 or i == total):
                print(f"  Progress: {i}/{total} runs ({100*i/total:.0f}%)")

    return pd.concat(results, ignore_index=True)


# ============================================================================
#  ANALYSIS
# ============================================================================


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats from final-month observations across MC."""
    final_month = df["month"].max()
    final = df[df["month"] == final_month]
    g = final.groupby("scenario")
    summary = g.agg(
        hhi_mean=("hhi", "mean"),
        hhi_std=("hhi", "std"),
        ai_lerner_mean=("ai_lerner", "mean"),
        ai_share_mean=("ai_market_share", "mean"),
        clearing_price_mean=("clearing_price", "mean"),
        ipp_exit_rate_mean=("ipp_exit_rate", "mean"),
        regulator_intervened_pct=("regulator_intervened", "mean"),
    ).round(3)
    # Order by scenario code for consistent reading
    order = [s.code for s in SCENARIOS_6CELL]
    summary = summary.reindex([s for s in order if s in summary.index])
    return summary


def comparative_tests(df: pd.DataFrame) -> dict:
    """
    Lightweight propositional proxy tests for baseline diagnostic.

    Note: These are quick proxies; full v3 propositional tests require PRIM, Sobol,
    PELT changepoint detection (see v3 §8). This function exists to give a quick
    sense of whether the baseline is producing scenario-differentiated dynamics.
    """
    final = df[df["month"] == df["month"].max()]
    results = {}

    # P1 proxy: HHI in High-compute vs Low-compute scenarios (axis A effect)
    high_A = final[final["scenario"].isin(["HLH", "HHL", "HHH"])]["hhi"]
    low_A = final[final["scenario"].isin(["LLL", "LLH", "LHH"])]["hhi"]
    if len(high_A) > 0 and len(low_A) > 0:
        results["P1_proxy_hhi_highA_minus_lowA"] = round(
            float(high_A.mean() - low_A.mean()), 3
        )

    # P3 proxy: HHL (ex-post regulation) vs HHH (ex-ante regulation), same A and B
    hhl = final[final["scenario"] == "HHL"]["hhi"]
    hhh = final[final["scenario"] == "HHH"]["hhi"]
    if len(hhl) > 0 and len(hhh) > 0:
        results["P3_proxy_HHL_minus_HHH_hhi"] = round(float(hhl.mean() - hhh.mean()), 3)

    # AI Lerner index: high-A vs low-A
    high_A_l = final[final["scenario"].isin(["HLH", "HHL", "HHH"])]["ai_lerner"]
    low_A_l = final[final["scenario"].isin(["LLL", "LLH", "LHH"])]["ai_lerner"]
    if len(high_A_l) > 0 and len(low_A_l) > 0:
        results["AI_lerner_highA_minus_lowA"] = round(
            float(high_A_l.mean() - low_A_l.mean()), 3
        )

    return results


def plot_trajectories(df: pd.DataFrame, out: Path) -> None:
    """6-panel plot with MC mean and 10-90 percentile bands."""
    g = df.groupby(["scenario", "month"])
    agg = g.agg(
        hhi_mean=("hhi", "mean"),
        hhi_low=("hhi", lambda x: x.quantile(0.10)),
        hhi_high=("hhi", lambda x: x.quantile(0.90)),
        share_mean=("ai_market_share", "mean"),
        share_low=("ai_market_share", lambda x: x.quantile(0.10)),
        share_high=("ai_market_share", lambda x: x.quantile(0.90)),
        price_mean=("clearing_price", "mean"),
        lerner_mean=("ai_lerner", "mean"),
        exit_mean=("ipp_exit_rate", "mean"),
        reg_mean=("regulator_intervened", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    palette = plt.colormaps["tab10"]
    scenarios = [s.code for s in SCENARIOS_6CELL]

    panels = [
        ("hhi", "HHI (Market Concentration)", "HHI", True),
        ("share", "AI Firms' Market Share", "Share", True),
        ("price", "Clearing Price ($/MWh)", "$/MWh", False),
        ("lerner", "AI Lerner Index", "(p − MC)/p", False),
        ("exit", "IPP Exit Rate", "Fraction exited", False),
        ("reg", "Regulator Intervention Rate", "Fraction (MC)", False),
    ]

    for ax, (key, title, ylabel, has_band) in zip(axes.flat, panels):
        for i, code in enumerate(scenarios):
            sub = agg[agg["scenario"] == code]
            color = palette(i)
            ax.plot(sub["month"], sub[f"{key}_mean"], label=code, color=color, lw=1.5)
            if has_band:
                ax.fill_between(
                    sub["month"], sub[f"{key}_low"], sub[f"{key}_high"],
                    alpha=0.15, color=color,
                )
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if key == "hhi":
            ax.axhline(1800, ls="--", color="grey", alpha=0.5, lw=1, label="_DOJ 1800")
            ax.axhline(3000, ls="--", color="red", alpha=0.5, lw=1, label="_DOJ 3000")
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out}")


# ============================================================================
#  CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run 6-cell scenario sweep for the ABM baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-mc", type=int, default=5,
                        help="Monte Carlo replications per scenario")
    parser.add_argument("--horizon", type=int, default=60,
                        help="Simulation horizon in months")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (1 = single thread; -1 = all cores)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: ./results)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick demo mode: 3 MC × 24 months")
    args = parser.parse_args()

    if args.quick:
        args.n_mc = 3
        args.horizon = 24

    output_dir = args.output or (Path.cwd() / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_workers_effective = os.cpu_count() if args.workers == -1 else args.workers

    print("=" * 60)
    print("ABM 6-CELL SCENARIO SWEEP (run_scenarios v2)")
    print("=" * 60)
    print(f"  Scenarios:  {[s.code for s in SCENARIOS_6CELL]}")
    print(f"  MC reps:    {args.n_mc} per cell × 6 = {6 * args.n_mc} total runs")
    print(f"  Horizon:    {args.horizon} months")
    print(f"  Workers:    {n_workers_effective}  (use --workers -1 for all {os.cpu_count()} cores)")
    print(f"  Output:     {output_dir}")
    print(f"  tqdm:       {'available' if HAS_TQDM else 'not installed (no progress bar)'}")
    print("=" * 60)

    # Save run config BEFORE running, so it survives crashes
    config = {
        "n_mc": args.n_mc,
        "horizon": args.horizon,
        "seed": args.seed,
        "workers": n_workers_effective,
        "scenarios": [s.code for s in SCENARIOS_6CELL],
        "timestamp_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
    }
    config_path = output_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    t0 = time.time()
    df = run_sweep(args.n_mc, args.horizon, args.seed, args.workers)
    elapsed = time.time() - t0

    # Update config with elapsed time
    config["elapsed_seconds"] = round(elapsed, 2)
    config["timestamp_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSweep complete: {elapsed:.1f}s ({len(df):,} rows)")

    # --- Save results ---
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    summary = summarize(df)
    summary_path = output_dir / "results_summary.csv"
    summary.to_csv(summary_path)

    tests = comparative_tests(df)
    tests_path = output_dir / "results_tests.json"
    with open(tests_path, "w") as f:
        json.dump(tests, f, indent=2)

    plot_path = output_dir / "results_plot.png"
    plot_trajectories(df, plot_path)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("FINAL-MONTH CROSS-SCENARIO SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    print("\n" + "=" * 60)
    print("DIAGNOSTIC PROXY TESTS")
    print("=" * 60)
    print("  (Quick proxies; full v3 tests require PRIM/Sobol — see v3 §8)")
    for k, v in tests.items():
        sign = "+" if v >= 0 else ""
        print(f"  {k:48s}: {sign}{v}")

    print("\n" + "=" * 60)
    print("BASELINE EXPECTATIONS (when M2 is disabled)")
    print("=" * 60)
    print("  - HHI ~ 700-800 (below the 1800 threshold when M2 is disabled — expected)")
    print("  - HHI of High-A scenarios should exceed Low-A (positive P1 proxy)")
    print("  - IPP exit rate should be active (new v2 dynamic)")
    print("  - With M2 enabled, expect HHI tipping to emerge — supports full P1 testing")

    print(f"\nOutputs in: {output_dir}")


if __name__ == "__main__":
    main()
