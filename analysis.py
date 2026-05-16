"""
analysis.py — Comprehensive ABM analysis utilities.

Provides three main analysis routines, each parallelizable for 16-core systems:

1. run_mechanism_decomposition(): M1×M2×M3 = 2³ = 8 configurations
2. compare_clearing_rules(): Uniform vs Pay-as-bid robustness (v3 §8.5)
3. run_p3_stress(): P3 (regulatory irreversibility) test with aggressive M2 preset

Plus visualization helpers (plot_decomposition, plot_clearing_comparison, plot_p3_stress).

Usage from CLI:
    python analysis.py --analysis decomp --n-mc 30 --workers 16
    python analysis.py --analysis clearing --n-mc 30 --workers 16
    python analysis.py --analysis p3 --n-mc 30 --workers 16
    python analysis.py --analysis all --n-mc 30 --workers 16

Usage from Jupyter:
    from analysis import run_mechanism_decomposition, plot_decomposition
    df = run_mechanism_decomposition(scenario, n_mc=30, workers=16)
    plot_decomposition(df)

Author: Chankook Park (HUFS)
Version: analysis-0.1 (May 2026)
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
from typing import Optional

# Local import handling (flat directory layout)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline_model import (
    ElectricityMarketABM, ModelParameters,
    SCENARIOS_6CELL, ScenarioConfig,
    default_params, p3_stress_params, pay_as_bid_params,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ============================================================================
#  WORKER FUNCTIONS (must be at module level for pickling)
# ============================================================================


def _run_decomp_task(args: tuple) -> dict:
    """Worker for mechanism decomposition runs."""
    scenario, config_label, m1, m2, m3, mc_idx, horizon, seed_base, base_overrides = args
    params_dict = dict(default_params().__dict__)
    params_dict["beta"] = 0.4 if m1 else 0.0
    params_dict["theta_DA"] = 2e6 if m2 else 1e18
    params_dict["delta_sigma_reduction"] = 0.3 if m3 else 0.0
    params_dict.update(base_overrides)
    params = ModelParameters(**params_dict)

    seed = seed_base + abs(hash((scenario.code, config_label, mc_idx))) % 10_000
    model = ElectricityMarketABM(
        scenario=scenario, params=params, horizon_months=horizon, seed=seed,
    )
    df = model.run()
    return {
        "scenario": scenario.code,
        "config": config_label,
        "M1": m1, "M2": m2, "M3": m3,
        "mc_index": mc_idx,
        "clearing_rule": params.clearing_rule,
        "final_hhi": float(df["hhi"].iloc[-1]),
        "final_ai_lerner": float(df["ai_lerner"].iloc[-1]),
        "final_ai_cap_share": float(df["ai_capacity_share"].iloc[-1]),
        "final_ai_market_share": float(df["ai_market_share"].iloc[-1]),
        "final_ipp_exit": float(df["ipp_exit_rate"].iloc[-1]),
        "n_acquisitions": int(df["cumulative_acquisitions"].iloc[-1]),
        "clearing_price": float(df["clearing_price"].mean()),         # v0.3+ canonical name
        "mean_clearing_price": float(df["clearing_price"].mean()),    # v0.2 legacy alias
        "regulator_intervened": bool(df["regulator_intervened"].iloc[-1]),
        "intervention_time": df["intervention_time"].iloc[-1],
    }


def _run_clearing_task(args: tuple) -> dict:
    """Worker for clearing rule comparison."""
    scenario, clearing_rule, mc_idx, horizon, seed_base, m3_delta = args
    params = ModelParameters(
        clearing_rule=clearing_rule,
        delta_sigma_reduction=m3_delta,
    )
    seed = seed_base + abs(hash((scenario.code, clearing_rule, mc_idx))) % 10_000
    model = ElectricityMarketABM(
        scenario=scenario, params=params, horizon_months=horizon, seed=seed,
    )
    df = model.run()
    return {
        "scenario": scenario.code,
        "clearing_rule": clearing_rule,
        "m3_delta": m3_delta,
        "mc_index": mc_idx,
        "final_hhi": float(df["hhi"].iloc[-1]),
        "final_ai_lerner": float(df["ai_lerner"].iloc[-1]),
        "final_ai_cap_share": float(df["ai_capacity_share"].iloc[-1]),
        "final_ipp_exit": float(df["ipp_exit_rate"].iloc[-1]),
        "n_acquisitions": int(df["cumulative_acquisitions"].iloc[-1]),
        "clearing_price": float(df["clearing_price"].mean()),         # v0.3+ canonical
        "mean_clearing_price": float(df["clearing_price"].mean()),    # v0.2 legacy
    }


def _run_p3_task(args: tuple) -> dict:
    """Worker for P3 stress test."""
    scenario, preset_label, mc_idx, horizon, seed_base = args
    if preset_label == "default":
        params = default_params()
    elif preset_label == "p3_stress":
        params = p3_stress_params()
    else:
        raise ValueError(f"Unknown preset: {preset_label}")
    seed = seed_base + abs(hash((scenario.code, preset_label, mc_idx))) % 10_000
    model = ElectricityMarketABM(
        scenario=scenario, params=params, horizon_months=horizon, seed=seed,
    )
    df = model.run()
    return {
        "scenario": scenario.code,
        "preset": preset_label,
        "mc_index": mc_idx,
        "final_hhi": float(df["hhi"].iloc[-1]),
        "max_hhi": float(df["hhi"].max()),
        "month_hhi_peak": int(df["hhi"].idxmax()),
        "final_ai_lerner": float(df["ai_lerner"].iloc[-1]),
        "final_ai_cap_share": float(df["ai_capacity_share"].iloc[-1]),
        "final_ipp_exit": float(df["ipp_exit_rate"].iloc[-1]),
        "n_acquisitions": int(df["cumulative_acquisitions"].iloc[-1]),
        "regulator_intervened": bool(df["regulator_intervened"].iloc[-1]),
        "intervention_time": df["intervention_time"].iloc[-1],
    }


# ============================================================================
#  PARALLEL EXECUTION HELPER
# ============================================================================


def _parallel_execute(worker_func, tasks: list, workers: int, desc: str = "Runs") -> list:
    """Execute tasks in parallel with optional progress bar."""
    if workers == -1:
        workers = os.cpu_count() or 1
    if workers > 1:
        with mp.Pool(workers) as pool:
            iterator = pool.imap_unordered(worker_func, tasks)
            if HAS_TQDM:
                iterator = tqdm(iterator, total=len(tasks), desc=desc, unit="run")
                return list(iterator)
            results = []
            for i, r in enumerate(iterator, 1):
                results.append(r)
                if i % max(1, len(tasks) // 10) == 0 or i == len(tasks):
                    print(f"  {desc}: {i}/{len(tasks)} ({100*i/len(tasks):.0f}%)")
            return results
    else:
        if HAS_TQDM:
            return [worker_func(t) for t in tqdm(tasks, desc=desc, unit="run")]
        return [worker_func(t) for t in tasks]


# ============================================================================
#  ANALYSIS 1: MECHANISM DECOMPOSITION
# ============================================================================


MECHANISM_CONFIGS = [
    # (label,        M1,    M2,    M3)
    ("None",        False, False, False),
    ("M1 only",     True,  False, False),
    ("M2 only",     False, True,  False),
    ("M3 only",     False, False, True),
    ("M1+M2",       True,  True,  False),
    ("M1+M3",       True,  False, True),
    ("M2+M3",       False, True,  True),
    ("All (full)",  True,  True,  True),
]


def run_mechanism_decomposition(
    scenario: ScenarioConfig,
    n_mc: int = 30,
    horizon: int = 120,
    seed_base: int = 42,
    workers: int = 1,
    base_overrides: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Run 2³ = 8 mechanism configurations × n_mc Monte Carlo replications.

    Parameters
    ----------
    scenario : ScenarioConfig
        Which of the 6 scenarios to use as the experimental base.
    n_mc : int
        Monte Carlo replications per configuration.
    horizon : int
        Months per simulation.
    workers : int
        Parallel processes (-1 = all cores).
    base_overrides : dict | None
        Additional ModelParameters overrides applied to every config
        (e.g., {'clearing_rule': 'pay_as_bid'} to run decomp under pay-as-bid).

    Returns
    -------
    pd.DataFrame with one row per (config, MC) combination. 8 × n_mc rows.
    """
    base_overrides = base_overrides or {}
    tasks = [
        (scenario, label, m1, m2, m3, mc, horizon, seed_base, base_overrides)
        for label, m1, m2, m3 in MECHANISM_CONFIGS
        for mc in range(n_mc)
    ]
    results = _parallel_execute(_run_decomp_task, tasks, workers, desc="Decomp")
    return pd.DataFrame(results)


def decomp_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean values by configuration, ordered for readability."""
    order = [c[0] for c in MECHANISM_CONFIGS]
    summary = df.groupby("config").agg(
        HHI_mean=("final_hhi", "mean"),
        HHI_sd=("final_hhi", "std"),
        AI_Lerner=("final_ai_lerner", "mean"),
        AI_cap_share=("final_ai_cap_share", "mean"),
        AI_market_share=("final_ai_market_share", "mean"),
        IPP_exit=("final_ipp_exit", "mean"),
        Acquisitions=("n_acquisitions", "mean"),
        Reg_intervened=("regulator_intervened", "mean"),
    ).reindex(order).round(3)
    return summary


def decomp_main_effects(df: pd.DataFrame, metric: str = "final_hhi") -> dict:
    """ANOVA-style main effects and 2-way interactions on chosen metric."""
    effects = {}
    for mech in ["M1", "M2", "M3"]:
        on = df[df[mech]][metric].mean()
        off = df[~df[mech]][metric].mean()
        effects[f"{mech}_main_effect"] = round(float(on - off), 3)
    for a, b in [("M1", "M2"), ("M1", "M3"), ("M2", "M3")]:
        both = df[df[a] & df[b]][metric].mean()
        only_a = df[df[a] & ~df[b]][metric].mean()
        only_b = df[~df[a] & df[b]][metric].mean()
        neither = df[~df[a] & ~df[b]][metric].mean()
        effects[f"{a}x{b}_interaction"] = round(float((both - only_a - only_b + neither) / 2), 3)
    # 3-way interaction (approximate)
    all_on = df[df["M1"] & df["M2"] & df["M3"]][metric].mean()
    none = df[~df["M1"] & ~df["M2"] & ~df["M3"]][metric].mean()
    effects["full_minus_none"] = round(float(all_on - none), 3)
    return effects


def plot_decomposition(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    title_suffix: str = "",
) -> None:
    """Two-panel bar chart: HHI and AI Lerner by configuration."""
    summary = decomp_summary(df)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    config_labels = summary.index.tolist()
    n_mechs = [
        int(c[1]) + int(c[2]) + int(c[3])
        for c in MECHANISM_CONFIGS if c[0] in config_labels
    ]
    color_map = ["#bbbbbb", "#6699cc", "#cc8855", "#cc3333"]
    colors = [color_map[n] for n in n_mechs]

    # Panel 1: HHI
    hhi_vals = summary["HHI_mean"].values
    hhi_sd = summary["HHI_sd"].values
    axes[0].bar(range(len(config_labels)), hhi_vals, yerr=hhi_sd, color=colors,
                capsize=4, error_kw={"alpha": 0.5})
    axes[0].set_xticks(range(len(config_labels)))
    axes[0].set_xticklabels(config_labels, rotation=20, ha="right")
    axes[0].set_ylabel(f"Final HHI (mean ± SD, n={len(df)//8} MC)")
    axes[0].set_title(f"HHI by Mechanism Configuration{title_suffix}")
    axes[0].axhline(1800, ls="--", color="red", alpha=0.5, label="DOJ 1800 threshold")
    axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
    for i, v in enumerate(hhi_vals):
        axes[0].text(i, v + 25, f"{v:.0f}", ha="center", fontsize=9)

    # Panel 2: AI Lerner
    lerner_vals = summary["AI_Lerner"].values
    axes[1].bar(range(len(config_labels)), lerner_vals, color=colors)
    axes[1].set_xticks(range(len(config_labels)))
    axes[1].set_xticklabels(config_labels, rotation=20, ha="right")
    axes[1].set_ylabel(f"AI Lerner Index (mean, n={len(df)//8} MC)")
    axes[1].set_title(f"AI Lerner by Mechanism Configuration{title_suffix}")
    axes[1].grid(alpha=0.3, axis="y")
    for i, v in enumerate(lerner_vals):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    plt.show()


# ============================================================================
#  ANALYSIS 2: CLEARING RULE COMPARISON (Uniform vs Pay-as-bid)
# ============================================================================


def compare_clearing_rules(
    scenario: ScenarioConfig,
    n_mc: int = 30,
    horizon: int = 120,
    seed_base: int = 42,
    workers: int = 1,
    m3_deltas: tuple = (0.0, 0.3),
) -> pd.DataFrame:
    """
    Compare uniform vs pay-as-bid clearing under multiple M3 levels.

    The point: M3 should be DEAD under uniform (as shown empirically) but
    ACTIVE under pay-as-bid, where bid magnitude directly affects revenue.

    Returns DataFrame with one row per (clearing_rule, m3_delta, MC).
    """
    tasks = [
        (scenario, rule, mc, horizon, seed_base, delta)
        for rule in ("uniform", "pay_as_bid")
        for delta in m3_deltas
        for mc in range(n_mc)
    ]
    results = _parallel_execute(_run_clearing_task, tasks, workers, desc="Clearing")
    return pd.DataFrame(results)


def clearing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary table grouped by (clearing_rule, m3_delta)."""
    return df.groupby(["clearing_rule", "m3_delta"]).agg(
        HHI=("final_hhi", "mean"),
        HHI_sd=("final_hhi", "std"),
        AI_Lerner=("final_ai_lerner", "mean"),
        AI_cap_share=("final_ai_cap_share", "mean"),
        IPP_exit=("final_ipp_exit", "mean"),
        Acquisitions=("n_acquisitions", "mean"),
        Mean_price=("mean_clearing_price", "mean"),
    ).round(3)


def plot_clearing_comparison(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Grouped bar chart: HHI · AI Lerner · Acquisitions by (rule, delta)."""
    summary = clearing_summary(df).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("HHI", "Final HHI"),
        ("AI_Lerner", "AI Lerner Index"),
        ("Acquisitions", "Mean Cumulative Acquisitions"),
    ]
    rule_colors = {"uniform": "#6699cc", "pay_as_bid": "#cc6655"}
    deltas = sorted(summary["m3_delta"].unique())

    for ax, (key, title) in zip(axes, metrics):
        x = np.arange(len(deltas))
        width = 0.35
        for i, rule in enumerate(["uniform", "pay_as_bid"]):
            sub = summary[summary["clearing_rule"] == rule].sort_values("m3_delta")
            ax.bar(x + (i - 0.5) * width, sub[key].values, width,
                   label=rule, color=rule_colors[rule])
        ax.set_xticks(x); ax.set_xticklabels([f"δ={d}" for d in deltas])
        ax.set_title(title); ax.set_xlabel("M3 parameter")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Clearing Rule × M3 Comparison", y=1.02, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    plt.show()


# ============================================================================
#  ANALYSIS 3: P3 STRESS TEST
# ============================================================================


def run_p3_stress(
    scenarios: list[ScenarioConfig] = None,
    n_mc: int = 30,
    horizon: int = 120,
    seed_base: int = 42,
    workers: int = 1,
) -> pd.DataFrame:
    """
    P3 stress test: compare default vs p3_stress preset to test regulatory irreversibility.

    The p3_stress preset uses aggressive M2 parameters to push HHI past the 1800 trigger,
    enabling actual P3 testing. Without this, default M2 plateaus HHI ~900-1000 and
    regulator never fires.

    Returns DataFrame with one row per (scenario, preset, MC).
    """
    scenarios = scenarios or SCENARIOS_6CELL
    tasks = [
        (scen, preset, mc, horizon, seed_base)
        for scen in scenarios
        for preset in ("default", "p3_stress")
        for mc in range(n_mc)
    ]
    results = _parallel_execute(_run_p3_task, tasks, workers, desc="P3")
    return pd.DataFrame(results)


def p3_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary by (scenario, preset)."""
    return df.groupby(["scenario", "preset"]).agg(
        HHI_mean=("final_hhi", "mean"),
        HHI_max=("max_hhi", "mean"),
        HHI_sd=("final_hhi", "std"),
        AI_Lerner=("final_ai_lerner", "mean"),
        IPP_exit=("final_ipp_exit", "mean"),
        Acquisitions=("n_acquisitions", "mean"),
        Reg_intervened_pct=("regulator_intervened", "mean"),
    ).round(3)


def p3_test(df: pd.DataFrame, preset: str = "p3_stress") -> dict:
    """
    Compute P3 proxy: HHL vs HHH HHI under given preset.
    Returns dict with HHL/HHH means and difference. Positive HHL−HHH supports P3 hypothesis.
    """
    sub = df[df["preset"] == preset]
    hhl = sub[sub["scenario"] == "HHL"]["final_hhi"]
    hhh = sub[sub["scenario"] == "HHH"]["final_hhi"]
    return {
        "preset": preset,
        "HHL_mean": round(float(hhl.mean()), 2),
        "HHH_mean": round(float(hhh.mean()), 2),
        "P3_proxy_HHL_minus_HHH": round(float(hhl.mean() - hhh.mean()), 2),
        "HHL_reg_intervened": round(float(
            sub[sub["scenario"] == "HHL"]["regulator_intervened"].mean()
        ), 3),
        "HHH_reg_intervened": round(float(
            sub[sub["scenario"] == "HHH"]["regulator_intervened"].mean()
        ), 3),
    }


def plot_p3_stress(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Compare HHI across scenarios under default vs p3_stress."""
    summary = p3_summary(df).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    scenarios_order = [s.code for s in SCENARIOS_6CELL]
    width = 0.4
    x = np.arange(len(scenarios_order))

    # Panel 1: HHI
    for i, preset in enumerate(["default", "p3_stress"]):
        vals = []
        for s in scenarios_order:
            row = summary[(summary["scenario"] == s) & (summary["preset"] == preset)]
            vals.append(row["HHI_mean"].values[0] if len(row) else 0)
        color = "#6699cc" if preset == "default" else "#cc3333"
        axes[0].bar(x + (i - 0.5) * width, vals, width, label=preset, color=color)
    axes[0].set_xticks(x); axes[0].set_xticklabels(scenarios_order)
    axes[0].set_ylabel("Final HHI (mean)")
    axes[0].set_title("HHI: default vs P3 stress preset")
    axes[0].axhline(1800, ls="--", color="red", alpha=0.5, label="DOJ 1800 trigger")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3, axis="y")

    # Panel 2: Regulator intervention rate
    for i, preset in enumerate(["default", "p3_stress"]):
        vals = []
        for s in scenarios_order:
            row = summary[(summary["scenario"] == s) & (summary["preset"] == preset)]
            vals.append(row["Reg_intervened_pct"].values[0] if len(row) else 0)
        color = "#6699cc" if preset == "default" else "#cc3333"
        axes[1].bar(x + (i - 0.5) * width, vals, width, label=preset, color=color)
    axes[1].set_xticks(x); axes[1].set_xticklabels(scenarios_order)
    axes[1].set_ylabel("Regulator Intervention Rate (fraction of MC)")
    axes[1].set_title("Regulator activation: default vs P3 stress")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    plt.show()


# ============================================================================
#  ANALYSIS 4: ACQUISITION PARAMETER SENSITIVITY
# ============================================================================


def _run_acq_sensitivity_task(args: tuple) -> dict:
    """Worker for acquisition parameter sensitivity sweep."""
    scenario, cost_mult, phi, utilization, m2_on, mc_idx, horizon, seed_base, base_overrides = args
    params_dict = dict(default_params().__dict__)
    # M1 fixed on, M3 fixed off (default)
    params_dict["beta"] = 0.4
    params_dict["delta_sigma_reduction"] = 0.0
    # M2 toggle + parameter overrides
    params_dict["theta_DA"] = 2e6 if m2_on else 1e18
    params_dict["acquisition_cost_multiplier"] = cost_mult
    params_dict["phi"] = phi
    params_dict["m2_utilization_rate"] = utilization
    if base_overrides:
        params_dict.update(base_overrides)
    params = ModelParameters(**params_dict)

    seed = seed_base + abs(hash((scenario.code, cost_mult, phi, utilization, m2_on, mc_idx))) % 10_000
    model = ElectricityMarketABM(
        scenario=scenario, params=params, horizon_months=horizon, seed=seed,
    )
    df = model.run()
    return {
        "scenario": scenario.code,
        "cost_mult": cost_mult,
        "phi": phi,
        "utilization": utilization,
        "M2_on": m2_on,
        "mc_index": mc_idx,
        "final_hhi": float(df["hhi"].iloc[-1]),
        "final_ai_lerner": float(df["ai_lerner"].iloc[-1]),
        "final_ai_cap_share": float(df["ai_capacity_share"].iloc[-1]),
        "final_ipp_exit": float(df["ipp_exit_rate"].iloc[-1]),
        "n_acquisitions": int(df["cumulative_acquisitions"].iloc[-1]),
    }


def run_acquisition_sensitivity(
    scenario: ScenarioConfig,
    cost_mult_grid: tuple = (0.7, 1.0, 1.3),
    phi_grid: tuple = (1.2, 1.5, 1.8),
    utilization_grid: tuple = (0.60, 0.70, 0.85),
    n_mc: int = 50,
    horizon: int = 120,
    seed_base: int = 42,
    workers: int = 1,
    base_overrides: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Sweep over acquisition parameters to test robustness of the M2 effect.

    For each combination of (cost_mult, phi, utilization), runs M2-on vs M2-off
    (M1 fixed on, M3 fixed off) with n_mc replications. The M2 effect is the
    mean HHI(M2-on) − mean HHI(M2-off) within each cell.

    Default grid: 3 × 3 × 3 = 27 cells × 2 M2 states × 50 MC = 2,700 runs.

    Parameters
    ----------
    scenario          : which scenario to use as the experimental base
    cost_mult_grid    : κ_acq sweep range (acquisition_cost_multiplier)
                        Default v1.0 = 1.0; lower = cheaper acquisitions
    phi_grid          : NPV trigger sweep range
                        Default v1.0 = 1.5; lower = more aggressive acquisition
    utilization_grid  : post-acquisition utilization rate (m2_utilization_rate)
                        Default v1.0 = 0.70; higher = more value extracted
    n_mc              : Monte Carlo replications per cell
    horizon           : simulation length in months
    workers           : parallel workers (-1 = all cores)
    base_overrides    : additional ModelParameters overrides

    Returns
    -------
    DataFrame with one row per (cost_mult, phi, utilization, M2_on, mc_index).
    Use acquisition_sensitivity_summary() to aggregate into M2-effect grid.
    """
    base_overrides = base_overrides or {}
    tasks = [
        (scenario, cm, ph, ut, m2_on, mc, horizon, seed_base, base_overrides)
        for cm in cost_mult_grid
        for ph in phi_grid
        for ut in utilization_grid
        for m2_on in (False, True)
        for mc in range(n_mc)
    ]
    results = _parallel_execute(_run_acq_sensitivity_task, tasks, workers, desc="AcqSens")
    return pd.DataFrame(results)


def acquisition_sensitivity_summary(
    df: pd.DataFrame,
    metric: str = "final_hhi",
) -> pd.DataFrame:
    """
    Aggregate the sensitivity sweep into a long-form effect table.

    For each (cost_mult, phi, utilization) combination, computes:
        M2 effect = mean(metric | M2 on) − mean(metric | M2 off)
        SE        = combined standard error
        n_acq     = mean number of acquisitions when M2 on

    Returns one row per parameter combination, sorted by M2 effect descending.
    """
    rows = []
    grouped = df.groupby(["cost_mult", "phi", "utilization"])
    for (cm, ph, ut), sub in grouped:
        on = sub[sub["M2_on"]][metric]
        off = sub[~sub["M2_on"]][metric]
        effect = float(on.mean() - off.mean())
        # Standard error of difference of independent means
        se = float(np.sqrt(on.var(ddof=1) / len(on) + off.var(ddof=1) / len(off)))
        n_acq_on = float(sub[sub["M2_on"]]["n_acquisitions"].mean())
        rows.append({
            "cost_mult": cm,
            "phi": ph,
            "utilization": ut,
            "M2_effect": round(effect, 1),
            "SE": round(se, 1),
            "ci_low": round(effect - 1.96 * se, 1),
            "ci_high": round(effect + 1.96 * se, 1),
            "n_acquisitions_M2on": round(n_acq_on, 1),
        })
    out = pd.DataFrame(rows).sort_values("M2_effect", ascending=False).reset_index(drop=True)
    return out


def acquisition_sensitivity_grid(
    df: pd.DataFrame,
    fix_utilization: Optional[float] = None,
    metric: str = "final_hhi",
) -> pd.DataFrame:
    """
    Reshape into a 2D grid (cost_mult × phi) with utilization fixed.

    Each cell shows the M2 effect at that parameter combination.
    Use this for a publication table or to feed a heatmap.

    Parameters
    ----------
    fix_utilization : if None, use the middle utilization in the data
    """
    summary = acquisition_sensitivity_summary(df, metric=metric)
    if fix_utilization is None:
        utilizations = sorted(summary["utilization"].unique())
        fix_utilization = utilizations[len(utilizations) // 2]
    sub = summary[summary["utilization"] == fix_utilization]
    pivot = sub.pivot(index="cost_mult", columns="phi", values="M2_effect")
    pivot.index.name = f"cost_mult (κ_acq) [utilization={fix_utilization}]"
    pivot.columns.name = "phi (NPV trigger)"
    return pivot


def plot_acquisition_sensitivity(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    metric: str = "final_hhi",
) -> None:
    """
    3-panel heatmap of M2 effect across (cost_mult × phi), one panel per utilization.

    Strong robustness signal: cells should be similarly colored (M2 effect ≈ constant).
    Sensitivity signal: cells should vary systematically with parameters.
    """
    summary = acquisition_sensitivity_summary(df, metric=metric)
    utilizations = sorted(summary["utilization"].unique())
    n_panels = len(utilizations)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    # Common color scale across panels
    vmin = summary["M2_effect"].min()
    vmax = summary["M2_effect"].max()

    for ax, ut in zip(axes, utilizations):
        sub = summary[summary["utilization"] == ut]
        pivot = sub.pivot(index="cost_mult", columns="phi", values="M2_effect")

        im = ax.imshow(
            pivot.values, aspect="auto", cmap="RdYlBu_r",
            vmin=vmin, vmax=vmax,
            origin="lower",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{r:.1f}" for r in pivot.index])
        ax.set_xlabel("phi (NPV trigger)")
        ax.set_ylabel("cost_mult (κ_acq)")
        ax.set_title(f"utilization = {ut}")

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(
                    j, i, f"{val:+.0f}",
                    ha="center", va="center",
                    color="white" if abs(val - (vmin + vmax) / 2) > (vmax - vmin) * 0.25 else "black",
                    fontsize=10, fontweight="bold",
                )

    fig.colorbar(im, ax=axes, label=f"M2 effect on {metric}", shrink=0.8)
    fig.suptitle(
        f"M2 effect sensitivity to acquisition parameters "
        f"(scenario {df['scenario'].iloc[0]}, n_mc={df.groupby(['cost_mult','phi','utilization','M2_on']).size().iloc[0]})",
        fontsize=11, y=1.02,
    )
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    plt.show()
    plt.close()

# ============================================================================
### Continuous compute-concentration sweep
# ============================================================================

def run_compute_concentration_sweep(
    scenario_template,
    target_hhi_grid=(2000, 2500, 3000, 3500, 4000, 4500, 5000),
    n_mc=50,
    horizon=120,
    workers=1,
    seed_base=42,
    beta_on=0.4,
    beta_off=0.0,
    # Legacy alias for backward compat
    compute_hhi_grid=None,
):
    """Continuous sweep of WITHIN-AI compute HHI, AI total share fixed at 0.70.

    Tests whether the M1 effect on AI Lerner exhibits discrete tipping as
    within-AI concentration rises from uniform (HHI=2000) toward one-dominant
    (HHI=5000). Used in Section 4.6 within-AI dimension (complement to
    between-group share sweep).

    v6+ implementation uses compute_shares_override to bypass Dirichlet draw,
    yielding deterministic share vectors at each target HHI level.

    Parameters
    ----------
    scenario_template : ScenarioConfig
    target_hhi_grid : tuple of int
        Within-AI HHI levels to sweep (10000/n_ai = 2000 is uniform floor).
    compute_hhi_grid : tuple, optional
        Legacy alias for target_hhi_grid; supported for backward compat.
    """
    if compute_hhi_grid is not None:
        target_hhi_grid = compute_hhi_grid

    tasks = []
    for target_hhi in target_hhi_grid:
        shares = _generate_compute_shares_for_hhi(int(target_hhi), n_firms=5,
                                                  ai_total_share=0.70)
        for beta in (beta_on, beta_off):
            for mc_idx in range(n_mc):
                tasks.append((scenario_template, shares, int(target_hhi),
                              beta, mc_idx, horizon, seed_base))
    results = _parallel_execute(_run_compute_sweep_task, tasks, workers,
                                 desc="ComputeSweep")
    return pd.DataFrame(results)


def _generate_compute_shares_for_hhi(target_hhi, n_firms=5, ai_total_share=0.70):
    """Generate an n_firms-vector that has the requested *within-AI* HHI.

    Construction: 1 dominant firm + (n_firms-1) uniform smaller firms,
    scaled so they sum to ai_total_share.

    HHI = 10000 × Σ (share_i / ai_total_share)² (HHI defined on within-group shares)
    """
    from scipy.optimize import brentq
    target_ss = target_hhi / 10000.0   # sum of squared *relative* (within-AI) shares

    def f(s1_rel):
        """s1_rel = top firm's within-AI share fraction."""
        s_rest = (1.0 - s1_rel) / (n_firms - 1)
        return s1_rel**2 + (n_firms - 1) * s_rest**2 - target_ss

    # Floor: uniform within-AI (s1_rel = 1/n_firms → HHI = 10000/n_firms)
    uniform_floor = 1.0 / n_firms
    if target_ss <= uniform_floor:
        # Target is at or below the uniform floor: return uniform shares
        return np.ones(n_firms) * (ai_total_share / n_firms)

    try:
        s1_rel = brentq(f, uniform_floor, 0.99)
    except ValueError:
        # Fallback: return uniform if target is unreachable
        return np.ones(n_firms) * (ai_total_share / n_firms)

    s_rest_rel = (1.0 - s1_rel) / (n_firms - 1)
    rel_shares = np.array([s1_rel] + [s_rest_rel] * (n_firms - 1))
    # Scale by ai_total_share to get absolute compute shares
    return rel_shares * ai_total_share


def _compute_hhi_from_shares(shares):
    """Compute HHI from absolute share vector (assumes shares are within-AI fractions)."""
    total = sum(shares)
    if total <= 0:
        return 0.0
    rel = np.asarray(shares) / total
    return float(10000.0 * np.sum(rel ** 2))


def _run_compute_sweep_task(args):
    """Worker for within-AI compute HHI sweep (v6+: uses compute_shares_override)."""
    scenario_template, shares, target_hhi, beta, mc_idx, horizon, seed_base = args

    # Use compute_shares_override to bypass Dirichlet draw
    params_dict = dict(default_params().__dict__)
    params_dict["beta"] = float(beta)
    params = ModelParameters(**params_dict)

    seed = seed_base + mc_idx * 1000 + int(target_hhi)
    model = ElectricityMarketABM(
        scenario=scenario_template,
        params=params,
        horizon_months=horizon,
        seed=seed,
        compute_shares_override=np.asarray(shares),
    )
    df = model.run()
    final = df.iloc[-1]
    return {
        "compute_hhi_target": int(target_hhi),
        "compute_hhi_realized": _compute_hhi_from_shares(shares),
        "beta": float(beta),
        "M1": beta > 0,
        "mc_index": mc_idx,
        "final_ai_lerner": float(final["ai_lerner"]),
        "final_hhi": float(final["hhi"]),
        "final_ai_cap_share": float(final["ai_capacity_share"]),
        "n_acquisitions": int(final["cumulative_acquisitions"]),
        "clearing_price": float(df["clearing_price"].mean()),
    }


def compute_sweep_summary(df_sweep, B=1000, seed=42):
    """Per-HHI-level M1 effect on AI Lerner with bootstrap 95% CIs."""
    rng = np.random.default_rng(seed)
    rows = []
    for hhi in sorted(df_sweep["compute_hhi_target"].unique()):
        sub = df_sweep[df_sweep["compute_hhi_target"] == hhi]
        on = sub[sub["M1"]]["final_ai_lerner"].values
        off = sub[~sub["M1"]]["final_ai_lerner"].values
        if len(on) == 0 or len(off) == 0:
            continue
        effect = on.mean() - off.mean()
        n_on, n_off = len(on), len(off)
        diffs = np.empty(B)
        for b in range(B):
            d_on = on[rng.integers(0, n_on, n_on)].mean()
            d_off = off[rng.integers(0, n_off, n_off)].mean()
            diffs[b] = d_on - d_off
        rows.append({
            "compute_hhi": int(hhi),
            "M1_effect_on_lerner": round(float(effect), 4),
            "ci_low": round(float(np.percentile(diffs, 2.5)), 4),
            "ci_high": round(float(np.percentile(diffs, 97.5)), 4),
            "ai_lerner_M1on": round(float(on.mean()), 4),
            "ai_lerner_M1off": round(float(off.mean()), 4),
        })
    return pd.DataFrame(rows)

# ============================================================================
#  CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run ABM analyses: decomposition, clearing comparison, P3 stress test, acq sensitivity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--analysis",
                        choices=["decomp", "clearing", "p3", "acq_sens", "all"],
                        default="decomp",
                        help="Which analysis to run")
    parser.add_argument("--scenario", default="HHL",
                        help="Scenario code for decomp/clearing/acq_sens (LLL, LLH, ..., HHH)")
    parser.add_argument("--n-mc", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (-1 = all cores)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output or (Path.cwd() / "analysis_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    scen = next(s for s in SCENARIOS_6CELL if s.code == args.scenario)

    def run_decomp():
        print("=" * 60)
        print(f"MECHANISM DECOMPOSITION (Scenario {scen.code}, n_mc={args.n_mc})")
        print("=" * 60)
        t0 = time.time()
        df = run_mechanism_decomposition(
            scen, n_mc=args.n_mc, horizon=args.horizon, workers=args.workers,
        )
        print(f"  Time: {time.time()-t0:.1f}s, runs: {len(df)}")
        df.to_csv(output_dir / "decomp_raw.csv", index=False)

        summary = decomp_summary(df)
        summary.to_csv(output_dir / "decomp_summary.csv")
        print("\nSummary:")
        print(summary.to_string())

        effects = decomp_main_effects(df)
        with open(output_dir / "decomp_effects.json", "w") as f:
            json.dump(effects, f, indent=2)
        print("\nMain effects on HHI:")
        for k, v in effects.items():
            print(f"  {k}: {v:+.2f}")

        plot_decomposition(df, output_dir / "decomp_plot.png",
                           title_suffix=f" (Scenario {scen.code})")
        return df

    def run_clearing():
        print("\n" + "=" * 60)
        print(f"CLEARING RULE COMPARISON (Scenario {scen.code}, n_mc={args.n_mc})")
        print("=" * 60)
        t0 = time.time()
        df = compare_clearing_rules(
            scen, n_mc=args.n_mc, horizon=args.horizon, workers=args.workers,
        )
        print(f"  Time: {time.time()-t0:.1f}s, runs: {len(df)}")
        df.to_csv(output_dir / "clearing_raw.csv", index=False)

        summary = clearing_summary(df)
        summary.to_csv(output_dir / "clearing_summary.csv")
        print("\nSummary:")
        print(summary.to_string())

        plot_clearing_comparison(df, output_dir / "clearing_plot.png")
        return df

    def run_p3():
        print("\n" + "=" * 60)
        print(f"P3 STRESS TEST (all 6 scenarios, n_mc={args.n_mc})")
        print("=" * 60)
        t0 = time.time()
        df = run_p3_stress(n_mc=args.n_mc, horizon=args.horizon, workers=args.workers)
        print(f"  Time: {time.time()-t0:.1f}s, runs: {len(df)}")
        df.to_csv(output_dir / "p3_raw.csv", index=False)

        summary = p3_summary(df)
        summary.to_csv(output_dir / "p3_summary.csv")
        print("\nSummary:")
        print(summary.to_string())

        for preset in ["default", "p3_stress"]:
            test = p3_test(df, preset=preset)
            print(f"\nP3 test under '{preset}':")
            for k, v in test.items():
                print(f"  {k}: {v}")

        plot_p3_stress(df, output_dir / "p3_plot.png")
        return df

    def run_acq_sens():
        print("\n" + "=" * 60)
        print(f"ACQUISITION SENSITIVITY (Scenario {scen.code}, n_mc={args.n_mc})")
        print("  Grid: cost_mult × phi × utilization = 3 × 3 × 3 = 27 cells × 2 M2 states")
        print("=" * 60)
        t0 = time.time()
        df = run_acquisition_sensitivity(
            scen, n_mc=args.n_mc, horizon=args.horizon, workers=args.workers,
        )
        print(f"  Time: {time.time()-t0:.1f}s, runs: {len(df)}")
        df.to_csv(output_dir / "acq_sens_raw.csv", index=False)

        summary = acquisition_sensitivity_summary(df)
        summary.to_csv(output_dir / "acq_sens_summary.csv", index=False)
        print("\nFull sensitivity table (sorted by M2 effect):")
        print(summary.to_string(index=False))

        # 2D grid at default utilization
        grid = acquisition_sensitivity_grid(df, fix_utilization=0.70)
        grid.to_csv(output_dir / "acq_sens_grid.csv")
        print("\n2D grid at utilization=0.70 (default):")
        print(grid.to_string())

        # Robustness diagnostic
        m2_effects = summary["M2_effect"].values
        print(f"\nM2 effect range across 27 parameter combinations:")
        print(f"  min:    {m2_effects.min():+.0f}")
        print(f"  median: {np.median(m2_effects):+.0f}")
        print(f"  max:    {m2_effects.max():+.0f}")
        print(f"  CV:     {m2_effects.std()/m2_effects.mean():.2f}")
        print(f"  All ★? {'yes' if all(s['ci_low'] > 0 for _, s in summary.iterrows()) else 'no'}")

        plot_acquisition_sensitivity(df, output_dir / "acq_sens_plot.png")
        return df

    if args.analysis == "decomp":
        run_decomp()
    elif args.analysis == "clearing":
        run_clearing()
    elif args.analysis == "p3":
        run_p3()
    elif args.analysis == "acq_sens":
        run_acq_sens()
    elif args.analysis == "all":
        run_decomp()
        run_clearing()
        run_p3()
        run_acq_sens()
        print("\nAll analyses complete. Outputs in:", output_dir)


if __name__ == "__main__":
    main()


# ============================================================================
#  v0.3 ADDITIONS — extended sweeps + counterfactual policy simulations
# ============================================================================
#
# Added in v0.3 (May 2026):
#   - Cross-scenario decomposition (all 6 scenarios × 8 mechanism configs)
#   - Agent count sensitivity (fringe size sweep 10-80 IPPs)
#   - Compute share sweep (between-group, bounded regime test)
#   - Bootstrap effects + ANOVA (for Table 2/3 inference)
#   - Four counterfactual policy simulations (break-up, PPA disc., cost disc., indicator)
#
# All routines parallelizable via _parallel_execute. Requires baseline_model v6.


# ============================================================================
#  CROSS-SCENARIO DECOMPOSITION (all 6 scenarios)
# ============================================================================

def run_cross_scenario_decomposition(
    scenarios: list[ScenarioConfig],
    n_mc: int = 150,
    horizon: int = 120,
    workers: int = 1,
    save_individual_runs: bool = False,
) -> pd.DataFrame:
    """Run M1×M2×M3 = 8 mechanism configs × n_MC across all scenarios in `scenarios`.

    Returns single DataFrame with column `scenario` indicating which cell.
    Total runs: len(scenarios) × 8 × n_mc.

    Parameters
    ----------
    scenarios : list[ScenarioConfig]
        Typically `SCENARIOS_6CELL` from baseline_model.
    n_mc : int
        Monte Carlo replications per (scenario, mechanism config) cell.
    horizon : int
        Simulation horizon in months.
    workers : int
        Number of parallel workers (1 = serial).
    save_individual_runs : bool
        If True, also return raw per-run output as separate DataFrame.

    Returns
    -------
    pd.DataFrame with columns: scenario, M1, M2, M3, mc_index,
        final_hhi, final_ai_lerner, final_ai_cap_share, ai_market_share,
        n_acquisitions, ipp_exit_rate, clearing_price.
    """
    all_results = []
    for scen in scenarios:
        df_scen = run_mechanism_decomposition(
            scen, n_mc=n_mc, horizon=horizon, workers=workers,
        )
        df_scen["scenario"] = scen.code
        all_results.append(df_scen)
    return pd.concat(all_results, ignore_index=True)


# ============================================================================
#  BOOTSTRAP INFERENCE: main effects + interactions + ANOVA
# ============================================================================

def compute_effects_and_interactions(
    df: pd.DataFrame,
    outcome: str = "final_hhi",
    B: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute M1, M2, M3 main effects + 3 pairwise interactions on `outcome`.

    Returns DataFrame with columns: effect, estimate, ci_low, ci_high, p_value.
    Effects: M1, M2, M3, M1xM2, M1xM3, M2xM3.

    Bootstrap inference uses B resamples; p-value is two-sided fraction of resamples
    crossing zero.
    """
    rng = np.random.default_rng(seed)

    def main_effect(d, m):
        """Mean difference: (m=on) − (m=off)."""
        return d.loc[d[m], outcome].mean() - d.loc[~d[m], outcome].mean()

    def interaction(d, m1, m2):
        """Pairwise interaction effect: ((on,on) − (on,off)) − ((off,on) − (off,off))."""
        on_on   = d.loc[d[m1] & d[m2], outcome].mean()
        on_off  = d.loc[d[m1] & ~d[m2], outcome].mean()
        off_on  = d.loc[~d[m1] & d[m2], outcome].mean()
        off_off = d.loc[~d[m1] & ~d[m2], outcome].mean()
        return (on_on - on_off) - (off_on - off_off)

    effects_def = {
        "M1": lambda d: main_effect(d, "M1"),
        "M2": lambda d: main_effect(d, "M2"),
        "M3": lambda d: main_effect(d, "M3"),
        "M1xM2": lambda d: interaction(d, "M1", "M2"),
        "M1xM3": lambda d: interaction(d, "M1", "M3"),
        "M2xM3": lambda d: interaction(d, "M2", "M3"),
    }

    rows = []
    n = len(df)
    for name, fn in effects_def.items():
        est = fn(df)
        boot = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, n, n)
            try:
                boot[b] = fn(df.iloc[idx])
            except Exception:
                boot[b] = np.nan
        boot = boot[~np.isnan(boot)]
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        # Two-sided p-value: fraction of resamples crossing zero
        if est >= 0:
            p_val = 2 * np.mean(boot <= 0)
        else:
            p_val = 2 * np.mean(boot >= 0)
        p_val = min(p_val, 1.0)
        rows.append({
            "effect": name,
            "estimate": round(float(est), 4),
            "ci_low": round(float(ci_low), 4),
            "ci_high": round(float(ci_high), 4),
            "p_value": round(float(p_val), 4),
        })
    return pd.DataFrame(rows)


def anova_decomp(df: pd.DataFrame, outcome: str = "final_hhi") -> pd.DataFrame:
    """Three-way ANOVA decomposition for M1×M2×M3 design.

    Returns DataFrame with columns: source, sum_sq, df, F, p_value.
    Sources include M1, M2, M3, M1:M2, M1:M3, M2:M3, M1:M2:M3, Residual.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        # Fallback to manual F computation if statsmodels unavailable
        return pd.DataFrame([{
            "source": "(statsmodels unavailable)",
            "sum_sq": np.nan, "df": np.nan, "F": np.nan, "p_value": np.nan,
        }])

    # Convert booleans to int for OLS
    d = df.copy()
    d["M1"] = d["M1"].astype(int)
    d["M2"] = d["M2"].astype(int)
    d["M3"] = d["M3"].astype(int)

    formula = f"{outcome} ~ M1 * M2 * M3"
    model = ols(formula, data=d).fit()
    table = sm.stats.anova_lm(model, typ=2)
    table = table.reset_index().rename(columns={
        "index": "source", "sum_sq": "sum_sq", "df": "df", "F": "F", "PR(>F)": "p_value"
    })
    table["sum_sq"] = table["sum_sq"].round(2)
    table["F"] = table["F"].round(2)
    table["p_value"] = table["p_value"].round(4)
    return table


# ============================================================================
#  AGENT COUNT SENSITIVITY (fringe size sweep)
# ============================================================================

def _run_agent_count_task(args: tuple) -> dict:
    """Worker for fringe size sweep. Uses P3 stress preset (aggressive M2)."""
    scenario, n_ipps, preset, mc_idx, horizon, seed_base = args
    if preset == "p3_stress":
        try:
            from baseline_model import P3_STRESS_PARAMS
            params = ModelParameters(**P3_STRESS_PARAMS.__dict__)
        except ImportError:
            params = ModelParameters(
                phi=1.2, theta_DA=1.0e8, acquisition_cost_multiplier=0.7,
                m2_utilization_rate=0.85, m2_cooldown_months=6,
            )
    else:
        params = default_params()

    seed = seed_base + abs(hash((scenario.code, int(n_ipps), int(mc_idx), preset))) % 10_000
    model = ElectricityMarketABM(
        scenario=scenario, params=params,
        n_ipps=int(n_ipps), horizon_months=horizon, seed=seed,
    )
    df = model.run()
    return {
        "scenario": scenario.code,
        "n_ipps": int(n_ipps),
        "preset": preset,
        "mc_index": mc_idx,
        "final_hhi": float(df["hhi"].iloc[-1]),
        "final_hhi_sd": float(df["hhi"].std()),
        "max_hhi": float(df["hhi"].max()),
        "max_hhi_month": int(df["hhi"].idxmax()),
        "regulator_activated": int(df["hhi"].max() >= scenario.hhi_trigger),
        "n_acquisitions": int(df["cumulative_acquisitions"].iloc[-1]),
        "final_ipp_exit": float(df["ipp_exit_rate"].iloc[-1]),
        "ai_lerner_mean": float(df["ai_lerner"].mean()),
    }


def run_agent_count_sensitivity(
    scenario: ScenarioConfig,
    n_ipps_grid: tuple = (10, 15, 20, 30, 40, 60, 80),
    preset: str = "p3_stress",
    n_mc: int = 50,
    horizon: int = 120,
    seed_base: int = 42,
    workers: int = 1,
) -> pd.DataFrame:
    """Sweep number of IPPs (fringe size) to test if regulatory trigger activation
    depends on fringe count. Uses P3 stress preset (aggressive M2) by default.

    Total runs: len(n_ipps_grid) × n_mc.
    """
    tasks = []
    for n in n_ipps_grid:
        for mc in range(n_mc):
            tasks.append((scenario, n, preset, mc, horizon, seed_base))
    results = _parallel_execute(_run_agent_count_task, tasks, workers, desc="AgentCount")
    return pd.DataFrame(results)


def agent_count_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-fringe-size summary: max HHI, regulator activation rate, acquisitions."""
    return df.groupby(["n_ipps", "preset"]).agg(
        final_hhi_mean=("final_hhi", "mean"),
        final_hhi_sd=("final_hhi", "std"),
        max_hhi_mean=("max_hhi", "mean"),
        max_hhi_sd=("max_hhi", "std"),
        regulator_pct=("regulator_activated", lambda x: x.mean() * 100),
        ipp_exit_mean=("final_ipp_exit", "mean"),
        n_acquisitions_mean=("n_acquisitions", "mean"),
        ai_lerner_mean=("ai_lerner_mean", "mean"),
    ).round(3)


def plot_agent_count_sensitivity(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Two-panel: max HHI vs n_IPPs (with DOJ trigger lines) + regulator activation rate."""
    summary = agent_count_summary(df).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for preset in summary["preset"].unique():
        sub = summary[summary["preset"] == preset]
        axes[0].errorbar(
            sub["n_ipps"], sub["max_hhi_mean"], yerr=sub["max_hhi_sd"],
            marker="o", linewidth=2, capsize=4, label=preset,
        )
    axes[0].axhline(1800, ls="--", color="firebrick", alpha=0.7, label="DOJ 1800 trigger")
    axes[0].axhline(3000, ls="--", color="darkred", alpha=0.5, label="DOJ 3000 (ex-post)")
    axes[0].set_xlabel("Number of IPPs (fringe size)")
    axes[0].set_ylabel("Max HHI over 120 months (mean ± SD)")
    axes[0].set_title("Structural ceiling vs fringe size")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for preset in summary["preset"].unique():
        sub = summary[summary["preset"] == preset]
        axes[1].plot(sub["n_ipps"], sub["regulator_pct"] / 100,
                     marker="o", linewidth=2, label=preset)
    axes[1].set_xlabel("Number of IPPs (fringe size)")
    axes[1].set_ylabel("Regulator activation rate (fraction of MC)")
    axes[1].set_title("Regulator triggering vs fringe size")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    plt.show()





def plot_compute_sweep(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Within-AI compute concentration sweep visualization.

    Plots M1 effect on AI Lerner across a continuous compute-HHI range with
    bootstrap 95% confidence band. Tests whether M1's markup effect exhibits
    discrete tipping as within-AI concentration rises.
    """
    summary = compute_sweep_summary(df)
    fig, ax = plt.subplots(figsize=(10, 6))

    x = summary["compute_hhi"].values
    y = summary["M1_effect_on_lerner"].values
    lo, hi = summary["ci_low"].values, summary["ci_high"].values

    ax.plot(x, y, marker="o", linewidth=2, color="#cc3333",
            label="M1 effect on AI Lerner")
    ax.fill_between(x, lo, hi, alpha=0.25, color="#cc3333",
                    label="95% bootstrap CI")
    ax.axhline(0, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("Within-AI compute HHI (10000 × Σ share²)")
    ax.set_ylabel("M1 main effect on AI Lerner index")
    ax.set_title("M1 effect on markup vs compute concentration\n(continuous sweep for tipping mechanism test)")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    plt.show()


# ============================================================================
#  COMPUTE SHARE SWEEP (between-group, bounded regime test)
# ============================================================================

def _run_share_sweep_task(args: tuple) -> dict:
    """Worker for between-group AI compute share sweep.
    Holds within-AI distribution uniform; varies AI's total share of compute.
    Toggles M1 via beta; M2 active at default, M3 off.
    """
    scenario, compute_shares, ai_share, beta, mc_idx, horizon, seed_base = args
    params_dict = dict(default_params().__dict__)
    params_dict["beta"] = beta
    params = ModelParameters(**params_dict)

    seed = seed_base + abs(hash((scenario.code, float(ai_share), float(beta), mc_idx))) % 10_000
    model = ElectricityMarketABM(
        scenario=scenario, params=params, horizon_months=horizon, seed=seed,
        compute_shares_override=compute_shares,
    )
    df = model.run()
    return {
        "scenario": scenario.code,
        "ai_total_share": float(ai_share),
        "beta": float(beta),
        "M1": beta > 0,
        "mc_index": mc_idx,
        "final_hhi": float(df["hhi"].iloc[-1]),
        "final_ai_lerner": float(df["ai_lerner"].iloc[-1]),
        "final_ai_cap_share": float(df["ai_capacity_share"].iloc[-1]),
        "final_ai_market_share": float(df["ai_market_share"].iloc[-1]),
        "n_acquisitions": int(df["cumulative_acquisitions"].iloc[-1]),
        "final_ipp_exit": float(df["ipp_exit_rate"].iloc[-1]),
        "clearing_price": float(df["clearing_price"].mean()),
    }


def run_compute_share_sweep(
    scenario: ScenarioConfig,
    ai_total_share_grid: tuple = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90),
    n_ai: int = 5,
    n_mc: int = 50,
    horizon: int = 120,
    seed_base: int = 42,
    workers: int = 1,
    beta_on: float = 0.4,
    beta_off: float = 0.0,
) -> pd.DataFrame:
    """Continuous sweep of AI aggregate compute share. Holds within-AI distribution
    uniform (HHI floor = 10000/n_ai = 2000 for n_ai=5).

    Total runs: len(grid) × 2 × n_mc (e.g. 8 × 2 × 50 = 800 at defaults).
    """
    tasks = []
    for ai_share in ai_total_share_grid:
        shares = np.ones(n_ai) * (ai_share / n_ai)
        for beta in (beta_on, beta_off):
            for mc in range(n_mc):
                tasks.append((scenario, shares, ai_share, beta, mc, horizon, seed_base))
    results = _parallel_execute(_run_share_sweep_task, tasks, workers, desc="ShareSweep")
    return pd.DataFrame(results)


def compute_share_sweep_summary(df: pd.DataFrame, B: int = 1000, seed: int = 42) -> pd.DataFrame:
    """M1 effect on AI Lerner at each AI-share level with bootstrap CIs."""
    rng = np.random.default_rng(seed)
    rows = []
    for ai_share in sorted(df["ai_total_share"].unique()):
        sub = df[df["ai_total_share"] == ai_share]
        on = sub[sub["M1"]]["final_ai_lerner"].values
        off = sub[~sub["M1"]]["final_ai_lerner"].values
        effect = on.mean() - off.mean()
        n_on, n_off = len(on), len(off)
        diffs = np.empty(B)
        for b in range(B):
            d_on = on[rng.integers(0, n_on, n_on)].mean()
            d_off = off[rng.integers(0, n_off, n_off)].mean()
            diffs[b] = d_on - d_off
        rows.append({
            "ai_total_share": round(float(ai_share), 2),
            "M1_effect_on_lerner": round(effect, 4),
            "ci_low": round(float(np.percentile(diffs, 2.5)), 4),
            "ci_high": round(float(np.percentile(diffs, 97.5)), 4),
            "ai_lerner_M1on": round(float(on.mean()), 4),
            "ai_lerner_M1off": round(float(off.mean()), 4),
        })
    return pd.DataFrame(rows)


def plot_compute_share_sweep(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Two-panel: M1 effect curve + underlying Lerner levels under M1 on/off."""
    summary = compute_share_sweep_summary(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = summary["ai_total_share"].values
    y = summary["M1_effect_on_lerner"].values
    lo, hi = summary["ci_low"].values, summary["ci_high"].values
    axes[0].plot(x, y, marker="o", linewidth=2, color="#cc3333", label="M1 effect on AI Lerner")
    axes[0].fill_between(x, lo, hi, alpha=0.25, color="#cc3333", label="95% bootstrap CI")
    axes[0].axhline(0, ls=":", color="gray", alpha=0.5)
    axes[0].axvline(0.30, ls="--", color="#6699cc", alpha=0.6, label="L regime (0.30)")
    axes[0].axvline(0.70, ls="--", color="#cc6699", alpha=0.6, label="H regime (0.70)")
    axes[0].set_xlabel("AI total compute share")
    axes[0].set_ylabel("M1 main effect on AI Lerner index")
    axes[0].set_title("Between-group dimension:\nM1 effect vs AI compute share")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="best")

    axes[1].plot(x, summary["ai_lerner_M1on"], marker="o", linewidth=2,
                 color="#cc3333", label="M1 on (β=0.4)")
    axes[1].plot(x, summary["ai_lerner_M1off"], marker="s", linewidth=2,
                 color="#6699cc", label="M1 off (β=0)")
    axes[1].axvline(0.30, ls="--", color="#6699cc", alpha=0.6)
    axes[1].axvline(0.70, ls="--", color="#cc6699", alpha=0.6)
    axes[1].set_xlabel("AI total compute share")
    axes[1].set_ylabel("AI Lerner index (mean)")
    axes[1].set_title("AI Lerner under M1 on vs off")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    plt.show()


# ============================================================================
#  COUNTERFACTUAL POLICY SIMULATIONS (v0.3 NEW — for Table 7)
# ============================================================================
#
# Four policies converted from §5.2 extrapolation to simulation contrasts:
#
#   1. run_counterfactual_breakup       — Implication 1: AI break-up from 0.80 → 0.50
#   2. run_counterfactual_ppa_disclosure — Implication 2: Non-AI PPA 0.10 → 0.30
#   3. run_counterfactual_cost_disclosure — Implication 3: κ_acq distribution narrow
#   4. run_counterfactual_lockup_indicator — Implication 4: Early-warning trigger
#
# Each returns a DataFrame with column `arm` ∈ {baseline, intervention}.


def _run_cf_breakup_task(args: tuple) -> dict:
    """Worker: AI compute share 0.80 (baseline) vs 0.50 (intervention)."""
    scenario, ai_share, arm, mc_idx, horizon, seed_base, n_ai = args
    shares = np.ones(n_ai) * (ai_share / n_ai)
    seed = seed_base + abs(hash((scenario.code, arm, mc_idx))) % 10_000

    model = ElectricityMarketABM(
        scenario=scenario, horizon_months=horizon, seed=seed,
        compute_shares_override=shares,
    )
    df = model.run()
    return {
        "arm": arm,
        "ai_share": float(ai_share),
        "mc_index": mc_idx,
        "final_hhi": float(df["hhi"].iloc[-1]),
        "final_ai_lerner": float(df["ai_lerner"].iloc[-1]),
        "final_ai_cap_share": float(df["ai_capacity_share"].iloc[-1]),
        "clearing_price": float(df["clearing_price"].mean()),
        "regulator_activated": int(df["hhi"].max() >= scenario.hhi_trigger),
    }


def run_counterfactual_breakup(
    scenario: ScenarioConfig,
    baseline_share: float = 0.80,
    intervention_share: float = 0.50,
    n_ai: int = 5,
    n_mc: int = 50,
    horizon: int = 120,
    seed_base: int = 42,
    workers: int = 1,
) -> pd.DataFrame:
    """Implication 1: Test whether AI break-up from saturation regime (0.80) to
    active regime (0.50) improves or worsens AI markup. Hypothesis: counter-
    intuitive INCREASE in Lerner because break-up moves the market from saturation
    (intra-AI competition compresses markups) into the active regime.

    Returns DataFrame with 2 × n_mc rows; column `arm` ∈ {baseline_080, breakup_050}.
    """
    tasks = []
    for ai_share, arm in [(baseline_share, f"baseline_{int(baseline_share*100):03d}"),
                          (intervention_share, f"breakup_{int(intervention_share*100):03d}")]:
        for mc in range(n_mc):
            tasks.append((scenario, ai_share, arm, mc, horizon, seed_base, n_ai))
    results = _parallel_execute(_run_cf_breakup_task, tasks, workers, desc="CFBreakup")
    return pd.DataFrame(results)


def _run_cf_ppa_task(args: tuple) -> dict:
    """Worker: non-AI PPA penetration intervention. Modeled as shifting
    bigtech_ppa_share that the model uses for PPA capacity allocation."""
    scenario, non_ai_ppa_share, arm, mc_idx, horizon, seed_base = args
    # Approximate the intervention by adjusting the scenario's effective PPA share.
    # The base scenario's bigtech_ppa_share applies to AI firms; non-AI PPA is an
    # additional buffer affecting the IPP pool available for M2 lock-up.
    # We implement this as a parameter that increases m2_distress_threshold
    # proportionally (less aggressive M2 because fewer targets available).
    params_dict = dict(default_params().__dict__)
    # Higher non-AI PPA → fewer IPPs available as targets → less M2 activity
    # Approximation: increase cooldown proportionally to non_ai_ppa_share
    extra_cooldown = int(round(24 * non_ai_ppa_share))
    params_dict["m2_cooldown_months"] = 12 + extra_cooldown
    params = ModelParameters(**params_dict)

    seed = seed_base + abs(hash((scenario.code, arm, mc_idx))) % 10_000
    model = ElectricityMarketABM(
        scenario=scenario, params=params, horizon_months=horizon, seed=seed,
    )
    df = model.run()
    return {
        "arm": arm,
        "non_ai_ppa_share": float(non_ai_ppa_share),
        "mc_index": mc_idx,
        "final_hhi": float(df["hhi"].iloc[-1]),
        "final_ai_lerner": float(df["ai_lerner"].iloc[-1]),
        "final_ai_cap_share": float(df["ai_capacity_share"].iloc[-1]),
        "n_acquisitions": int(df["cumulative_acquisitions"].iloc[-1]),
        "clearing_price": float(df["clearing_price"].mean()),
    }


def run_counterfactual_ppa_disclosure(
    scenario: ScenarioConfig,
    baseline_non_ai_ppa: float = 0.10,
    intervention_non_ai_ppa: float = 0.30,
    n_mc: int = 50,
    horizon: int = 120,
    seed_base: int = 42,
    workers: int = 1,
) -> pd.DataFrame:
    """Implication 2: Non-AI PPA disclosure increases diverse long-term contracting
    from 0.10 → 0.30, which depletes the IPP target pool available to M2.
    Hypothesis: HHI decreases under intervention because fewer lock-up targets remain.
    """
    tasks = []
    for share, arm in [(baseline_non_ai_ppa, f"baseline_{int(baseline_non_ai_ppa*100):03d}"),
                       (intervention_non_ai_ppa, f"disclosure_{int(intervention_non_ai_ppa*100):03d}")]:
        for mc in range(n_mc):
            tasks.append((scenario, share, arm, mc, horizon, seed_base))
    results = _parallel_execute(_run_cf_ppa_task, tasks, workers, desc="CFPPA")
    return pd.DataFrame(results)


def _run_cf_cost_task(args: tuple) -> dict:
    """Worker: κ_acq distribution narrowing. Models acquisition-pricing oversight
    by clamping the cost multiplier to a tight range around 1.0.
    """
    scenario, cost_mult, arm, mc_idx, horizon, seed_base = args
    params_dict = dict(default_params().__dict__)
    params_dict["acquisition_cost_multiplier"] = float(cost_mult)
    params = ModelParameters(**params_dict)

    seed = seed_base + abs(hash((scenario.code, arm, mc_idx))) % 10_000
    model = ElectricityMarketABM(
        scenario=scenario, params=params, horizon_months=horizon, seed=seed,
    )
    df = model.run()
    return {
        "arm": arm,
        "acquisition_cost_multiplier": float(cost_mult),
        "mc_index": mc_idx,
        "final_hhi": float(df["hhi"].iloc[-1]),
        "final_ai_lerner": float(df["ai_lerner"].iloc[-1]),
        "final_ai_cap_share": float(df["ai_capacity_share"].iloc[-1]),
        "n_acquisitions": int(df["cumulative_acquisitions"].iloc[-1]),
    }


def run_counterfactual_cost_disclosure(
    scenario: ScenarioConfig,
    n_mc: int = 50,
    horizon: int = 120,
    seed_base: int = 42,
    workers: int = 1,
    intervention_strength: str = "narrow",
    cost_range_half_width: Optional[float] = None,
) -> pd.DataFrame:
    """Implication 3: Section 203 cost-multiplier disclosure narrows the effective
    range of κ_acq. Baseline: high variance across acquirer types (model uses 0.7
    in half of runs, 1.3 in the other half).

    Two ways to specify intervention magnitude:
    1. intervention_strength (categorical, default "narrow"):
       - "narrow": κ_acq ∈ {0.9, 1.0, 1.1}
       - "very_narrow": κ_acq ∈ {0.99, 1.00, 1.01}
    2. cost_range_half_width (continuous, overrides intervention_strength when set):
       - Float w → κ_acq sampled from {1-w, 1.0, 1+w}
       - Allows dose-response sweep (e.g. w ∈ [0.30, 0.10, 0.05, 0.01, 0.001])

    Hypothesis: HHI decreases because the most aggressive acquisitions (low κ_acq)
    are prevented by transparency. Smaller half-widths = stricter disclosure.
    """
    rng = np.random.default_rng(seed_base)
    tasks = []
    # Baseline arm: alternate between aggressive (0.7) and conservative (1.3)
    for mc in range(n_mc):
        cost_mult = 0.7 if mc % 2 == 0 else 1.3
        tasks.append((scenario, cost_mult, "baseline_wide", mc, horizon, seed_base))
    # Intervention arm: build grid
    if cost_range_half_width is not None:
        w = float(cost_range_half_width)
        narrow_grid = [1.0 - w, 1.0, 1.0 + w]
        arm_label = f"disclosure_w{w:.4f}".rstrip("0").rstrip(".")
    elif intervention_strength == "very_narrow":
        narrow_grid = [0.99, 1.00, 1.01]
        arm_label = "disclosure_very_narrow"
    else:
        narrow_grid = [0.9, 1.0, 1.1]
        arm_label = "disclosure_narrow"
    for mc in range(n_mc):
        cost_mult = narrow_grid[mc % 3]
        tasks.append((scenario, cost_mult, arm_label, mc, horizon, seed_base))
    results = _parallel_execute(_run_cf_cost_task, tasks, workers, desc="CFCost")
    return pd.DataFrame(results)


def _run_cf_indicator_task(args: tuple) -> dict:
    """Worker: lock-up rate early-warning indicator. Reduces effective hhi_trigger
    when cumulative lock-up rate exceeds 0.50 within 24 months.

    Implemented as scenario configuration with lowered hhi_trigger for the
    intervention arm. Approximates the indicator's policy effect: aggressive
    monitoring kicks in earlier.
    """
    scenario, indicator_active, arm, mc_idx, horizon, seed_base, intervention_trigger = args

    if indicator_active:
        # Intervention: scenario with hhi_trigger lowered (default 1200, can be set lower)
        class _ModifiedScenario:
            def __init__(self, base, trigger):
                self.code = base.code + "_INDIC"
                self.compute = base.compute
                self.ppa = base.ppa
                self.regulation = base.regulation
                self._trigger = trigger
            @property
            def hhi_trigger(self): return self._trigger
            @property
            def reg_delay_months(self): return 0
            @property
            def ai_total_compute_share(self): return scenario.ai_total_compute_share
            @property
            def bigtech_ppa_share(self): return scenario.bigtech_ppa_share

        scen_to_use = _ModifiedScenario(scenario, float(intervention_trigger))
    else:
        scen_to_use = scenario

    seed = seed_base + abs(hash((scenario.code, arm, mc_idx))) % 10_000
    model = ElectricityMarketABM(
        scenario=scen_to_use, horizon_months=horizon, seed=seed,
    )
    df = model.run()
    return {
        "arm": arm,
        "indicator_active": bool(indicator_active),
        "intervention_trigger": float(intervention_trigger) if indicator_active else None,
        "mc_index": mc_idx,
        "final_hhi": float(df["hhi"].iloc[-1]),
        "max_hhi": float(df["hhi"].max()),
        "final_ai_lerner": float(df["ai_lerner"].iloc[-1]),
        "regulator_activated": int(df["hhi"].max() >= (float(intervention_trigger) if indicator_active else scenario.hhi_trigger)),
        "n_acquisitions": int(df["cumulative_acquisitions"].iloc[-1]),
    }


def run_counterfactual_lockup_indicator(
    scenario: ScenarioConfig,
    n_mc: int = 50,
    horizon: int = 120,
    seed_base: int = 42,
    workers: int = 1,
    intervention_hhi_trigger: float = 1200.0,
) -> pd.DataFrame:
    """Implication 4: Lock-up rate early-warning indicator. Approximated by
    lowering the hhi_trigger in the intervention arm (default 1200, configurable).

    Hypothesis: regulator activation rate increases substantially under the
    intervention even when absolute HHI remains within "normal" range.

    Parameters
    ----------
    intervention_hhi_trigger : float, default 1200.0
        Lowered trigger for the intervention arm. Set lower (e.g. 900) for
        more aggressive scenarios where mean HHI ~ 800-900.
    """
    tasks = []
    for arm, active in [("baseline_no_indicator", False),
                        ("indicator_active", True)]:
        for mc in range(n_mc):
            tasks.append((scenario, active, arm, mc, horizon, seed_base,
                         intervention_hhi_trigger))
    results = _parallel_execute(_run_cf_indicator_task, tasks, workers, desc="CFIndicator")
    return pd.DataFrame(results)


def counterfactual_summary(
    df: pd.DataFrame,
    outcome: str = "final_hhi",
    B: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap inference for counterfactual contrasts.

    Returns dict with baseline_mean, intervention_mean, effect_size, ci_low,
    ci_high, p_value.
    """
    rng = np.random.default_rng(seed)
    arms = df["arm"].unique()
    if len(arms) != 2:
        raise ValueError(f"Expected exactly 2 arms, got: {arms}")
    # Identify baseline vs intervention by arm name
    baseline_arm = [a for a in arms if "baseline" in a.lower()][0]
    intervention_arm = [a for a in arms if a != baseline_arm][0]

    baseline_vals = df[df["arm"] == baseline_arm][outcome].values
    intervention_vals = df[df["arm"] == intervention_arm][outcome].values

    baseline_mean = baseline_vals.mean()
    intervention_mean = intervention_vals.mean()
    effect = intervention_mean - baseline_mean

    n_b, n_i = len(baseline_vals), len(intervention_vals)
    diffs = np.empty(B)
    for b in range(B):
        b_resample = baseline_vals[rng.integers(0, n_b, n_b)].mean()
        i_resample = intervention_vals[rng.integers(0, n_i, n_i)].mean()
        diffs[b] = i_resample - b_resample
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_val = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))

    return {
        "baseline_arm": baseline_arm,
        "intervention_arm": intervention_arm,
        "baseline_mean": float(baseline_mean),
        "intervention_mean": float(intervention_mean),
        "effect_size": float(effect),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_val),
    }


def plot_counterfactuals(
    dfs: dict,
    save_path: Optional[Path] = None,
) -> None:
    """4-panel comparison of counterfactual outcomes.

    Parameters
    ----------
    dfs : dict
        Keys must include: 'breakup', 'ppa', 'cost', 'indicator'.
        Each value is the DataFrame returned by the corresponding run_counterfactual_*.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: Break-up (Lerner)
    df_b = dfs["breakup"]
    arms = sorted(df_b["arm"].unique())
    for i, arm in enumerate(arms):
        vals = df_b[df_b["arm"] == arm]["final_ai_lerner"]
        axes[0, 0].boxplot([vals], positions=[i], widths=0.5, patch_artist=True,
                            boxprops=dict(facecolor=["lightblue", "salmon"][i]))
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(arms, fontsize=9)
    axes[0, 0].set_ylabel("AI Lerner index")
    axes[0, 0].set_title("Counterfactual 1: Break-up\n(0.80 saturation → 0.50 active)")
    axes[0, 0].grid(alpha=0.3)

    # Panel 2: PPA disclosure (HHI)
    df_p = dfs["ppa"]
    arms = sorted(df_p["arm"].unique())
    for i, arm in enumerate(arms):
        vals = df_p[df_p["arm"] == arm]["final_hhi"]
        axes[0, 1].boxplot([vals], positions=[i], widths=0.5, patch_artist=True,
                            boxprops=dict(facecolor=["lightblue", "salmon"][i]))
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(arms, fontsize=9)
    axes[0, 1].set_ylabel("Final HHI")
    axes[0, 1].set_title("Counterfactual 2: Non-AI PPA disclosure\n(0.10 → 0.30)")
    axes[0, 1].grid(alpha=0.3)

    # Panel 3: Cost disclosure (HHI)
    df_c = dfs["cost"]
    arms = sorted(df_c["arm"].unique())
    for i, arm in enumerate(arms):
        vals = df_c[df_c["arm"] == arm]["final_hhi"]
        axes[1, 0].boxplot([vals], positions=[i], widths=0.5, patch_artist=True,
                            boxprops=dict(facecolor=["lightblue", "salmon"][i]))
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(arms, fontsize=9)
    axes[1, 0].set_ylabel("Final HHI")
    axes[1, 0].set_title("Counterfactual 3: §203 cost disclosure\n(wide κ → narrow κ)")
    axes[1, 0].grid(alpha=0.3)

    # Panel 4: Lock-up indicator (regulator activation)
    df_i = dfs["indicator"]
    arms = sorted(df_i["arm"].unique())
    activation = [df_i[df_i["arm"] == arm]["regulator_activated"].mean() * 100 for arm in arms]
    axes[1, 1].bar([0, 1], activation, color=["lightblue", "salmon"])
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(arms, fontsize=9)
    axes[1, 1].set_ylabel("Regulator activation rate (%)")
    axes[1, 1].set_title("Counterfactual 4: Early-warning indicator\n(HHI trigger 1800 → 1200)")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(alpha=0.3, axis="y")

    plt.suptitle("Policy counterfactual simulations (n_MC = 50 per arm)", fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    plt.show()


# ============================================================================
#  END OF v0.3 ADDITIONS
# ============================================================================
