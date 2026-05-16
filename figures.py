"""
figures.py — Publication-quality figure generation for the manuscript.

Generates four figures for the Energy Policy submission:

Figure 1 — Single-scenario dynamics: 6-panel time series (HHI, price,
           Lerner, AI share, acquisitions, IPP exit) with MC 10-90 band.
Figure 2 — Mechanism decomposition: 8-config bar chart with bootstrap CIs,
           HHI and Lerner panels.
Figure 3 — Forest plot: main effects + interactions with 95% CIs and stars.
           (Generated via inference.forest_plot; included here for one-stop.)
Figure 4 — P3 stress test: 2-panel comparison (default vs stress preset)
           with regulator activation rates.

All figures: 300 DPI, Energy Policy column-width compliant (single or double).
Designed to be the entire figure package the manuscript submits.

Usage from Jupyter:
    from figures import generate_all_figures
    generate_all_figures(
        df_single_run,     # 1 scenario × 1 MC × 120 months (or panel data)
        df_decomp,         # mechanism decomposition output (8 × n_mc)
        effects_hhi,       # output of inference.decomp_effects_with_ci
        df_p3,             # P3 stress test output
        output_dir="figures",
    )

Author: Chankook Park (HUFS)
Version: figures-0.1 (May 2026)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Energy Policy figure standards
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
})


# ============================================================================
#  FIGURE 1 — SINGLE-SCENARIO 6-PANEL DYNAMICS
# ============================================================================


def figure1_single_scenario(
    df_panel: pd.DataFrame,
    save_path: Optional[Path] = None,
    scenario_label: str = "HHL",
    show: bool = True,
) -> None:
    """
    6-panel time series for one scenario across MC replications.

    Inputs
    ------
    df_panel : panel DataFrame with columns
        ['month', 'mc_index', 'hhi', 'clearing_price', 'ai_lerner',
         'ai_market_share', 'cumulative_acquisitions', 'ipp_exit_rate']
        Pass a single-scenario subset (e.g., df[df.scenario=='HHL']).
    """
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))

    panels = [
        ("hhi",                    "HHI (concentration)",     "HHI",              [(1800, "red", "DOJ 1,800"), (3000, "darkred", "DOJ 3,000")]),
        ("clearing_price",         "Clearing price",           "$/MWh",            []),
        ("ai_lerner",              "AI Lerner index",          "(p − MC)/p",       []),
        ("ai_market_share",        "AI firms' market share",   "Fraction",         []),
        ("cumulative_acquisitions","M2 acquisitions (cumulative)", "Count",         []),
        ("ipp_exit_rate",          "IPP exit rate",            "Fraction exited",  []),
    ]

    for ax, (col, title, ylabel, refs) in zip(axes.flat, panels):
        if col not in df_panel.columns:
            ax.text(0.5, 0.5, f"Column '{col}' missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        grouped = df_panel.groupby("month")[col]
        mean = grouped.mean()
        p10 = grouped.quantile(0.10)
        p90 = grouped.quantile(0.90)
        ax.fill_between(mean.index, p10.values, p90.values, alpha=0.2, color="steelblue", label="10-90% MC band")
        ax.plot(mean.index, mean.values, color="steelblue", lw=1.5, label="MC mean")
        for level, color, lbl in refs:
            ax.axhline(level, ls="--", color=color, alpha=0.6, lw=1, label=lbl)
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        if refs or col == "hhi":
            ax.legend(loc="best", framealpha=0.9)

    fig.suptitle(
        f"Figure 1. Wholesale-market dynamics under scenario {scenario_label} "
        f"({df_panel['mc_index'].nunique()} Monte Carlo replications, 120 months)",
        y=1.005,
    )
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure 1 saved: {save_path}")
    if show:
        plt.show()
    plt.close()


# ============================================================================
#  FIGURE 2 — MECHANISM DECOMPOSITION BAR CHART
# ============================================================================


def figure2_decomposition(
    df_decomp: pd.DataFrame,
    save_path: Optional[Path] = None,
    scenario_label: str = "HHL",
    show: bool = True,
) -> None:
    """
    8-configuration decomposition: HHI and Lerner panels with bootstrap CIs.

    Inputs
    ------
    df_decomp : output of run_mechanism_decomposition()
                columns: 'config', 'M1', 'M2', 'M3', 'final_hhi', 'final_ai_lerner', ...
    """
    config_order = [
        "None", "M1 only", "M2 only", "M3 only",
        "M1+M2", "M1+M3", "M2+M3", "All (full)",
    ]

    # Aggregate by config with bootstrap CI on the mean
    rng = np.random.default_rng(42)
    rows = []
    for cfg in config_order:
        sub = df_decomp[df_decomp["config"] == cfg]
        if len(sub) == 0:
            continue
        for metric in ("final_hhi", "final_ai_lerner"):
            vals = sub[metric].values
            mean = vals.mean()
            # Bootstrap CI on the mean
            B = 1000
            boots = np.empty(B)
            for b in range(B):
                boots[b] = rng.choice(vals, size=len(vals), replace=True).mean()
            ci_lo = np.quantile(boots, 0.025)
            ci_hi = np.quantile(boots, 0.975)
            rows.append({"config": cfg, "metric": metric, "mean": mean,
                         "ci_low": ci_lo, "ci_high": ci_hi})
    agg = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(config_order))
    # Color by number of mechanisms active
    n_mech = {"None": 0, "M1 only": 1, "M2 only": 1, "M3 only": 1,
              "M1+M2": 2, "M1+M3": 2, "M2+M3": 2, "All (full)": 3}
    color_map = ["#cccccc", "#9ecae1", "#fdae6b", "#cc3333"]
    colors = [color_map[n_mech[c]] for c in config_order]

    # Panel A: HHI
    hhi_sub = agg[agg["metric"] == "final_hhi"].set_index("config").reindex(config_order)
    yerr_lo = hhi_sub["mean"].values - hhi_sub["ci_low"].values
    yerr_hi = hhi_sub["ci_high"].values - hhi_sub["mean"].values
    axes[0].bar(x, hhi_sub["mean"].values, color=colors, edgecolor="black", linewidth=0.5,
                yerr=[yerr_lo, yerr_hi], capsize=4, ecolor="black", error_kw={"alpha": 0.7})
    axes[0].axhline(1800, ls="--", color="red", alpha=0.5, lw=1, label="DOJ 1,800 threshold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(config_order, rotation=25, ha="right")
    axes[0].set_ylabel("Final HHI (mean ± 95% bootstrap CI)")
    axes[0].set_title("(a) HHI by mechanism configuration")
    axes[0].legend(loc="upper left")
    for xi, m in zip(x, hhi_sub["mean"].values):
        axes[0].text(xi, m + 30, f"{m:.0f}", ha="center", fontsize=8.5)

    # Panel B: Lerner
    ler_sub = agg[agg["metric"] == "final_ai_lerner"].set_index("config").reindex(config_order)
    yerr_lo = ler_sub["mean"].values - ler_sub["ci_low"].values
    yerr_hi = ler_sub["ci_high"].values - ler_sub["mean"].values
    axes[1].bar(x, ler_sub["mean"].values, color=colors, edgecolor="black", linewidth=0.5,
                yerr=[yerr_lo, yerr_hi], capsize=4, ecolor="black", error_kw={"alpha": 0.7})
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(config_order, rotation=25, ha="right")
    axes[1].set_ylabel("AI Lerner index (mean ± 95% bootstrap CI)")
    axes[1].set_title("(b) AI Lerner by mechanism configuration")
    for xi, m in zip(x, ler_sub["mean"].values):
        axes[1].text(xi, m + 0.005, f"{m:.3f}", ha="center", fontsize=8.5)

    # Legend for color
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=color_map[0], edgecolor="black", label="0 mechanisms"),
        Patch(facecolor=color_map[1], edgecolor="black", label="1 mechanism"),
        Patch(facecolor=color_map[2], edgecolor="black", label="2 mechanisms"),
        Patch(facecolor=color_map[3], edgecolor="black", label="3 mechanisms"),
    ]
    axes[1].legend(handles=legend_elems, loc="lower right", title="Active count")

    n_mc = df_decomp.groupby("config").size().iloc[0]
    fig.suptitle(
        f"Figure 2. Mechanism decomposition under scenario {scenario_label} "
        f"(n_MC = {n_mc} per configuration; 95% bootstrap CIs)",
        y=1.02,
    )
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure 2 saved: {save_path}")
    if show:
        plt.show()
    plt.close()


# ============================================================================
#  FIGURE 3 — FOREST PLOT (delegates to inference.forest_plot but combines HHI + Lerner)
# ============================================================================


def figure3_forest(
    effects_hhi: pd.DataFrame,
    effects_lerner: pd.DataFrame,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Side-by-side forest plot for HHI and Lerner main effects + interactions.

    Inputs
    ------
    effects_hhi, effects_lerner : outputs of inference.decomp_effects_with_ci()
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, effects, metric, xlabel in zip(
        axes,
        [effects_hhi, effects_lerner],
        ["HHI", "AI Lerner"],
        ["Effect on HHI (points)", "Effect on Lerner (index)"],
    ):
        n = len(effects)
        y = np.arange(n)[::-1]
        max_abs = max(
            effects["estimate"].abs().max(),
            effects["ci_low"].abs().max(),
            effects["ci_high"].abs().max(),
        )
        fmt = "{:+.3f}" if max_abs < 1 else ("{:+.2f}" if max_abs < 10 else "{:+.0f}")

        for i, row in enumerate(effects.itertuples()):
            color = "#1f77b4" if row.p_value < 0.05 else "#999999"
            xerr_lo = row.estimate - row.ci_low
            xerr_hi = row.ci_high - row.estimate
            ax.errorbar(
                row.estimate, y[i],
                xerr=[[xerr_lo], [xerr_hi]],
                fmt="o", markersize=8,
                capsize=5, capthick=1.5, elinewidth=1.5,
                ecolor=color, mfc=color, mec="black", mew=1.2,
            )

        ax.axvline(0, ls="--", color="black", alpha=0.4, lw=1)
        ax.set_yticks(y)
        labels = [f"{r.effect}  ({r.significant})" for r in effects.itertuples()]
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_title(f"({'a' if metric == 'HHI' else 'b'}) {metric}")

        # Annotations
        xmax = max(effects["ci_high"].max(), 0)
        xmin = min(effects["ci_low"].min(), 0)
        span = xmax - xmin
        text_x = xmax + 0.05 * span
        for i, row in enumerate(effects.itertuples()):
            p_disp = "<0.001" if row.p_value < 0.001 else f"={row.p_value:.3f}"
            ax.text(
                text_x, y[i],
                f"{fmt.format(row.estimate)}  [{fmt.format(row.ci_low)}, {fmt.format(row.ci_high)}]  p{p_disp}",
                va="center", fontsize=8.5, family="monospace",
            )
        ax.set_xlim(xmin - 0.05 * span, xmax + 0.95 * span)

    fig.suptitle(
        "Figure 3. Mechanism main effects and 2-way interactions "
        "(bootstrap 95% CI, B=2000; blue = significant at α=0.05)",
        y=1.02,
    )
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure 3 saved: {save_path}")
    if show:
        plt.show()
    plt.close()


# ============================================================================
#  FIGURE 4 — P3 STRESS TEST
# ============================================================================


def figure4_p3_stress(
    df_p3: pd.DataFrame,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    P3 stress: 2-panel comparison of default vs aggressive M2 preset.

    Inputs
    ------
    df_p3 : output of run_p3_stress()
            columns: 'scenario', 'preset' (default/p3_stress), 'final_hhi',
                     'regulator_intervened', 'final_ipp_exit', 'n_acquisitions'
    """
    # Aggregate
    summary = df_p3.groupby(["scenario", "preset"]).agg(
        hhi_mean=("final_hhi", "mean"),
        hhi_sd=("final_hhi", "std"),
        reg_rate=("regulator_intervened", "mean"),
        ipp_exit=("final_ipp_exit", "mean"),
        n_acq=("n_acquisitions", "mean"),
    ).reset_index()
    scenarios = ["LLL", "LLH", "LHH", "HLH", "HHL", "HHH"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(scenarios))
    width = 0.4
    colors = {"default": "#6699cc", "p3_stress": "#cc3333"}

    # Panel A: HHI with thresholds
    for i, preset in enumerate(["default", "p3_stress"]):
        vals = [
            summary[(summary.scenario == s) & (summary.preset == preset)]["hhi_mean"].iloc[0]
            for s in scenarios
        ]
        sds = [
            summary[(summary.scenario == s) & (summary.preset == preset)]["hhi_sd"].iloc[0]
            for s in scenarios
        ]
        axes[0].bar(
            x + (i - 0.5) * width, vals, width,
            yerr=sds, capsize=3, ecolor="black", error_kw={"alpha": 0.5},
            label=preset.replace("_", " "), color=colors[preset],
            edgecolor="black", linewidth=0.5,
        )
    axes[0].axhline(1800, ls="--", color="red", alpha=0.6, lw=1, label="DOJ 1,800 (ex-ante)")
    axes[0].axhline(3000, ls="--", color="darkred", alpha=0.6, lw=1, label="DOJ 3,000 (ex-post)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenarios)
    axes[0].set_xlabel("Scenario")
    axes[0].set_ylabel("Final HHI (mean ± SD)")
    axes[0].set_title("(a) HHI: default vs aggressive M2 preset")
    axes[0].legend(loc="upper left", fontsize=8)

    # Panel B: Regulator activation rate
    for i, preset in enumerate(["default", "p3_stress"]):
        vals = [
            summary[(summary.scenario == s) & (summary.preset == preset)]["reg_rate"].iloc[0]
            for s in scenarios
        ]
        axes[1].bar(
            x + (i - 0.5) * width, vals, width,
            label=preset.replace("_", " "), color=colors[preset],
            edgecolor="black", linewidth=0.5,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenarios)
    axes[1].set_ylabel("Regulator intervention rate (fraction of MC)")
    axes[1].set_xlabel("Scenario")
    axes[1].set_title("(b) Regulator activation: default vs stress")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="upper left", fontsize=8)

    n_mc = df_p3.groupby(["scenario", "preset"]).size().iloc[0]
    fig.suptitle(
        f"Figure 4. P3 stress test: structural HHI ceiling persists below regulatory triggers "
        f"(n_MC = {n_mc} per cell)",
        y=1.03,
    )
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure 4 saved: {save_path}")
    if show:
        plt.show()
    plt.close()


# ============================================================================
#  ORCHESTRATOR — Generate all 4 figures at once
# ============================================================================


def generate_all_figures(
    df_panel: pd.DataFrame,
    df_decomp: pd.DataFrame,
    effects_hhi: pd.DataFrame,
    effects_lerner: pd.DataFrame,
    df_p3: pd.DataFrame,
    output_dir: str = "figures",
    scenario_label: str = "HHL",
) -> None:
    """
    Generate all four publication figures in one call.

    Each figure is saved as figure{N}.png (300 dpi) in output_dir.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    figure1_single_scenario(df_panel, save_path=out / "figure1_dynamics.png",
                            scenario_label=scenario_label, show=False)
    figure2_decomposition(df_decomp, save_path=out / "figure2_decomposition.png",
                          scenario_label=scenario_label, show=False)
    figure3_forest(effects_hhi, effects_lerner, save_path=out / "figure3_forest.png",
                   show=False)
    figure4_p3_stress(df_p3, save_path=out / "figure4_p3_stress.png", show=False)

    print(f"\nAll 4 figures generated in {out.resolve()}")
    print("  figure1_dynamics.png        — single-scenario 6-panel")
    print("  figure2_decomposition.png   — 8-config bar chart (HHI + Lerner)")
    print("  figure3_forest.png          — forest plot (HHI + Lerner)")
    print("  figure4_p3_stress.png       — P3 stress test 2-panel")
