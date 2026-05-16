"""
inference.py — Statistical inference utilities for ABM mechanism decomposition.

Adds rigorous statistical inference on top of analysis.py's point estimates:

- bootstrap_ci(): non-parametric 95% CI via B resamples for any estimator
- decomp_effects_with_ci(): main effects + 2-way interactions with bootstrap CIs
- decomp_anova(): two-way / three-way ANOVA F-tests with p-values
- forest_plot(): publication-quality forest plot of effects with CIs

Designed as a drop-in supplement to analysis.py:

    from analysis import run_mechanism_decomposition
    from inference import decomp_effects_with_ci, decomp_anova, forest_plot

    df = run_mechanism_decomposition(scenario, n_mc=30, workers=16)
    effects = decomp_effects_with_ci(df, metric='final_hhi', B=1000)
    anova_table = decomp_anova(df, metric='final_hhi')
    forest_plot(effects, save_path='figures/forest_hhi.png')

Author: Chankook Park (HUFS)
Version: inference-0.1 (May 2026)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn(
        "statsmodels not installed. ANOVA functions disabled. "
        "Install with: pip install statsmodels"
    )


# ============================================================================
#  GENERIC BOOTSTRAP
# ============================================================================


def bootstrap_ci(
    data_groups: dict[str, np.ndarray],
    estimator: Callable[[dict[str, np.ndarray]], float],
    B: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    """
    Compute bootstrap CI for an arbitrary estimator over grouped samples.

    Parameters
    ----------
    data_groups : dict mapping group label -> 1d numpy array of observations
    estimator   : function taking a dict of resampled arrays, returning a scalar
    B           : number of bootstrap resamples (default 1000)
    alpha       : significance level for two-sided CI (default 0.05 -> 95% CI)
    rng         : numpy Generator for reproducibility (default: fresh)

    Returns
    -------
    (point_estimate, ci_low, ci_high)

    The bootstrap resamples *within each group* independently (stratified),
    preserving sample sizes per group. This is the standard approach for
    factorial-design effects where each cell is a separate stratum.
    """
    rng = rng or np.random.default_rng()
    point = estimator(data_groups)
    boots = np.empty(B)
    for b in range(B):
        resampled = {
            k: rng.choice(v, size=len(v), replace=True)
            for k, v in data_groups.items()
        }
        boots[b] = estimator(resampled)
    ci_low = np.quantile(boots, alpha / 2)
    ci_high = np.quantile(boots, 1 - alpha / 2)
    return point, ci_low, ci_high


def _bootstrap_pvalue(
    data_groups: dict[str, np.ndarray],
    estimator: Callable[[dict[str, np.ndarray]], float],
    B: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Two-sided bootstrap p-value for null hypothesis: estimator = 0.

    Computes the bootstrap distribution of (estimate - point) and asks how
    far the point lies in the distribution. p = 2 * min(P(boot <= 0), P(boot >= 0))
    is the standard percentile-method p-value. Uses the same stratified
    resampling as bootstrap_ci.
    """
    rng = rng or np.random.default_rng()
    point = estimator(data_groups)
    boots = np.empty(B)
    for b in range(B):
        resampled = {
            k: rng.choice(v, size=len(v), replace=True)
            for k, v in data_groups.items()
        }
        boots[b] = estimator(resampled)
    # Two-sided percentile p-value
    p_left = np.mean(boots <= 0)
    p_right = np.mean(boots >= 0)
    return float(2 * min(p_left, p_right))


# ============================================================================
#  MAIN EFFECTS + INTERACTIONS WITH CIs
# ============================================================================


def decomp_effects_with_ci(
    df: pd.DataFrame,
    metric: str = "final_hhi",
    B: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute mechanism main effects and 2-way interactions with bootstrap CIs.

    Inputs
    ------
    df     : output of run_mechanism_decomposition() (240 rows: 8 configs × 30 MC)
    metric : column name to analyze (e.g. 'final_hhi', 'final_ai_lerner')
    B      : bootstrap resamples

    Returns
    -------
    DataFrame with columns:
        effect            : 'M1_main' / 'M2_main' / 'M3_main' / 'M1xM2' / ...
        estimate          : point estimate
        ci_low, ci_high   : 95% bootstrap CI
        p_value           : bootstrap p-value (H0: effect=0)
        n_obs             : total observations entering the contrast
        significant       : ★★★ / ★★ / ★ / —
    """
    rng = np.random.default_rng(seed)
    rows = []

    # --- Main effects: difference of means between mech=True vs mech=False ---
    for mech in ["M1", "M2", "M3"]:
        on_vals = df[df[mech]][metric].values
        off_vals = df[~df[mech]][metric].values

        def main_effect_estimator(groups):
            return float(groups["on"].mean() - groups["off"].mean())

        groups = {"on": on_vals, "off": off_vals}
        point, lo, hi = bootstrap_ci(groups, main_effect_estimator, B=B, rng=rng)
        rng_p = np.random.default_rng(seed + 1)
        pval = _bootstrap_pvalue(groups, main_effect_estimator, B=B, rng=rng_p)

        rows.append({
            "effect": f"{mech}_main",
            "estimate": round(point, 3),
            "ci_low": round(lo, 3),
            "ci_high": round(hi, 3),
            "p_value": round(pval, 4),
            "n_obs": len(on_vals) + len(off_vals),
            "significant": _star(pval),
        })

    # --- 2-way interactions: ANOVA-style synergy term ---
    for a, b in [("M1", "M2"), ("M1", "M3"), ("M2", "M3")]:
        # 4 cells: (a=T,b=T), (a=T,b=F), (a=F,b=T), (a=F,b=F)
        cell_TT = df[df[a] & df[b]][metric].values
        cell_TF = df[df[a] & ~df[b]][metric].values
        cell_FT = df[~df[a] & df[b]][metric].values
        cell_FF = df[~df[a] & ~df[b]][metric].values

        def interaction_estimator(groups):
            # (both - only_a - only_b + neither) / 2
            return float(
                (groups["TT"].mean() - groups["TF"].mean()
                 - groups["FT"].mean() + groups["FF"].mean()) / 2
            )

        groups = {"TT": cell_TT, "TF": cell_TF, "FT": cell_FT, "FF": cell_FF}
        point, lo, hi = bootstrap_ci(groups, interaction_estimator, B=B, rng=rng)
        rng_p = np.random.default_rng(seed + 2)
        pval = _bootstrap_pvalue(groups, interaction_estimator, B=B, rng=rng_p)

        rows.append({
            "effect": f"{a}x{b}",
            "estimate": round(point, 3),
            "ci_low": round(lo, 3),
            "ci_high": round(hi, 3),
            "p_value": round(pval, 4),
            "n_obs": sum(len(v) for v in groups.values()),
            "significant": _star(pval),
        })

    return pd.DataFrame(rows)


def _star(p: float) -> str:
    """Significance stars for quick visual scanning."""
    if p < 0.001:
        return "★★★"
    elif p < 0.01:
        return "★★"
    elif p < 0.05:
        return "★"
    else:
        return "—"


# ============================================================================
#  ANOVA (statsmodels)
# ============================================================================


def decomp_anova(
    df: pd.DataFrame,
    metric: str = "final_hhi",
    include_3way: bool = True,
) -> pd.DataFrame:
    """
    Three-way ANOVA (Type II SS) on the M1×M2×M3 factorial design.

    Returns the standard ANOVA table with F-stats and p-values for:
        M1, M2, M3                         (main effects)
        M1:M2, M1:M3, M2:M3                (2-way interactions)
        M1:M2:M3                           (3-way interaction, optional)
        Residual

    This is the journal-canonical complement to the bootstrap CIs.
    Reviewers will look for one of these two; we provide both.
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required for ANOVA. pip install statsmodels")

    # Cast bools to ints so statsmodels treats them as factors via C()
    work = df.copy()
    for c in ["M1", "M2", "M3"]:
        work[c] = work[c].astype(int)

    if include_3way:
        formula = f"{metric} ~ C(M1)*C(M2)*C(M3)"
    else:
        formula = f"{metric} ~ C(M1) + C(M2) + C(M3) + C(M1):C(M2) + C(M1):C(M3) + C(M2):C(M3)"

    model = smf.ols(formula, data=work).fit()
    table = sm.stats.anova_lm(model, typ=2)
    table = table.round(4)
    # Rename for readability
    rename_map = {
        "C(M1)": "M1",
        "C(M2)": "M2",
        "C(M3)": "M3",
        "C(M1):C(M2)": "M1:M2",
        "C(M1):C(M3)": "M1:M3",
        "C(M2):C(M3)": "M2:M3",
        "C(M1):C(M2):C(M3)": "M1:M2:M3",
    }
    table.index = [rename_map.get(idx, idx) for idx in table.index]
    return table


# ============================================================================
#  FOREST PLOT (publication-quality)
# ============================================================================


def forest_plot(
    effects: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "Mechanism effects on HHI (95% bootstrap CI)",
    xlabel: str = "Effect size (HHI points)",
    show: bool = True,
) -> None:
    """
    Forest plot for main effects + interactions with 95% CIs.

    Inputs
    ------
    effects : DataFrame from decomp_effects_with_ci()
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(effects)
    y = np.arange(n)[::-1]  # top-to-bottom reading order

    # Color by significance
    colors = []
    for p in effects["p_value"]:
        if p < 0.05:
            colors.append("#1f77b4")  # blue (significant)
        else:
            colors.append("#999999")  # grey (not significant)

    # Draw error bars one row at a time so each can have its own color
    for i, row in enumerate(effects.itertuples()):
        xerr_low = row.estimate - row.ci_low
        xerr_high = row.ci_high - row.estimate
        ax.errorbar(
            row.estimate, y[i],
            xerr=[[xerr_low], [xerr_high]],
            fmt="o",
            markersize=8,
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            ecolor=colors[i],
            mfc=colors[i],
            mec="black",
            mew=1.2,
        )

    # Zero line
    ax.axvline(0, ls="--", color="black", alpha=0.4, lw=1)

    # Labels
    ax.set_yticks(y)
    labels = [
        f"{row.effect}  ({row.significant})"
        for row in effects.itertuples()
    ]
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    # Add annotation: estimate [ci_low, ci_high]
    xmax = max(effects["ci_high"].max(), 0)
    xmin = min(effects["ci_low"].min(), 0)
    span = xmax - xmin
    text_x = xmax + 0.05 * span
    # Adaptive decimals: small magnitudes need more precision
    max_abs = max(abs(xmin), abs(xmax))
    if max_abs < 1:
        fmt = "{:+.3f}"
    elif max_abs < 10:
        fmt = "{:+.2f}"
    else:
        fmt = "{:+.1f}"
    for i, row in enumerate(effects.itertuples()):
        ax.text(
            text_x, y[i],
            f"{fmt.format(row.estimate)}  [{fmt.format(row.ci_low)}, {fmt.format(row.ci_high)}]  p={row.p_value:.3f}",
            va="center", fontsize=8.5, family="monospace",
        )
    # Extend x-axis to fit annotations
    ax.set_xlim(xmin - 0.05 * span, xmax + 0.9 * span)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Forest plot saved: {save_path}")
    if show:
        plt.show()
    plt.close()


# ============================================================================
#  CROSS-SCENARIO COMPARISON
# ============================================================================


def compare_across_scenarios(
    effects_by_scenario: dict,
    effects_filter: Optional[list] = None,
    show_ci: bool = True,
) -> pd.DataFrame:
    """
    Compile a single comparison table from per-scenario effects DataFrames.

    Inputs
    ------
    effects_by_scenario : dict mapping scenario code -> output of decomp_effects_with_ci()
                          e.g., {"LLL": effects_df, "LLH": effects_df, ...}
    effects_filter      : optional list of effect names to include (default: all)
    show_ci             : if True, include "[ci_low, ci_high]" in each cell

    Returns
    -------
    DataFrame with effects as rows, scenarios as columns. Each cell shows
    estimate, significance star, and (optionally) CI bounds. Use to check
    whether the pattern (which effects are significant, what signs, what
    magnitudes) is stable across the 6-cell scenario design.

    Example
    -------
    >>> results = {}
    >>> for scen in SCENARIOS_6CELL:
    ...     df = run_mechanism_decomposition(scen, n_mc=150, workers=16)
    ...     results[scen.code] = decomp_effects_with_ci(df, metric="final_hhi")
    >>> print(compare_across_scenarios(results))
    """
    if not effects_by_scenario:
        raise ValueError("effects_by_scenario is empty")

    # Use the effects list from the first scenario as the canonical order
    first_scen = next(iter(effects_by_scenario.values()))
    if effects_filter is None:
        effects_filter = first_scen["effect"].tolist()

    # Determine adaptive decimal format from all values across scenarios
    all_vals = []
    for eff_df in effects_by_scenario.values():
        all_vals.extend(eff_df["estimate"].abs().tolist())
        all_vals.extend(eff_df["ci_low"].abs().tolist())
        all_vals.extend(eff_df["ci_high"].abs().tolist())
    max_abs = max(all_vals) if all_vals else 1
    if max_abs < 1:
        fmt_est = "{:+.3f}"
        fmt_ci = "{:+.3f}"
    elif max_abs < 10:
        fmt_est = "{:+.2f}"
        fmt_ci = "{:+.2f}"
    else:
        fmt_est = "{:+.0f}"
        fmt_ci = "{:+.0f}"

    rows = []
    for effect_name in effects_filter:
        row = {"effect": effect_name}
        for scen_code, eff_df in effects_by_scenario.items():
            sub = eff_df[eff_df["effect"] == effect_name]
            if len(sub) == 0:
                row[scen_code] = "—"
                continue
            r = sub.iloc[0]
            if show_ci:
                cell = (
                    f"{fmt_est.format(r['estimate'])} "
                    f"[{fmt_ci.format(r['ci_low'])},{fmt_ci.format(r['ci_high'])}] "
                    f"{r['significant']}"
                )
            else:
                cell = f"{fmt_est.format(r['estimate'])} {r['significant']}"
            row[scen_code] = cell
        rows.append(row)
    return pd.DataFrame(rows).set_index("effect")


# ============================================================================
#  COMPACT SUMMARY (text output for paper / cover letter)
# ============================================================================


def summarize_effects(
    effects: pd.DataFrame,
    anova: Optional[pd.DataFrame] = None,
    metric_name: str = "HHI",
) -> str:
    """
    Format a compact text summary for inclusion in manuscript or cover letter.

    Example output:
        M2_main  +207.0  [+142.3, +269.4]  p<0.001  (★★★)
        M1xM2    +60.2   [+12.1, +108.4]   p=0.013  (★)
        ...
    """
    # Adaptive decimal places based on magnitude of estimates
    max_abs = max(
        effects["estimate"].abs().max(),
        effects["ci_low"].abs().max(),
        effects["ci_high"].abs().max(),
    )
    if max_abs < 1:
        fmt = "{:+8.4f}"
    elif max_abs < 10:
        fmt = "{:+8.3f}"
    else:
        fmt = "{:+8.1f}"

    lines = [f"Mechanism effects on {metric_name} (bootstrap 95% CI, B=1000)",
             "-" * 70]
    for row in effects.itertuples():
        p_disp = "<0.001" if row.p_value < 0.001 else f"={row.p_value:.3f}"
        lines.append(
            f"  {row.effect:<10s} {fmt.format(row.estimate)}  "
            f"[{fmt.format(row.ci_low)}, {fmt.format(row.ci_high)}]  "
            f"p{p_disp}  ({row.significant})"
        )
    if anova is not None:
        lines.append("")
        lines.append("Three-way ANOVA (Type II SS)")
        lines.append("-" * 70)
        lines.append(anova.to_string())
    return "\n".join(lines)
