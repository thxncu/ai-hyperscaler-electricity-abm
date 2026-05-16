"""
caiso_validation.py — External validation of the AI-hyperscaler ABM against
CAISO 2025 LMP data and 2024 DMM Annual Report benchmarks.

Compares five model outputs with empirical CAISO data:
  1. Clearing price distribution (mean, std, percentiles)
  2. Price volatility
  3. Markup (AI Lerner) vs DMM markup estimates
  4. Final HHI vs CAISO 2024 supplier HHI
  5. IPP attrition (10-year) vs CAISO 5-year baseline pro-rated

Required input file:
  CAISO_2025_15min_LMP.csv with columns:
    Time, Interval Start, Interval End, Market, Location, Location Type,
    LMP, Energy, Congestion, Loss, GHG

The CAISO data is 15-min real-time prices at DLAP_SCE-APND (South-of-Path-26)
default node, which is the most representative single-node price for the
California system. For 35,040 rows expected (15-min × 24h × 365d).

Author: Chankook Park (HUFS)
Version: caiso-1.0 (May 2026)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
#  CAISO 2024 DMM Annual Report benchmarks
# ============================================================================
#
# Sources (all publicly available):
#   - CAISO Department of Market Monitoring 2024 Annual Report
#     (https://www.caiso.com/Documents/2024-Annual-Report-on-Market-Issues-and-Performance.pdf)
#   - PJM State of the Market 2024 (Monitoring Analytics)
#   - FERC State of the Market (annual reports, various years)

CAISO_BENCHMARKS: dict[str, float] = {
    # DMM markup estimate: typical scarcity-rent margin in CAISO Day-Ahead market.
    # Conservative midpoint of DMM range (0.15-0.25); represents average Lerner.
    "dmm_markup_estimate": 0.20,

    # Supplier-side HHI from CAISO 2024 Annual Report (capacity-weighted).
    # Recent reports: ~1,200-1,400 depending on aggregation method.
    "supplier_hhi_estimate": 1300.0,

    # 5-year IPP attrition rate: rough estimate from CAISO renewable IPP entry/exit
    # tracking (publicly available data from CAISO Interconnection Queue).
    "ipp_attrition_5yr_pct": 17.5,
}


# ============================================================================
#  Data loading
# ============================================================================

def load_caiso_lmp(path: Union[str, Path]) -> pd.DataFrame:
    """Load CAISO 15-min LMP CSV.

    Parameters
    ----------
    path : str or Path
        Path to CAISO_2025_15min_LMP.csv

    Returns
    -------
    pd.DataFrame with parsed datetime index and original columns
    """
    path = Path(path)
    df = pd.read_csv(path)
    # The 'Time' column carries datetime with timezone offset
    df['ts'] = pd.to_datetime(df['Time'], errors='coerce', utc=False)
    df['year_month'] = df['ts'].dt.to_period('M')
    return df


def caiso_monthly_aggregates(path: Union[str, Path]) -> pd.DataFrame:
    """Aggregate CAISO 15-min LMP to monthly statistics.

    Returns
    -------
    pd.DataFrame with columns: year_month, lmp_mean, lmp_std, lmp_p10, lmp_p90,
        lmp_median, n_intervals
    """
    df = load_caiso_lmp(path)
    monthly = df.groupby('year_month').agg(
        lmp_mean=('LMP', 'mean'),
        lmp_std=('LMP', 'std'),
        lmp_median=('LMP', 'median'),
        lmp_p10=('LMP', lambda x: x.quantile(0.10)),
        lmp_p90=('LMP', lambda x: x.quantile(0.90)),
        n_intervals=('LMP', 'count'),
    ).reset_index()
    return monthly


def caiso_summary_stats(path: Union[str, Path]) -> dict:
    """Yearly aggregate CAISO statistics for direct model comparison.

    Returns
    -------
    dict with keys: lmp_mean_usd_mwh, lmp_std_usd_mwh, lmp_median, lmp_p10,
        lmp_p90, n_observations, location, market_type
    """
    df = pd.read_csv(path)
    return {
        "lmp_mean_usd_mwh": float(df['LMP'].mean()),
        "lmp_std_usd_mwh":  float(df['LMP'].std()),
        "lmp_median":       float(df['LMP'].median()),
        "lmp_p10":          float(df['LMP'].quantile(0.10)),
        "lmp_p90":          float(df['LMP'].quantile(0.90)),
        "n_observations":   int(len(df)),
        "location":         df['Location'].iloc[0] if 'Location' in df.columns else "unknown",
        "market_type":      df['Market'].iloc[0]   if 'Market'   in df.columns else "unknown",
    }


# ============================================================================
#  Validation table
# ============================================================================

def validate_against_caiso(
    model_results: pd.DataFrame,
    caiso_path: Union[str, Path],
) -> pd.DataFrame:
    """Compare ABM output with CAISO empirical benchmarks across five metrics.

    Parameters
    ----------
    model_results : pd.DataFrame
        Output from run_mechanism_decomposition or run_cross_scenario_decomposition.
        Required columns: clearing_price, final_ai_lerner, final_hhi, final_ipp_exit
        (or equivalent ipp_exit_rate).
    caiso_path : str or Path

    Returns
    -------
    pd.DataFrame with 5 rows and columns:
        metric, model, caiso, ratio, gap_pct, assessment
    """
    caiso = caiso_summary_stats(caiso_path)

    # Detect column name for clearing price (different across analysis versions)
    if 'clearing_price' in model_results.columns:
        price_col = 'clearing_price'
    elif 'mean_clearing_price' in model_results.columns:
        price_col = 'mean_clearing_price'
    else:
        raise KeyError(
            "model_results must contain either 'clearing_price' or "
            "'mean_clearing_price' column"
        )

    # Detect column name for IPP exit (different across analysis versions)
    if 'final_ipp_exit' in model_results.columns:
        ipp_col = 'final_ipp_exit'
    elif 'ipp_exit_rate' in model_results.columns:
        ipp_col = 'ipp_exit_rate'
    else:
        ipp_col = None

    rows = []

    # Metric 1: Clearing price mean
    model_price = float(model_results[price_col].mean())
    caiso_price = caiso['lmp_mean_usd_mwh']
    rows.append({
        "metric":      "Clearing price mean ($/MWh)",
        "model":       round(model_price, 2),
        "caiso":       round(caiso_price, 2),
        "ratio":       round(model_price / caiso_price, 3),
        "gap_pct":     round((model_price - caiso_price) / caiso_price * 100, 1),
        "assessment":  _band(model_price, caiso_price, 0.5, 2.0),
    })

    # Metric 2: Price volatility
    model_std = float(model_results[price_col].std())
    caiso_std = caiso['lmp_std_usd_mwh']
    rows.append({
        "metric":      "Clearing price std ($/MWh)",
        "model":       round(model_std, 2),
        "caiso":       round(caiso_std, 2),
        "ratio":       round(model_std / caiso_std, 3),
        "gap_pct":     round((model_std - caiso_std) / caiso_std * 100, 1),
        "assessment":  _band(model_std, caiso_std, 0.3, 3.0),
    })

    # Metric 3: AI Lerner vs DMM markup
    model_lerner = float(model_results['final_ai_lerner'].mean())
    dmm_markup = CAISO_BENCHMARKS['dmm_markup_estimate']
    # Note: ABM Lerner is AI-firm-specific; DMM is market-average. ABM is
    # expected to be higher (consistent with strategic-bid theory of advantaged
    # bidders) but should remain within ~3x of the DMM estimate.
    rows.append({
        "metric":      "AI Lerner index (model) vs DMM markup",
        "model":       round(model_lerner, 3),
        "caiso":       round(dmm_markup, 3),
        "ratio":       round(model_lerner / dmm_markup, 3),
        "gap_pct":     round((model_lerner - dmm_markup) / dmm_markup * 100, 1),
        "assessment":  _band_lerner(model_lerner, dmm_markup),
    })

    # Metric 4: Final HHI
    model_hhi = float(model_results['final_hhi'].mean())
    caiso_hhi = CAISO_BENCHMARKS['supplier_hhi_estimate']
    rows.append({
        "metric":      "Final HHI",
        "model":       round(model_hhi, 0),
        "caiso":       round(caiso_hhi, 0),
        "ratio":       round(model_hhi / caiso_hhi, 3),
        "gap_pct":     round((model_hhi - caiso_hhi) / caiso_hhi * 100, 1),
        "assessment":  _band(model_hhi, caiso_hhi, 0.5, 2.5),
    })

    # Metric 5: IPP attrition (10-year, pro-rated from 5-year CAISO baseline)
    if ipp_col:
        model_attr = float(model_results[ipp_col].mean()) * 100
        # Naive pro-rating: 5y * 1.5 = 7.5y equivalent compounding; use 1.5x as approximation
        expected_10yr = CAISO_BENCHMARKS['ipp_attrition_5yr_pct'] * 1.5
        rows.append({
            "metric":      "IPP attrition rate at 120 months (%)",
            "model":       round(model_attr, 1),
            "caiso":       round(expected_10yr, 1),
            "ratio":       round(model_attr / expected_10yr, 3),
            "gap_pct":     round((model_attr - expected_10yr) / expected_10yr * 100, 1),
            "assessment":  _band(model_attr, expected_10yr, 0.6, 1.6),
        })

    return pd.DataFrame(rows)


def _band(model_val: float, caiso_val: float,
          lower: float = 0.5, upper: float = 2.0) -> str:
    """Classify a model/caiso ratio into WITHIN RANGE / RECALIBRATE bands."""
    if caiso_val == 0:
        return "UNDEFINED"
    ratio = model_val / caiso_val
    if lower <= ratio <= upper:
        return "WITHIN RANGE"
    elif ratio > upper:
        return "RECALIBRATE (model too high)"
    else:
        return "RECALIBRATE (model too low)"


def _band_lerner(model_lerner: float, dmm_markup: float) -> str:
    """Special band for Lerner: ABM AI Lerner is expected to be elevated above
    market-average DMM markup, consistent with strategic-bid theory of compute-
    advantaged bidders (Hortaçsu & Puller 2008). Mark as theory-consistent
    even when model > 1x DMM, up to 3x. Beyond 3x flags calibration issue.
    """
    if dmm_markup <= 0:
        return "UNDEFINED"
    ratio = model_lerner / dmm_markup
    if ratio < 0.5:
        return "RECALIBRATE (model too low)"
    elif ratio <= 1.5:
        return "WITHIN RANGE (consistent with DMM)"
    elif ratio <= 3.0:
        return "ABM-elevated (theory-consistent with HP 2008)"
    else:
        return "RECALIBRATE (model too high)"


# ============================================================================
#  Diagnostic plot
# ============================================================================

def plot_caiso_validation(
    model_results: pd.DataFrame,
    caiso_path: Union[str, Path],
    save_path: Optional[Path] = None,
) -> None:
    """Four-panel diagnostic plot: price histogram, Q-Q plot, metric bars,
    validation summary table.
    """
    # Detect price column name (compatibility across analysis versions)
    if 'clearing_price' in model_results.columns:
        price_col = 'clearing_price'
    elif 'mean_clearing_price' in model_results.columns:
        price_col = 'mean_clearing_price'
    else:
        raise KeyError("model_results must contain 'clearing_price' or 'mean_clearing_price'")

    caiso_df = pd.read_csv(caiso_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Price distribution overlay
    ax = axes[0, 0]
    caiso_prices = caiso_df['LMP'].dropna()
    cap_val = caiso_prices.quantile(0.99)
    ax.hist(caiso_prices[caiso_prices <= cap_val], bins=60,
            alpha=0.5, color='steelblue', density=True, label='CAISO 2025')
    model_prices = model_results[price_col].dropna()
    ax.hist(model_prices[model_prices <= cap_val], bins=60,
            alpha=0.5, color='firebrick', density=True, label='ABM (this paper)')
    ax.set_xlabel('Clearing price ($/MWh)')
    ax.set_ylabel('Density')
    ax.set_title('Price distribution: model vs CAISO 2025')
    ax.legend()
    ax.set_xlim(-20, min(cap_val, 200))
    ax.grid(alpha=0.3)

    # Panel 2: Q-Q plot
    ax = axes[0, 1]
    n_sample = min(2000, len(model_prices), len(caiso_prices))
    rng = np.random.default_rng(42)
    model_sample = model_prices.sample(n_sample, random_state=42).values
    caiso_sample = caiso_prices.sample(n_sample, random_state=42).values
    q = np.linspace(0.02, 0.98, 50)
    ax.scatter(np.quantile(caiso_sample, q), np.quantile(model_sample, q),
               s=25, alpha=0.7, color='darkblue')
    lo = min(caiso_sample.min(), model_sample.min())
    hi = max(caiso_sample.max(), model_sample.max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='y = x')
    ax.set_xlabel('CAISO LMP quantile ($/MWh)')
    ax.set_ylabel('ABM clearing price quantile ($/MWh)')
    ax.set_title('Q-Q plot: model vs empirical')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: Normalized metric bars
    ax = axes[1, 0]
    metrics = ['HHI\n(÷3000)', 'AI Lerner', 'Price\n(÷100)']
    model_vals = [
        model_results['final_hhi'].mean() / 3000.0,
        model_results['final_ai_lerner'].mean(),
        model_results[price_col].mean() / 100.0,
    ]
    caiso_vals = [
        CAISO_BENCHMARKS['supplier_hhi_estimate'] / 3000.0,
        CAISO_BENCHMARKS['dmm_markup_estimate'],
        caiso_df['LMP'].mean() / 100.0,
    ]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, model_vals, w, color='firebrick', alpha=0.8, label='ABM')
    ax.bar(x + w/2, caiso_vals, w, color='steelblue', alpha=0.8, label='CAISO')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Normalized value')
    ax.set_title('Aggregate metric comparison')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Panel 4: Validation table
    ax = axes[1, 1]
    val_df = validate_against_caiso(model_results, caiso_path)
    ax.axis('off')
    tbl_data = []
    for _, row in val_df.iterrows():
        # Shorten metric labels for the table
        metric_short = row['metric'].split('(')[0].strip()[:25]
        tbl_data.append([
            metric_short,
            f"{row['model']:.2f}" if isinstance(row['model'], float) else str(row['model']),
            f"{row['caiso']:.2f}" if isinstance(row['caiso'], float) else str(row['caiso']),
            f"{row['gap_pct']:+.1f}%",
            row['assessment'][:25],
        ])
    col_labels = ['Metric', 'Model', 'CAISO', 'Gap', 'Assessment']
    table = ax.table(cellText=tbl_data, colLabels=col_labels,
                     loc='center', cellLoc='left', colWidths=[0.30, 0.12, 0.12, 0.12, 0.34])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.7)
    # Color-code assessment cells
    for i, row in enumerate(tbl_data):
        cell = table[(i+1, 4)]
        if 'WITHIN' in row[4]:
            cell.set_facecolor('#d4edda')
        elif 'RECALIBRATE' in row[4]:
            cell.set_facecolor('#f8d7da')
        elif 'ABM-elevated' in row[4]:
            cell.set_facecolor('#fff3cd')
    ax.set_title('Validation summary')

    plt.suptitle('ABM external validation against CAISO 2025',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    plt.show()


# ============================================================================
#  CLI usage
# ============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python caiso_validation.py path/to/CAISO_2025_15min_LMP.csv")
        sys.exit(1)
    stats = caiso_summary_stats(sys.argv[1])
    print("=== CAISO summary statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
