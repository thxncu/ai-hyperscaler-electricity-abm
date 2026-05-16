"""
empirical_anchor.py — Real-world hyperscaler energy transaction database
and comparison with simulated M2 acquisitions.

Provides:
- TRANSACTIONS: pandas DataFrame of 8 major hyperscaler energy deals (2024-2025)
                covering acquisitions, PPAs, and master development agreements.
                Compiled from SEC filings, company press releases, FERC dockets,
                and verified trade press.
- empirical_summary(): aggregate stats for the empirical sample
- appendix_table_a1(): formatted DataFrame ready for manuscript Appendix
- compare_to_simulation(): side-by-side comparison with simulated M2 acquisitions
- HYPERSCALER_CASH: cash and short-term investments of major hyperscalers
                    at year-end 2023 (calibration anchor for theta_DA / M2 trigger)

Purpose:
1. Provides reproducible empirical anchor for M2 mechanism calibration
2. Strengthens §3.3 "compiled from public M&A databases and FERC filings"
   from a vague claim into a verifiable Appendix Table A1
3. Supports the regime-conditional finding by showing the empirical
   universe of hyperscaler procurement matches the H-regime parameter set

Author: Chankook Park (HUFS)
Version: anchor-0.1 (May 2026)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

# ============================================================================
#  HYPERSCALER ENERGY TRANSACTIONS, 2024-2025
# ============================================================================
#
# Inclusion criteria:
#   1. Hyperscaler buyer: Amazon (AWS), Microsoft, Google, Meta, or Oracle
#   2. Deal size ≥ 300 MW OR deal value ≥ $500M (filters to "structural" deals)
#   3. Announcement date Jan 2024 - Dec 2025
#   4. Public confirmation via SEC filing or official press release
#
# Capacity figures are nameplate MW. For acquisitions, capacity = data center
# campus power capacity. For PPAs, capacity = contracted offtake at full ramp.

TRANSACTIONS = pd.DataFrame([
    {
        "date": "2024-03",
        "buyer": "Amazon (AWS)",
        "seller": "Talen Energy",
        "asset": "Cumulus data center campus, Susquehanna PA",
        "deal_type": "Asset acquisition",
        "capacity_mw": 960,
        "duration_yrs": np.nan,  # acquisition, not time-bounded
        "tech": "Nuclear (host)",
        "iso": "PJM",
        "deal_value_musd": 650.0,
        "source": "Talen 8-K 2024-03-04; SEC 4-08-K",
    },
    {
        "date": "2024-05",
        "buyer": "Microsoft",
        "seller": "Brookfield Asset Management",
        "asset": "Global renewable framework (2026-2030 build)",
        "deal_type": "PPA framework",
        "capacity_mw": 10500,
        "duration_yrs": 5.0,  # 2026-2030 development period
        "tech": "Solar/Wind",
        "iso": "US + Europe (multiple)",
        "deal_value_musd": 10000.0,  # ~$10B FT estimate
        "source": "Brookfield-Microsoft joint release 2024-05-01; FT 2024-05-01",
    },
    {
        "date": "2024-09",
        "buyer": "Microsoft",
        "seller": "Constellation Energy",
        "asset": "Three Mile Island Unit 1 restart",
        "deal_type": "PPA (restart)",
        "capacity_mw": 835,
        "duration_yrs": 20.0,
        "tech": "Nuclear",
        "iso": "PJM",
        "deal_value_musd": 1600.0,  # Constellation restart investment; PPA value undisclosed
        "source": "Constellation 8-K 2024-09-20; Microsoft Cloud Blog 2024-09-20",
    },
    {
        "date": "2024-10",
        "buyer": "Google",
        "seller": "Kairos Power",
        "asset": "Advanced SMR fleet by 2035",
        "deal_type": "Master plant development + PPA",
        "capacity_mw": 500,
        "duration_yrs": 11.0,  # signing to 2035 commissioning
        "tech": "Nuclear (SMR)",
        "iso": "Multiple service territories",
        "deal_value_musd": np.nan,  # multi-billion, undisclosed
        "source": "Kairos-Google joint release 2024-10-14",
    },
    {
        "date": "2024-10",
        "buyer": "Amazon (AWS)",
        "seller": "Energy Northwest / X-Energy",
        "asset": "Cascade SMR project Washington",
        "deal_type": "Development + equity investment",
        "capacity_mw": 960,  # 320 MW initial → 960 MW expanded (12 reactors)
        "duration_yrs": np.nan,  # 2030 first online, no terminal date
        "tech": "Nuclear (SMR)",
        "iso": "WECC",
        "deal_value_musd": 500.0,  # X-Energy equity round led by AWS
        "source": "Amazon About blog 2024-10-16; Utility Dive 2024-10-16",
    },
    {
        "date": "2024-10",
        "buyer": "Amazon (AWS)",
        "seller": "Dominion Energy",
        "asset": "North Anna SMR development (Virginia)",
        "deal_type": "MOU (development)",
        "capacity_mw": 300,  # minimum stated
        "duration_yrs": np.nan,
        "tech": "Nuclear (SMR)",
        "iso": "PJM",
        "deal_value_musd": np.nan,
        "source": "Dominion release 2024-10-16; Virginia Mercury 2024-10-17",
    },
    {
        "date": "2025-06",
        "buyer": "Meta",
        "seller": "Constellation Energy",
        "asset": "Clinton Clean Energy Center, Illinois",
        "deal_type": "PPA",
        "capacity_mw": 1121,
        "duration_yrs": 20.0,
        "tech": "Nuclear",
        "iso": "MISO Zone 4",
        "deal_value_musd": np.nan,  # multi-billion, undisclosed
        "source": "Constellation press release 2025-06-03; Meta About 2025-06-03",
    },
    {
        "date": "2025-06",
        "buyer": "Amazon (AWS)",
        "seller": "Talen Energy",
        "asset": "Susquehanna nuclear PPA (expanded, front-of-meter)",
        "deal_type": "PPA",
        "capacity_mw": 1920,  # ramp to 1.92 GW by 2032
        "duration_yrs": 17.0,  # through 2042
        "tech": "Nuclear",
        "iso": "PJM",
        "deal_value_musd": 18000.0,  # Talen investor presentation: ~$18B contract value
        "source": "Talen investor presentation 2025-06-11; Utility Dive 2025-06-11",
    },
])


# ============================================================================
#  HYPERSCALER CASH POSITIONS (M2 trigger calibration anchor)
# ============================================================================
# Source: SEC 10-K filings, year-end 2023. Combined "cash and short-term
# investments" line. Numbers in USD millions.

HYPERSCALER_CASH = pd.DataFrame([
    {"firm": "Microsoft",   "cash_short_term_invest_musd": 111_262, "fy_end": "2023-06"},
    {"firm": "Alphabet",    "cash_short_term_invest_musd": 110_916, "fy_end": "2023-12"},
    {"firm": "Amazon",      "cash_short_term_invest_musd":  86_780, "fy_end": "2023-12"},
    {"firm": "Meta",        "cash_short_term_invest_musd":  65_402, "fy_end": "2023-12"},
    {"firm": "Oracle",      "cash_short_term_invest_musd":  10_454, "fy_end": "2024-05"},
])


# ============================================================================
#  SUMMARY STATISTICS
# ============================================================================


def empirical_summary() -> pd.DataFrame:
    """Aggregate statistics for the empirical transaction database."""
    df = TRANSACTIONS
    return pd.DataFrame([{
        "n_transactions":            len(df),
        "total_capacity_mw":         df["capacity_mw"].sum(),
        "median_capacity_mw":        df["capacity_mw"].median(),
        "mean_capacity_mw":          df["capacity_mw"].mean(),
        "p25_capacity_mw":           df["capacity_mw"].quantile(0.25),
        "p75_capacity_mw":           df["capacity_mw"].quantile(0.75),
        "median_duration_yrs":       df["duration_yrs"].dropna().median(),
        "total_disclosed_value_busd": df["deal_value_musd"].sum() / 1000,
        "n_buyers":                  df["buyer"].nunique(),
        "buyer_HHI_in_sample":       _buyer_concentration_hhi(df),
        "nuclear_share":             (df["tech"].str.contains("Nuclear")).mean(),
        "asset_acquisition_share":   (df["deal_type"] == "Asset acquisition").mean(),
    }]).T.rename(columns={0: "value"})


def _buyer_concentration_hhi(df: pd.DataFrame) -> float:
    """HHI of buyer capacity shares within the transaction sample."""
    shares = df.groupby("buyer")["capacity_mw"].sum() / df["capacity_mw"].sum()
    return float((shares ** 2 * 10000).sum())


def hyperscaler_cash_summary() -> pd.DataFrame:
    """Summary of hyperscaler cash reserves (M2 trigger calibration anchor)."""
    df = HYPERSCALER_CASH
    return pd.DataFrame([{
        "total_cash_busd":   df["cash_short_term_invest_musd"].sum() / 1000,
        "median_cash_busd":  df["cash_short_term_invest_musd"].median() / 1000,
        "max_cash_busd":     df["cash_short_term_invest_musd"].max() / 1000,
        "n_firms":           len(df),
    }]).T.rename(columns={0: "value"}).round(2)


# ============================================================================
#  APPENDIX TABLE A1 (manuscript-ready)
# ============================================================================


def appendix_table_a1() -> pd.DataFrame:
    """
    Format TRANSACTIONS for direct insertion into manuscript Appendix.
    Returns a clean DataFrame with manuscript-style column labels and
    "Undisclosed" / "n.a." placeholders for missing values.
    """
    df = TRANSACTIONS.copy()
    df["Value (USD M)"] = df["deal_value_musd"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "Undisclosed"
    )
    df["Duration"] = df["duration_yrs"].apply(
        lambda x: f"{int(x)} yrs" if pd.notna(x) else "n.a."
    )
    df["Capacity (MW)"] = df["capacity_mw"].apply(lambda x: f"{int(x):,}")
    out = df[[
        "date", "buyer", "seller", "asset", "deal_type",
        "Capacity (MW)", "Duration", "tech", "iso", "Value (USD M)",
    ]].copy()
    out.columns = [
        "Date", "Buyer", "Seller", "Asset", "Deal type",
        "Capacity (MW)", "Duration", "Technology", "ISO/Region", "Value",
    ]
    return out.reset_index(drop=True)


# ============================================================================
#  COMPARISON WITH SIMULATED M2 ACQUISITIONS
# ============================================================================


def compare_to_simulation(
    sim_acquisitions: pd.DataFrame,
    cap_factor: float = 0.50,
) -> pd.DataFrame:
    """
    Side-by-side comparison of empirical transactions vs simulated M2 acquisitions.

    Inputs
    ------
    sim_acquisitions : DataFrame with at least one of:
        - 'acquired_capacity_mwh_month'  (model logs capacity transferred per event)
        - 'capacity_mw'                  (already in MW)
        Optionally:
        - 'acquisition_cost_usd'         (cost paid)
        - 'npv_ratio'                    (real-options NPV ratio that triggered the deal)
    cap_factor : assumed capacity factor for MWh→MW conversion (default 0.50)
                 720 hours/month × cap_factor = MWh/month per MW nameplate

    Returns
    -------
    DataFrame with metrics as rows, "Empirical" and "Simulated" as columns,
    suitable for direct paste into §3.3 or Appendix.

    Note
    ----
    Aggregate match in distribution shape (median, p25, p75) is the
    calibration claim; absolute scale will differ because the simulated
    market represents a mid-sized ISO while empirical deals span the
    entire U.S. hyperscaler universe.
    """
    df = sim_acquisitions.copy()

    # Convert MWh/month to nameplate MW if needed
    if "capacity_mw" not in df.columns:
        if "acquired_capacity_mwh_month" in df.columns:
            df["capacity_mw"] = df["acquired_capacity_mwh_month"] / (720 * cap_factor)
        else:
            raise ValueError(
                "sim_acquisitions must contain 'capacity_mw' or 'acquired_capacity_mwh_month'"
            )

    emp = TRANSACTIONS["capacity_mw"]
    sim = df["capacity_mw"]

    rows = [
        ("N (transactions / acquisition events)", len(emp),               len(sim)),
        ("Median capacity (MW)",                  emp.median(),           sim.median()),
        ("Mean capacity (MW)",                    emp.mean(),             sim.mean()),
        ("P25 capacity (MW)",                     emp.quantile(0.25),     sim.quantile(0.25)),
        ("P75 capacity (MW)",                     emp.quantile(0.75),     sim.quantile(0.75)),
        ("Max capacity (MW)",                     emp.max(),              sim.max()),
        ("Coefficient of variation",              emp.std() / emp.mean(), sim.std() / sim.mean()),
    ]
    out = pd.DataFrame(rows, columns=["metric", "Empirical (2024-2025)", "Simulated (M2 on)"])

    # Round numerics
    for c in ("Empirical (2024-2025)", "Simulated (M2 on)"):
        out[c] = out[c].apply(lambda x: round(x, 2) if isinstance(x, float) else x)
    return out


# ============================================================================
#  CLI / DEMO
# ============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("EMPIRICAL ANCHOR DATABASE — HYPERSCALER ENERGY TRANSACTIONS 2024-2025")
    print("=" * 80)

    print("\nTransaction database (raw):")
    print(TRANSACTIONS.to_string(index=False))

    print("\n" + "=" * 80)
    print("Aggregate summary statistics")
    print("=" * 80)
    print(empirical_summary().to_string())

    print("\n" + "=" * 80)
    print("Hyperscaler cash reserves (M2 trigger anchor, YE 2023)")
    print("=" * 80)
    print(HYPERSCALER_CASH.to_string(index=False))
    print(hyperscaler_cash_summary().to_string())

    print("\n" + "=" * 80)
    print("APPENDIX TABLE A1 (manuscript-ready)")
    print("=" * 80)
    print(appendix_table_a1().to_string(index=False))
