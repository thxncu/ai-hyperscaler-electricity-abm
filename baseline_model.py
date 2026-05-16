"""
AI Compute Power and Electricity Market Restructuring: ABM Baseline (v6)

v6 changelog (Scale rescaling for empirical parity with mid-sized ISOs):
- CHANGE: baseline_demand_mwh: 100,000 → 18,000,000 MWh/month
  (CAISO ≈ 25 GW load × 720 h ≈ 18,000,000 MWh/month)
- CHANGE: AI capacity range: 5,000-15,000 → 900,000-2,700,000 MWh/month (×180)
- CHANGE: Utility capacity range: 3,000-10,000 → 540,000-1,800,000 (×180)
- CHANGE: IPP capacity range: 200-1,500 → 36,000-270,000 (×180)
- CHANGE: theta_DA: 2M → 360M $ (×180, activation threshold proportional to scale)
- CHANGE: PPA capacity scale × 180
- CHANGE: fixed_cost_ratio 0.04 → 0.025 (CAISO IPP attrition calibration)
- CHANGE: lambda_exit 18 → 36 months (3-yr consecutive losses, empirical alignment)
- All other parameters scale-invariant (kappa, phi, alpha, beta, gamma, m2_utilization_rate).

v4 changelog:
- ADD: Pay-as-bid clearing rule (motivated by M3-dead finding in uniform-price)
  - ISO.clear_market now respects clearing_rule parameter ('uniform' or 'pay_as_bid')
  - Each firm tracks last_price_received (firm-specific under pay-as-bid)
  - AI Lerner index computed from firm-specific received prices
- ADD: P3 stress test preset parameters (P3_STRESS_PARAMS) — aggressive M2 to push
  HHI past 1800 trigger for active P3 testing
- ADD: Self-test #10 (pay-as-bid mechanism check)
- ADD: Convenience constants (DEFAULT_PARAMS, P3_STRESS_PARAMS)

v3 changelog (M2 mechanism activated):
- ADD: Mechanism M2 (capital → asset acquisition) — v3 §2.5 eqs. (3a, 3b)
- ADD: AIFirm._consider_m2_acquisition() — real-options-based acquisition decision
- ADD: Eight M2-related parameters in ModelParameters

v2 changelog (post code-review fixes):
- BUGFIX: IPP exit logic now functions correctly
- BUGFIX: Regulator no longer mutates model.params permanently
- BUGFIX: M3 mechanism direction corrected
- Mesa 3 compliance

Literature anchors (v3 §2.5, §8.5, Appendix C):
- Williamson (1985) TCE
- Hortaçsu, Luco, Puller & Zhu (2024) JPE — VI premium 1.2-2.5x
- Dixit & Pindyck (1994) — Real options
- Khan (2017) Yale LJ — Big Tech strategic acquisitions
- Hortaçsu & Puller (2008) RAND — pay-as-bid vs uniform comparison
- Holmberg & Newbery (2010) Util. Pol. — clearing rule sensitivity

Author: Chankook Park (HUFS)
Version: baseline-0.4 (May 2026)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import mesa


# ============================================================================
#  CONFIGURATION / SCENARIO STRUCTURES
# ============================================================================


class ComputeRegime(Enum):
    """Scenario axis A: compute concentration."""
    DISPERSED = "L"   # top-10 share ≤ 30%
    OLIGOPOLY = "H"   # top-3 share ≥ 70%


class PPARegime(Enum):
    """Scenario axis B: clean energy preemption strength."""
    LOW = "L"   # Big Tech PPA ≤ 25%
    HIGH = "H"  # Big Tech PPA ≥ 60%


class RegulationRegime(Enum):
    """Scenario axis C: regulatory intervention timing."""
    EX_POST = "L"   # delay 24 months, HHI* = 3000
    EX_ANTE = "H"   # delay 0 months, HHI* = 1800


@dataclass(frozen=True)  # v2: immutable
class ScenarioConfig:
    """6-cell empirical-coherent scenario configuration (see v3 §1.3)."""
    code: str
    compute: ComputeRegime
    ppa: PPARegime
    regulation: RegulationRegime

    @property
    def hhi_trigger(self) -> float:
        return 1800.0 if self.regulation == RegulationRegime.EX_ANTE else 3000.0

    @property
    def reg_delay_months(self) -> int:
        return 0 if self.regulation == RegulationRegime.EX_ANTE else 24

    @property
    def ai_total_compute_share(self) -> float:
        return 0.70 if self.compute == ComputeRegime.OLIGOPOLY else 0.30

    @property
    def bigtech_ppa_share(self) -> float:
        return 0.60 if self.ppa == PPARegime.HIGH else 0.25


# The 6 empirically coherent scenarios (LH and HL excluded due to A↔B correlation)
SCENARIOS_6CELL: list[ScenarioConfig] = [
    ScenarioConfig("LLL", ComputeRegime.DISPERSED, PPARegime.LOW,  RegulationRegime.EX_POST),
    ScenarioConfig("LLH", ComputeRegime.DISPERSED, PPARegime.LOW,  RegulationRegime.EX_ANTE),
    # LHL excluded: empirically incoherent (low compute concentration + high PPA preemption)
    ScenarioConfig("LHH", ComputeRegime.DISPERSED, PPARegime.HIGH, RegulationRegime.EX_ANTE),
    # HLL excluded: empirically incoherent (high compute + low PPA)
    ScenarioConfig("HLH", ComputeRegime.OLIGOPOLY, PPARegime.LOW,  RegulationRegime.EX_ANTE),
    ScenarioConfig("HHL", ComputeRegime.OLIGOPOLY, PPARegime.HIGH, RegulationRegime.EX_POST),
    ScenarioConfig("HHH", ComputeRegime.OLIGOPOLY, PPARegime.HIGH, RegulationRegime.EX_ANTE),
]


@dataclass
class ModelParameters:
    """Agent-level parameters with empirical anchors. See v3 Appendix B."""
    # --- Mechanism M1: compute → forecasting (Equation 1) ---
    alpha: float = 0.15           # Baseline forecast error level [Lago et al. 2021]
    beta: float = 0.4             # Compute-skill conversion elasticity (conservative)
    forecast_func: str = "power"  # f ∈ {"power", "log", "sigmoid"}
    sigma_max_multiplier: float = 5.0  # v2: parametric (was magic number)

    # --- Bidding rule (Equation 2) ---
    kappa: float = 0.35           # v2: lowered to fit Hortaçsu-Puller 5-25% empirical range
    gamma: float = 1.0            # Confidence weighting curvature

    # --- Mechanism M3: private load info (v2: meaningful direction) ---
    # Multiplicative reduction of σ for AI firms. delta=0 disables.
    # delta=0.3 → AI firms' σ is 30% smaller than the compute-share-derived baseline.
    delta_sigma_reduction: float = 0.0

    # --- Vertical integration / asset acquisition (Equations 3a, 3b) — v3 ACTIVE ---
    phi: float = 1.5              # NPV ratio threshold [Hortaçsu et al. 2024]
    # M2 trigger: cash threshold ($) before acquisition considered
    # v6: 2M × 180 = 360M, keeping activation threshold proportional to market scale.
    theta_DA: float = 360_000_000.0
    # Acquisition cost = capacity × $50/MWh × 12 months × acquisition_cost_multiplier (years)
    # 1.0 = 1 year of reference revenue. Calibrated so low-MC IPPs are economically
    # attractive targets but high-MC IPPs are not (matches Big Tech PPA preference for renewables).
    acquisition_cost_multiplier: float = 1.0
    # NPV planning horizon for acquisition decision
    m2_planning_horizon_months: int = 60
    # Annual discount rate for NPV calculation
    m2_discount_rate_annual: float = 0.08
    # Assumed utilization rate of acquired capacity under AI firm's operation.
    # 0.7 reflects Big Tech's portfolio optimization advantage post-acquisition.
    m2_utilization_rate: float = 0.7
    # IPP candidate filter: minimum consecutive loss months required.
    # 0 = no distress filter (any alive IPP is a candidate, NPV ratio gates the deal).
    # Empirically, requiring distress (>=1) starves the candidate pool because
    # IPPs either stay profitable or exit within lambda_exit months.
    m2_distress_threshold: int = 0
    # Per-firm cooldown between successive acquisitions (months)
    m2_cooldown_months: int = 12

    # --- Cost structure (v2: NEW — enables meaningful exit dynamics) ---
    # Monthly fixed cost as a fraction of nameplate capacity revenue at $50/MWh × 720h.
    # Without fixed cost, firms can never lose money in a uniform-price auction with
    # bids ≥ MC, so the exit rule (λ_exit) never fires.
    # v6: 0.04 → 0.025 to align model IPP attrition with CAISO empirical baseline (~17.5%/5y).
    # v6.1: 0.025 → 0.018 — 120-month attrition 54% → ~28% (CAISO 10-yr ≈ 26%).
    fixed_cost_ratio: float = 0.018

    # --- Exit rule ---
    # v6: 18 → 36 months (3 years consecutive losses) for empirical alignment.
    lambda_exit: int = 36

    # --- Demand and renewable variability ---
    rho_demand: float = 0.04      # Annual demand growth
    eta_re: float = 0.25          # Renewable output CV
    demand_noise_sd: float = 0.05  # Stochastic demand shock SD as fraction of mean

    # --- Market design (v4 NEW) ---
    # Clearing rule: 'uniform' (pay-as-clear, default; CAISO/PJM/ERCOT standard)
    # or 'pay_as_bid' (each firm settles at its own bid; UK NETA-era, some discriminatory designs)
    # M3 mechanism's effect is conditional on clearing rule: dead under uniform, active under pay_as_bid.
    clearing_rule: str = "uniform"


# ============================================================================
#  M1 MECHANISM: COMPUTE → FORECASTING ACCURACY
# ============================================================================


def forecast_error(
    compute_share: float,
    mean_share: float,
    alpha: float,
    beta: float,
    functional_form: str = "power",
) -> float:
    """
    Equation (1): σ_i = α · f(c_i / c̄; β)

    The functional form f is treated as a robustness dimension (v3 §2.2).
    Lower σ means a more accurate forecast.

    Anchored on:
    - Wolak (2003): information precision determines bid placement
    - Lago et al. (2021): RMSE declines as a power function of model capacity
    - Brogaard et al. (2014): price discovery concentrates with compute capacity

    Parameters
    ----------
    compute_share : agent's share of total compute
    mean_share    : mean compute share across agents (c̄)
    alpha         : baseline error level
    beta          : conversion elasticity
    functional_form : "power", "log", or "sigmoid"

    Returns
    -------
    forecast error standard deviation σ_i ∈ (0, ∞)
    """
    if mean_share <= 0:
        return alpha
    ratio = max(compute_share / mean_share, 1e-6)

    if functional_form == "power":
        return alpha * (ratio ** (-beta))
    elif functional_form == "log":
        return max(alpha * (1.0 - beta * math.log(ratio)), 0.01)
    elif functional_form == "sigmoid":
        return alpha / (1.0 + math.exp(beta * (ratio - 1.0)))
    else:
        raise ValueError(f"Unknown functional form: {functional_form}")


# ============================================================================
#  AGENTS
# ============================================================================


class GenericFirm(mesa.Agent):
    """
    Base class for all market-participating firms.
    Subclasses: AIFirm, Utility, IPP.
    """

    def __init__(
        self,
        model: "ElectricityMarketABM",
        marginal_cost: float,
        capacity_mwh: float,
        compute_share: float = 0.0,
        ppa_capacity: float = 0.0,
    ) -> None:
        super().__init__(model)
        self.marginal_cost = marginal_cost   # $/MWh
        self.capacity_mwh = capacity_mwh     # Hourly capacity in MWh
        self.compute_share = compute_share   # Mechanism M1: c_i
        self.ppa_capacity = ppa_capacity     # Mechanism M2 placeholder
        # State
        self.alive = True
        self.cash = 0.0
        self.consecutive_loss_months = 0
        # Records
        self.last_forecast_error: float = 0.0
        self.last_bid: float = marginal_cost
        self.last_quantity_cleared: float = 0.0
        self.last_profit: float = 0.0
        self.last_price_received: float = 0.0  # v4: firm-specific under pay_as_bid
        self.exit_month: Optional[int] = None

    # ---- Mechanism M3 hook ----
    def _effective_sigma(self, base_sigma: float) -> float:
        """
        Mechanism M3: AI firms' own-load information yields a *more accurate*
        short-term forecast (lower σ). Default: no adjustment.
        Overridden by AIFirm.
        """
        return base_sigma

    # ---- Bidding rule (Equation 2) ----
    def submit_bid(self, expected_clearing_price: float) -> float:
        """
        p_i,t = MC_i + κ · w_conf(σ_i) · (p̂_t − MC_i)

        Anchored on Klemperer & Meyer (1989) SFE; markup magnitudes from
        Hortaçsu & Puller (2008): 5-25% above MC in the Texas spot market.
        """
        p = self.model.params
        beta_eff = self.model.beta_effective   # v2: regulator modifies via beta_effective
        # Forecast error (M1) + Mechanism M3 adjustment
        sigma_base = forecast_error(
            self.compute_share, self.model.mean_compute_share,
            p.alpha, beta_eff, p.forecast_func,
        )
        sigma = self._effective_sigma(sigma_base)
        self.last_forecast_error = sigma

        # Predicted price with forecast noise; v2: clamp at zero
        noise = self.model.random.gauss(0.0, sigma * expected_clearing_price)
        p_hat = max(expected_clearing_price + noise, 0.0)

        # Confidence weight (higher confidence when σ is small)
        sigma_max = max(p.alpha * p.sigma_max_multiplier, 1e-3)
        confidence = max(min(1.0 - sigma / sigma_max, 1.0), 0.0) ** p.gamma

        # Markup
        markup = p.kappa * confidence * max(p_hat - self.marginal_cost, 0.0)
        bid = self.marginal_cost + markup
        self.last_bid = bid
        return bid

    # ---- Step ----
    def step(self) -> None:
        if not self.alive:
            return
        expected_price = self.model.last_clearing_price
        bid = self.submit_bid(expected_price)
        self.model.iso.collect_bid(self, bid)

    # ---- Settlement (called by ISO after clearing) ----
    def settle(self, cleared_quantity: float, clearing_price: float) -> None:
        # v2: include fixed cost so that profit can actually be negative.
        revenue = cleared_quantity * clearing_price
        variable_cost = cleared_quantity * self.marginal_cost
        # Fixed cost: a fraction of capacity-month reference revenue at $50/MWh.
        # Treats capacity_mwh as monthly capacity (consistent with monthly clearing).
        # fixed_cost_ratio=0.04 → 4% × $50 = $2 per MWh of capacity per month.
        fixed_cost = self.capacity_mwh * 50.0 * self.model.params.fixed_cost_ratio
        profit = revenue - variable_cost - fixed_cost

        self.last_quantity_cleared = cleared_quantity
        self.last_price_received = clearing_price  # v4: track firm-specific price
        self.last_profit = profit
        self.cash += profit

        if profit < 0:
            self.consecutive_loss_months += 1
        else:
            self.consecutive_loss_months = max(self.consecutive_loss_months - 1, 0)

        if self.consecutive_loss_months >= self.model.params.lambda_exit:
            self.alive = False
            self.exit_month = self.model.month_index


class AIFirm(GenericFirm):
    """
    Hyperscaler / AI firm agent.

    Distinguishing features:
    - Higher compute_share than other firms (M1 advantage)
    - Mechanism M3: own-load information yields lower σ (v2: directionally correct)
    - Mechanism M2: capital → asset acquisition (v3: ACTIVE)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # M2 state: when did this firm last acquire?
        self._last_acquisition_month: int = -10**9  # effectively -infinity

    def _effective_sigma(self, base_sigma: float) -> float:
        """
        Mechanism M3 (v2 corrected): own-load information reduces σ.
        delta_sigma_reduction = 0.3 → AI firms' σ is 30% lower than M1 alone.
        """
        reduction = self.model.params.delta_sigma_reduction
        return base_sigma * (1.0 - reduction)

    def _consider_m2_acquisition(self) -> bool:
        """
        Mechanism M2 (v3 §2.5 — Equations 3a, 3b): Capital → asset acquisition.

        AI firms with accumulated cash acquire stressed IPP assets,
        growing their generating capacity and producing structural concentration.
        This is the missing piece that should reverse the negative P1 proxy
        observed in the baseline (information rent without market growth).

        Decision rule:
          1. Trigger: cash >= theta_DA AND cooldown elapsed
          2. Find distressed IPP candidates (consecutive_loss_months >= threshold)
          3. For each, compute PV(expected profit from target's capacity) / acquisition_cost
          4. Target = candidate giving highest NPV ratio
          5. Execute if NPV ratio >= phi

        Anchored on:
        - Williamson (1985) TCE — asset specificity drives integration
        - Hortaçsu, Luco, Puller & Zhu (2024) JPE — VI premium 1.2-2.5x
        - Dixit & Pindyck (1994) — Real options under uncertainty
        - Khan (2017) Yale LJ — Big Tech strategic acquisition patterns

        Returns
        -------
        True if an acquisition was executed, False otherwise.
        """
        p = self.model.params

        # Trigger 1: cash threshold
        if self.cash < p.theta_DA:
            return False

        # Trigger 2: cooldown (avoid bunching all decisions in one tick)
        if self.model.month_index - self._last_acquisition_month < p.m2_cooldown_months:
            return False

        # Find candidates: alive IPPs experiencing recent losses
        candidates = [
            a for a in self.model.agents
            if isinstance(a, IPP) and a.alive
            and a.capacity_mwh > 0
            and a.consecutive_loss_months >= p.m2_distress_threshold
        ]
        if not candidates:
            return False

        # Compute NPV ratio for each candidate; pick best
        expected_price = self.model.last_clearing_price
        H = p.m2_planning_horizon_months
        monthly_discount = p.m2_discount_rate_annual / 12.0
        if monthly_discount > 0:
            pv_factor = (1.0 - (1.0 + monthly_discount) ** (-H)) / monthly_discount
        else:
            pv_factor = float(H)

        best_target = None
        best_ratio = 0.0
        best_acq_cost = 0.0
        for target in candidates:
            margin = max(expected_price - target.marginal_cost, 0.0)
            monthly_profit = target.capacity_mwh * margin * p.m2_utilization_rate
            pv_profit = monthly_profit * pv_factor
            # Acquisition cost: target capacity × reference revenue × multiplier
            ref_revenue = target.capacity_mwh * 50.0 * 12.0
            acq_cost = ref_revenue * p.acquisition_cost_multiplier
            if acq_cost <= 0:
                continue
            ratio = pv_profit / acq_cost
            if ratio > best_ratio:
                best_ratio = ratio
                best_target = target
                best_acq_cost = acq_cost

        # Real-options threshold (phi)
        if best_target is None or best_ratio < p.phi:
            return False

        # Final cash check
        if best_acq_cost > self.cash:
            return False

        # Execute acquisition: transfer capacity and decommission target
        transferred = best_target.capacity_mwh
        self.capacity_mwh += transferred
        self.cash -= best_acq_cost

        best_target.capacity_mwh = 0.0
        best_target.alive = False
        best_target.exit_month = self.model.month_index

        # Log
        self._last_acquisition_month = self.model.month_index
        self.model.acquisitions_history.append({
            "month": self.model.month_index,
            "acquirer_compute_share": self.compute_share,
            "acquirer_capacity_after": self.capacity_mwh,
            "target_mc": best_target.marginal_cost,
            "capacity_transferred": transferred,
            "cost": best_acq_cost,
            "npv_ratio": best_ratio,
        })
        return True

    def step(self) -> None:
        if not self.alive:
            return
        super().step()  # Submit bid for current month
        # M2 mechanism: capital → asset acquisition (v3)
        self._consider_m2_acquisition()


class Utility(GenericFirm):
    """Traditional integrated utility (large baseload + retail)."""
    pass


class IPP(GenericFirm):
    """Small independent power producer (renewable/distributed)."""
    pass


class Regulator(mesa.Agent):
    """
    Adaptive regulator (Equation 4).

    Anchored on Stigler (1971), Peltzman (1976), Becker (1983).
    The welfare-maximizing baseline; partial-capture variant is a TODO hook.
    """

    def __init__(self, model: "ElectricityMarketABM"):
        super().__init__(model)
        self.has_intervened = False
        self.intervention_time: Optional[int] = None

    def step(self) -> None:
        m = self.model
        hhi = m.compute_hhi()
        m.last_hhi = hhi
        threshold = m.scen.hhi_trigger
        t_min = m.scen.reg_delay_months
        if (
            not self.has_intervened
            and hhi >= threshold
            and m.month_index >= t_min
        ):
            self._intervene()

    def _intervene(self) -> None:
        """
        Baseline intervention: transparency obligation reduces compute-share-derived
        forecasting advantage.

        v2: stores effect in model.beta_effective rather than mutating params,
        which avoids cross-run contamination when ModelParameters is shared.

        TODO[capture]: a partial-capture variant scales this less aggressively
        based on a `lobby_intensity` parameter.
        """
        self.has_intervened = True
        self.intervention_time = self.model.month_index
        # v2 fix: do not mutate model.params; use model.beta_effective.
        self.model.beta_effective *= 0.5


class ISO(mesa.Agent):
    """
    Independent System Operator: collects bids, clears uniform-price.
    """

    def __init__(self, model: "ElectricityMarketABM"):
        super().__init__(model)
        self.bid_book: list[tuple[GenericFirm, float]] = []

    def collect_bid(self, firm: GenericFirm, bid: float) -> None:
        self.bid_book.append((firm, bid))

    def clear_market(self, total_demand_mwh: float) -> tuple[float, dict[GenericFirm, float]]:
        """
        Market clearing supporting two rules (v4):

        - 'uniform' (default): pay-as-clear. Marginal bid sets the price; all cleared
          firms receive this uniform price. CAISO, PJM, ERCOT, MISO standard.

        - 'pay_as_bid': discriminatory pricing. Each cleared firm receives its OWN bid.
          UK NETA era, some Latin American markets. Bid accuracy now affects revenue,
          so Mechanism M3 (information advantage → bid precision) gains structural effect.

        References:
        - Klemperer & Meyer (1989) Econometrica — Supply Function Equilibrium
        - Holmberg & Newbery (2010) Util. Pol. — clearing rule sensitivity
        - Hortaçsu & Puller (2008) RAND — empirical markup under uniform
        """
        if not self.bid_book:
            return 0.0, {}

        rule = self.model.params.clearing_rule
        bids_sorted = sorted(self.bid_book, key=lambda x: x[1])
        cleared = {firm: 0.0 for firm, _ in self.bid_book}
        remaining = total_demand_mwh
        marginal_bid = bids_sorted[0][1]  # fallback

        # Determine cleared quantities (same for both rules)
        for firm, bid in bids_sorted:
            if remaining <= 1e-6:
                break
            take = min(firm.capacity_mwh, remaining)
            cleared[firm] = take
            remaining -= take
            marginal_bid = bid  # Last cleared bid is the marginal one

        # Settle based on clearing rule
        if rule == "uniform":
            # All cleared firms receive marginal_bid (uniform price)
            for firm, _ in self.bid_book:
                firm.settle(cleared.get(firm, 0.0), marginal_bid)
            clearing_price_metric = marginal_bid
        elif rule == "pay_as_bid":
            # Each firm receives its OWN bid
            for firm, bid in self.bid_book:
                firm.settle(cleared.get(firm, 0.0), bid)
            # Capacity-weighted average of cleared bids for the metric
            total_q = sum(cleared.values())
            if total_q > 0:
                clearing_price_metric = sum(
                    b * cleared.get(f, 0.0) for f, b in self.bid_book
                ) / total_q
            else:
                clearing_price_metric = marginal_bid
        else:
            raise ValueError(
                f"Unknown clearing_rule: {rule!r}. Use 'uniform' or 'pay_as_bid'."
            )

        self.bid_book.clear()
        return clearing_price_metric, cleared


# ============================================================================
#  MAIN MODEL
# ============================================================================


class ElectricityMarketABM(mesa.Model):
    """
    Baseline ABM for AI computing power and electricity market restructuring.

    Single-zone, uniform-price, monthly resolution baseline (extensible).
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        params: Optional[ModelParameters] = None,
        n_ai_firms: int = 5,
        n_utilities: int = 10,
        n_ipps: int = 40,
        horizon_months: int = 120,        # 10 years
        # v6: rescaled 180× from prior 100,000 to match a mid-sized U.S. ISO
        # (CAISO ≈ 25 GW load × 720 h ≈ 18,000,000 MWh/month). Required for
        # empirical scale parity with the 2024-2025 transaction record.
        baseline_demand_mwh: float = 18_000_000.0,
        seed: Optional[int] = None,
        # v5+: optional override of within-AI compute shares (bypasses Dirichlet draw).
        # Used by run_compute_concentration_sweep, run_compute_share_sweep, and
        # counterfactual_breakup. Pass np.ndarray of length n_ai_firms.
        compute_shares_override: Optional[np.ndarray] = None,
    ):
        # v2: use rng (Mesa 3 preferred); fall back to seed for backward compat.
        # Suppress the (intentionally muted) FutureWarning re: seed kwarg.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            super().__init__(seed=seed)

        # v2: single RNG (model.random and np_rng both derived from same seed)
        self.np_rng = np.random.default_rng(seed)

        self.scen = scenario
        # v2: deep-ish copy so external param objects are never mutated
        self.params = ModelParameters(**(params.__dict__ if params else {}))
        self.horizon_months = horizon_months
        self.baseline_demand_mwh = baseline_demand_mwh

        # State
        self.month_index = 0
        self.last_clearing_price = 50.0
        self.last_hhi = 0.0
        # v2: separate runtime-mutable beta from immutable params
        self.beta_effective: float = self.params.beta
        # v3: M2 mechanism — track all acquisition events
        self.acquisitions_history: list[dict] = []
        # v5+: optional within-AI compute shares override
        self._compute_shares_override = compute_shares_override

        # Create agents
        self._create_firms(n_ai_firms, n_utilities, n_ipps)
        self.iso = ISO(self)
        self.regulator = Regulator(self)

        # Time series storage
        self.history: list[dict] = []

    # ---- Firm creation ----
    def _create_firms(self, n_ai: int, n_util: int, n_ipp: int) -> None:
        """
        Initialize firms with compute shares according to scenario A axis.

        Compute concentration:
          - Dispersed: total AI compute share ~30%, evenly distributed
          - Oligopoly: total AI compute share ~70%, Dirichlet(α=2 for top-3)

        v5+: If `compute_shares_override` was passed at init, use it directly
        instead of the Dirichlet draw. Used by compute-concentration / share
        sweeps and counterfactual break-up to set deterministic share patterns.
        """
        if self._compute_shares_override is not None:
            # v5+ override path: caller provides exact within-AI compute distribution
            shares = np.asarray(self._compute_shares_override, dtype=float)
            assert len(shares) == n_ai, (
                f"compute_shares_override length {len(shares)} != n_ai_firms {n_ai}"
            )
            ai_compute = shares
            ai_total_share = float(shares.sum())
        else:
            ai_total_share = self.scen.ai_total_compute_share

            if n_ai >= 3:
                # Dirichlet with higher α for top-3 → concentrated within AI group
                ai_compute = self.np_rng.dirichlet(
                    alpha=[2.0, 2.0, 2.0] + [1.0] * (n_ai - 3),
                ) * ai_total_share
            else:
                ai_compute = np.ones(n_ai) / n_ai * ai_total_share

        # Remaining compute spread thinly among non-AI firms (representing internal IT)
        remaining = max(1.0 - ai_total_share, 0.0)
        rest_per_firm = remaining / max(n_util + n_ipp, 1)

        # AI firms (hyperscalers)
        for i in range(n_ai):
            AIFirm(
                model=self,
                marginal_cost=self.random.uniform(30, 45),
                # v6: 5,000-15,000 × 180 → realistic hyperscaler-scale capacity
                capacity_mwh=self.random.uniform(900_000, 2_700_000),
                compute_share=float(ai_compute[i]),
                # v6: PPA capacity also scaled 180×
                ppa_capacity=self.scen.bigtech_ppa_share * 1_800_000.0,
            )

        # Large utilities
        for _ in range(n_util):
            Utility(
                model=self,
                marginal_cost=self.random.uniform(40, 65),
                # v6: 3,000-10,000 × 180
                capacity_mwh=self.random.uniform(540_000, 1_800_000),
                compute_share=rest_per_firm,
            )

        # Small IPPs (renewable-heavy; wider MC dispersion)
        for _ in range(n_ipp):
            IPP(
                model=self,
                marginal_cost=self.random.uniform(20, 80),
                # v6: 200-1,500 × 180
                capacity_mwh=self.random.uniform(36_000, 270_000),
                compute_share=rest_per_firm,
            )

    # ---- Helpers ----
    @property
    def firms(self) -> list[GenericFirm]:
        return [a for a in self.agents if isinstance(a, GenericFirm) and a.alive]

    @property
    def ai_firms(self) -> list[AIFirm]:
        return [a for a in self.agents if isinstance(a, AIFirm) and a.alive]

    @property
    def mean_compute_share(self) -> float:
        firms = self.firms
        if not firms:
            return 0.0
        return float(np.mean([f.compute_share for f in firms]))

    # ---- Step ----
    def step(self) -> None:
        """One simulation step = one month."""
        self.month_index += 1

        # Stochastic demand
        growth_factor = (1.0 + self.params.rho_demand) ** (self.month_index / 12.0)
        demand_base = self.baseline_demand_mwh * growth_factor
        shock = self.np_rng.normal(0.0, self.params.demand_noise_sd) * demand_base
        total_demand = max(demand_base + shock, 0.0)

        # v2: shuffle agent step order (Mesa 3 idiom). For uniform-price clearing
        # the order doesn't affect the outcome, but it's good practice for future
        # extensions where firms react to each other.
        firms_to_step = list(self.firms)
        self.random.shuffle(firms_to_step)
        for firm in firms_to_step:
            firm.step()

        clearing_price, _ = self.iso.clear_market(total_demand)
        self.last_clearing_price = clearing_price

        self.regulator.step()
        self._record_metrics(clearing_price, total_demand)

    # ---- Metrics ----
    def compute_hhi(self) -> float:
        firms = self.firms
        total_q = sum(f.last_quantity_cleared for f in firms)
        if total_q <= 0:
            return 0.0
        shares_sq = sum((f.last_quantity_cleared / total_q) ** 2 for f in firms)
        return shares_sq * 10000.0

    def compute_ai_lerner_index(self) -> float:
        """
        v4: Uses firm-specific last_price_received so the metric is correct under
        both uniform-price (all firms same price) and pay-as-bid (firm-specific prices).
        Lerner_i = (price_i - MC_i) / price_i; aggregated as capacity-weighted average.
        """
        ai = [f for f in self.ai_firms if f.last_quantity_cleared > 0]
        if not ai:
            return 0.0
        total_q = sum(f.last_quantity_cleared for f in ai)
        return sum(
            max((f.last_price_received - f.marginal_cost) / max(f.last_price_received, 1e-6), 0.0)
            * f.last_quantity_cleared
            for f in ai
        ) / total_q

    def compute_ai_market_share(self) -> float:
        firms = self.firms
        total_q = sum(f.last_quantity_cleared for f in firms)
        if total_q <= 0:
            return 0.0
        ai_q = sum(f.last_quantity_cleared for f in self.ai_firms)
        return ai_q / total_q

    def compute_exit_rate(self) -> float:
        all_ipps = [a for a in self.agents if isinstance(a, IPP)]
        if not all_ipps:
            return 0.0
        return sum(1 for a in all_ipps if not a.alive) / len(all_ipps)

    def compute_ai_capacity_share(self) -> float:
        """v3: Total AI capacity / Total active capacity. Tracks M2 effect."""
        firms = self.firms
        total_cap = sum(f.capacity_mwh for f in firms)
        if total_cap <= 0:
            return 0.0
        ai_cap = sum(f.capacity_mwh for f in self.ai_firms)
        return ai_cap / total_cap

    def _record_metrics(self, clearing_price: float, demand: float) -> None:
        self.history.append({
            "month": self.month_index,
            "scenario": self.scen.code,
            "clearing_price": clearing_price,
            "demand_mwh": demand,
            "hhi": self.last_hhi,
            "ai_lerner": self.compute_ai_lerner_index(),
            "ai_market_share": self.compute_ai_market_share(),
            "ai_capacity_share": self.compute_ai_capacity_share(),   # v3
            "cumulative_acquisitions": len(self.acquisitions_history),  # v3
            "ipp_exit_rate": self.compute_exit_rate(),
            "regulator_intervened": self.regulator.has_intervened,
            "intervention_time": self.regulator.intervention_time,
            "beta_effective": self.beta_effective,
        })

    def run(self) -> pd.DataFrame:
        for _ in range(self.horizon_months):
            self.step()
        return pd.DataFrame(self.history)


# ============================================================================
#  PRESETS (v4)
# ============================================================================


def default_params() -> ModelParameters:
    """Default parameter set (M1+M2 active, M3 inactive, uniform clearing)."""
    return ModelParameters()


def p3_stress_params() -> ModelParameters:
    """
    P3 stress test preset: aggressive M2 to push HHI past the 1800 trigger.

    Use this preset when testing the P3 (regulatory irreversibility) hypothesis.
    Default parameters produce HHI plateau ~900-1000, below the 1800 trigger, so
    regulator never fires and P3 cannot be tested. This preset:
    - lowers acquisition cost (multiplier 1.0 → 0.7)
    - raises post-acquisition utilization (0.7 → 0.85)
    - relaxes NPV threshold (phi 1.5 → 1.2)
    - reduces cooldown to 6 months (faster acquisition cycle)
    """
    return ModelParameters(
        acquisition_cost_multiplier=0.7,
        m2_utilization_rate=0.85,
        phi=1.2,
        m2_cooldown_months=6,
    )


def pay_as_bid_params() -> ModelParameters:
    """Default + pay-as-bid clearing. Use to test M3 mechanism's structural role."""
    return ModelParameters(clearing_rule="pay_as_bid")


# ============================================================================
#  CONVENIENCE: RUN ALL 6 SCENARIOS
# ============================================================================


def run_all_scenarios(
    n_mc: int = 10,
    horizon_months: int = 120,
    seed_base: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """
    Run all 6 empirically coherent scenarios with `n_mc` Monte Carlo replications each.

    Returns long-format DataFrame with columns: scenario, mc_index, month, ...metrics.
    """
    rows = []
    for scen in SCENARIOS_6CELL:
        for mc in range(n_mc):
            seed = seed_base + hash((scen.code, mc)) % 10_000
            model = ElectricityMarketABM(
                scenario=scen,
                params=ModelParameters(),  # Fresh params each run
                horizon_months=horizon_months,
                seed=seed,
                **kwargs,
            )
            ts = model.run()
            ts["mc_index"] = mc
            rows.append(ts)
    return pd.concat(rows, ignore_index=True)


# ============================================================================
#  BUILT-IN UNIT TESTS (v2 NEW)
# ============================================================================


def _self_test() -> None:
    """Lightweight sanity checks. Run via:  python baseline_model.py --test """
    import sys

    print("=" * 60)
    print("BASELINE MODEL SELF-TESTS")
    print("=" * 60)

    # Test 1: forecast_error monotonicity (power form)
    print("\n[1] forecast_error monotonicity (power form)...")
    e_low = forecast_error(0.05, 0.10, alpha=0.15, beta=0.4, functional_form="power")
    e_high = forecast_error(0.20, 0.10, alpha=0.15, beta=0.4, functional_form="power")
    assert e_high < e_low, f"Higher compute should give lower σ: {e_low=}, {e_high=}"
    print(f"    OK: σ(low compute)={e_low:.3f} > σ(high compute)={e_high:.3f}")

    # Test 2: three functional forms all work
    print("\n[2] All three functional forms work...")
    for form in ["power", "log", "sigmoid"]:
        e = forecast_error(0.20, 0.10, alpha=0.15, beta=0.4, functional_form=form)
        assert e > 0 and math.isfinite(e), f"{form} produced bad σ: {e}"
        print(f"    OK: {form}: σ={e:.4f}")

    # Test 3: model runs end-to-end without errors
    print("\n[3] Model runs end-to-end (12 months)...")
    model = ElectricityMarketABM(scenario=SCENARIOS_6CELL[0], horizon_months=12, seed=42)
    df = model.run()
    assert len(df) == 12, f"Expected 12 rows, got {len(df)}"
    print(f"    OK: 12 months simulated, final HHI={df['hhi'].iloc[-1]:.0f}")

    # Test 4: regulator does NOT mutate params (v2 fix)
    print("\n[4] Regulator does not mutate ModelParameters (v2 bugfix)...")
    shared_params = ModelParameters()
    original_beta = shared_params.beta
    # Force a high-HHI scenario by manually triggering intervention
    m = ElectricityMarketABM(
        scenario=SCENARIOS_6CELL[5],  # HHH
        params=shared_params,
        horizon_months=3,
        seed=42,
    )
    m.regulator._intervene()  # Force
    assert shared_params.beta == original_beta, \
        f"shared_params.beta was mutated: {shared_params.beta} != {original_beta}"
    print(f"    OK: shared params.beta unchanged ({original_beta}); model.beta_effective={m.beta_effective}")

    # Test 5: profit CAN be negative (v2 fix — fixed costs)
    print("\n[5] Profit can become negative (v2 bugfix — exit dynamics)...")
    model = ElectricityMarketABM(
        scenario=SCENARIOS_6CELL[0],
        params=ModelParameters(fixed_cost_ratio=0.10),  # raise fixed cost
        horizon_months=24,
        seed=42,
    )
    df = model.run()
    # Check ALL agents (including exited) since exited firms drop from model.firms
    all_firms = [a for a in model.agents if isinstance(a, GenericFirm)]
    any_loss = any(
        f.last_profit < 0 or f.consecutive_loss_months > 0 or not f.alive
        for f in all_firms
    )
    final_exit_rate = df['ipp_exit_rate'].iloc[-1]
    assert any_loss, "No firm experienced losses — exit dynamics still broken"
    print(f"    OK: at least one firm had negative-profit months; "
          f"IPP exit rate after 24 months = {final_exit_rate:.1%}")

    # Test 6: M3 mechanism reduces σ for AI firms (v2 fix)
    print("\n[6] M3 mechanism reduces σ for AI firms (v2 corrected direction)...")
    m_no_m3 = ElectricityMarketABM(
        scenario=SCENARIOS_6CELL[0],
        params=ModelParameters(delta_sigma_reduction=0.0),
        horizon_months=3, seed=42,
    )
    m_no_m3.step()
    sigma_no_m3 = m_no_m3.ai_firms[0].last_forecast_error

    m_with_m3 = ElectricityMarketABM(
        scenario=SCENARIOS_6CELL[0],
        params=ModelParameters(delta_sigma_reduction=0.3),
        horizon_months=3, seed=42,
    )
    m_with_m3.step()
    sigma_with_m3 = m_with_m3.ai_firms[0].last_forecast_error

    assert sigma_with_m3 < sigma_no_m3, \
        f"M3 should reduce σ: {sigma_with_m3=} should be < {sigma_no_m3=}"
    print(f"    OK: σ without M3={sigma_no_m3:.4f}, σ with M3={sigma_with_m3:.4f}")

    # Test 7: Determinism — same seed → same results
    print("\n[7] Determinism: same seed produces identical results...")
    m1 = ElectricityMarketABM(scenario=SCENARIOS_6CELL[0], horizon_months=6, seed=42)
    df1 = m1.run()
    m2 = ElectricityMarketABM(scenario=SCENARIOS_6CELL[0], horizon_months=6, seed=42)
    df2 = m2.run()
    assert df1.equals(df2), "Same seed produced different results — non-deterministic!"
    print(f"    OK: deterministic")

    # Test 8 (v3 NEW): M2 disabled (theta_DA = infinity) → no acquisitions
    print("\n[8] M2 disabled (theta_DA = inf) → no acquisitions occur (v3)...")
    m_no_m2 = ElectricityMarketABM(
        scenario=SCENARIOS_6CELL[5],   # HHH
        params=ModelParameters(theta_DA=1e18),  # effectively disabled
        horizon_months=60,
        seed=42,
    )
    m_no_m2.run()
    n_acq_disabled = len(m_no_m2.acquisitions_history)
    assert n_acq_disabled == 0, \
        f"M2 should be disabled but {n_acq_disabled} acquisitions occurred"
    print(f"    OK: 0 acquisitions with theta_DA=inf")

    # Test 9 (v3 NEW): M2 enabled → acquisitions occur and AI capacity share grows
    print("\n[9] M2 enabled → acquisitions occur, AI capacity share grows (v3)...")
    m_with_m2 = ElectricityMarketABM(
        scenario=SCENARIOS_6CELL[5],   # HHH (high A → AI firms are big earners)
        params=ModelParameters(),       # default — M2 active
        horizon_months=120,
        seed=42,
    )
    df_m2 = m_with_m2.run()
    n_acq = len(m_with_m2.acquisitions_history)
    initial_ai_cap_share = df_m2["ai_capacity_share"].iloc[0]
    final_ai_cap_share = df_m2["ai_capacity_share"].iloc[-1]
    assert n_acq > 0, "M2 enabled but zero acquisitions over 120 months"
    assert final_ai_cap_share > initial_ai_cap_share, \
        f"AI capacity share should grow with M2: {initial_ai_cap_share:.3f} → {final_ai_cap_share:.3f}"
    print(f"    OK: {n_acq} acquisitions over 120 months; "
          f"AI capacity share {initial_ai_cap_share:.3f} → {final_ai_cap_share:.3f}; "
          f"final HHI={df_m2['hhi'].iloc[-1]:.0f}")

    # Test 10 (v4 NEW): pay-as-bid clearing produces firm-specific received prices
    print("\n[10] Pay-as-bid clearing makes M3 active (v4)...")
    # Compare AI Lerner under uniform vs pay-as-bid, with M3 ON
    m_uniform_m3 = ElectricityMarketABM(
        scenario=SCENARIOS_6CELL[5],
        params=ModelParameters(
            delta_sigma_reduction=0.3,
            clearing_rule="uniform",
        ),
        horizon_months=60, seed=42,
    )
    df_u = m_uniform_m3.run()
    m_payasbid_m3 = ElectricityMarketABM(
        scenario=SCENARIOS_6CELL[5],
        params=ModelParameters(
            delta_sigma_reduction=0.3,
            clearing_rule="pay_as_bid",
        ),
        horizon_months=60, seed=42,
    )
    df_p = m_payasbid_m3.run()
    lerner_u = df_u["ai_lerner"].iloc[-1]
    lerner_p = df_p["ai_lerner"].iloc[-1]
    # Under pay-as-bid, firms receive their own bids (which are above MC by markup),
    # so Lerner should be DIFFERENT from uniform. The exact direction depends on dynamics.
    assert abs(lerner_p - lerner_u) > 1e-6, \
        f"Pay-as-bid should produce different Lerner than uniform: u={lerner_u}, p={lerner_p}"
    # Also verify firms have different last_price_received under pay-as-bid
    prices_received = [f.last_price_received for f in m_payasbid_m3.firms if f.last_quantity_cleared > 0]
    assert len(set([round(p, 4) for p in prices_received])) > 1, \
        "Under pay-as-bid, cleared firms should have heterogeneous received prices"
    print(f"    OK: uniform Lerner={lerner_u:.4f}, pay-as-bid Lerner={lerner_p:.4f}; "
          f"price heterogeneity confirmed ({len(set([round(p,2) for p in prices_received]))} distinct prices)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


def run_self_tests(verbose: bool = True) -> int:
    """Public wrapper for the self-test suite.

    Captures the print output if verbose=False and returns the number of tests
    that passed. The internal `_self_test()` raises AssertionError on any
    failure, so a successful return guarantees all 10 tests passed.

    Parameters
    ----------
    verbose : bool
        If True, print test progress. If False, suppress output.

    Returns
    -------
    int : number of tests passed (10 on full success).
    """
    import io
    import contextlib

    if verbose:
        _self_test()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            _self_test()

    # All 10 tests are present in _self_test; we return 10 on success.
    # If any test fails, _self_test raises AssertionError before reaching here.
    return 10


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _self_test()
    else:
        # Smoke run: 12 months scenario LLL
        print("Smoke test: scenario LLL, 12 months, seed=42")
        print("(Run with `python baseline_model.py --test` for full self-tests.)\n")
        model = ElectricityMarketABM(
            scenario=SCENARIOS_6CELL[0],
            horizon_months=12,
            seed=42,
        )
        df = model.run()
        cols = ["month", "clearing_price", "hhi", "ai_lerner", "ai_market_share",
                "ai_capacity_share", "cumulative_acquisitions", "ipp_exit_rate"]
        print(df[cols].to_string(index=False))
        print(f"\nFinal HHI: {df['hhi'].iloc[-1]:.1f}")
        print(f"Final AI Lerner: {df['ai_lerner'].iloc[-1]:.4f}")
        print(f"Final AI capacity share: {df['ai_capacity_share'].iloc[-1]:.3f}")
        print(f"Final IPP exit rate: {df['ipp_exit_rate'].iloc[-1]:.2%}")
        print(f"Cumulative acquisitions: {df['cumulative_acquisitions'].iloc[-1]}")
        print(f"Regulator intervened: {df['regulator_intervened'].iloc[-1]}")
