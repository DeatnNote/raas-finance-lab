"""Robot unit economics, deal pricing and fleet financing, in code.

One robot, month by month: build and install it, earn usage revenue at the customer's rate and utilisation,
pay for maintenance and support, optionally finance most of the build with an equipment loan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobotCase:
    robot_cost: float = 75_000.0        # build cost (bill of materials + assembly)
    install_cost: float = 12_000.0      # field deployment and site integration
    deploy_fee: float = 6_000.0         # one-time fee charged to the customer
    rate_per_hour: float = 25.0
    hours_per_month: float = 220.0      # productive hours billed
    maintenance: float = 1_100.0        # per month: field service + parts
    support: float = 900.0              # per month: remote operators, cloud, monitoring
    life_months: int = 48
    salvage_share: float = 0.10
    loan_share: float = 0.0             # share of build cost financed
    loan_apr: float = 0.095
    loan_months: int = 36
    discount_rate: float = 0.12


def cash_flows(c: RobotCase, levered: bool = False) -> np.ndarray:
    n = c.life_months
    cf = np.zeros(n + 1)
    cf[0] = -(c.robot_cost + c.install_cost) + c.deploy_fee
    cf[1:] = c.rate_per_hour * c.hours_per_month - c.maintenance - c.support
    cf[n] += c.salvage_share * c.robot_cost
    if levered and c.loan_share > 0:
        principal = c.loan_share * c.robot_cost
        r = c.loan_apr / 12
        pmt = principal * r / (1 - (1 + r) ** -c.loan_months)
        cf[0] += principal
        cf[1:c.loan_months + 1] -= pmt
    return cf


def npv(cf: np.ndarray, annual: float) -> float:
    r = (1 + annual) ** (1 / 12) - 1
    return float(np.sum(cf / (1 + r) ** np.arange(len(cf))))


def irr(cf: np.ndarray) -> float:
    """Annualised IRR by bisection on the monthly rate (nan if the flows never pay back)."""
    if cf.sum() <= 0:
        return float("nan")
    lo, hi = -0.99, 1.0
    f = lambda r: np.sum(cf / (1 + r) ** np.arange(len(cf)))
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float((1 + lo) ** 12 - 1)


def payback_months(cf: np.ndarray) -> float:
    """Months until cumulative cash turns positive, interpolated within the month."""
    cum = np.cumsum(cf)
    idx = np.where(cum >= 0)[0]
    if not len(idx):
        return float("inf")
    i = int(idx[0])
    if i == 0:
        return 0.0
    return float(i - 1 + (-cum[i - 1]) / cf[i])


def summary(c: RobotCase) -> dict:
    u, l = cash_flows(c), cash_flows(c, levered=True)
    return {"monthly_contribution": float(u[1]), "contribution_margin": float(u[1] / (c.rate_per_hour * c.hours_per_month)),
            "upfront_cash": float(-u[0]), "payback_months": payback_months(u), "irr": irr(u),
            "npv": npv(u, c.discount_rate), "levered_upfront_cash": float(-l[0]),
            "levered_payback_months": payback_months(l), "levered_irr": irr(l)}


def sensitivity(c: RobotCase) -> list:
    """Payback months when one driver moves, everything else fixed."""
    base = summary(c)["payback_months"]
    rows = []
    for field, lo, hi in [("hours_per_month", 0.8, 1.2), ("rate_per_hour", 0.9, 1.1), ("robot_cost", 0.85, 1.15),
                          ("maintenance", 0.7, 1.3), ("install_cost", 0.7, 1.3)]:
        v = getattr(c, field)
        rows.append({"driver": field, "low": lo, "high": hi, "base_payback": base,
                     "payback_low": summary(replace(c, **{field: v * lo}))["payback_months"],
                     "payback_high": summary(replace(c, **{field: v * hi}))["payback_months"]})
    return rows


def hours_for_payback(c: RobotCase, months: int) -> float:
    """Billed hours per month a robot needs to pay back within `months` (unlevered)."""
    lo, hi = 0.0, 744.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if payback_months(cash_flows(replace(c, hours_per_month=mid))) <= months:
            hi = mid
        else:
            lo = mid
    return float(np.ceil(hi * 10) / 10)


def fleet_plan(c: RobotCase, deploys_per_month: list, loan_share: float) -> pd.DataFrame:
    """Company cash need for a deployment schedule: every robot follows the per-robot cash flows."""
    one = cash_flows(replace(c, loan_share=loan_share), levered=loan_share > 0)
    n = len(deploys_per_month) + c.life_months + 1
    total = np.zeros(n)
    for t, k in enumerate(deploys_per_month):
        total[t:t + len(one)] += k * one
    cum = np.cumsum(total)
    return pd.DataFrame({"month": np.arange(n), "cash_flow": total, "cumulative": cum})
