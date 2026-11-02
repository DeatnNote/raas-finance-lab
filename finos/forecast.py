"""Driver-based revenue plan, built bottom-up per customer, and backtested against naive forecasts.

At the planning date (end of 2025-12) the plan knows: each contract's terms, every robot live today and its
age, signed deals with go-live dates, open CRM opportunities, and history up to that month. It forecasts the
next six months (2026-01..06) of usage revenue net of SLA credits.

  robots(customer, m)   live robots + signed go-lives (+ open opportunities x stage probability, as a separate line)
  hours per robot       each customer's recent hours per robot per weekday (mature robots), times a learned
                        ramp factor by month since go-live, times the weekdays in the month
  revenue               each customer's contract terms applied (minimum commitment, tiers), minus the
                        historical SLA-credit rate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .billing import rate_usage
from .sim import MONTHS, Business

STAGE_P = {"qualified": 0.15, "proposal": 0.30, "negotiation": 0.50}


def robot_months(b: Business) -> pd.DataFrame:
    """Robot-month history: hours per robot per month and months since the robot went live."""
    u = b.usage_true.groupby(["robot_id", "customer_id", "month"]).agg(hours=("hours", "sum"), days=("date", "nunique")).reset_index()
    first = b.assignments.groupby("robot_id")["from"].min().dt.to_period("M")
    u["age"] = (u.month - u.robot_id.map(first)).apply(lambda x: x.n)
    return u


def actuals(b: Business, invoices: pd.DataFrame) -> pd.DataFrame:
    rm = robot_months(b)
    full = rm[rm.days >= 28]
    g = invoices.groupby("month").agg(usage=("usage_charge", "sum"), credits=("sla_credit", "sum"),
                                      hours=("hours", "sum"), billable=("billable_hours", "sum")).reset_index()
    g["revenue"] = g.usage + g.credits
    g["robot_months"] = g.month.map(rm.groupby("month").days.sum() / rm.groupby("month").month.count().map(
        lambda _: 1) / g.month.map(lambda m: m.days_in_month))
    g["robots_live"] = g.month.map(rm.groupby("month").robot_id.nunique())
    g["hours_per_robot"] = g.hours / g.robots_live
    g["rate_per_billable_hour"] = g.usage / g.billable
    return g


def weekdays(m: pd.Period) -> int:
    return int(np.busday_count(m.start_time.date(), (m + 1).start_time.date()))


def ramp_curve(b: Business, origin: pd.Period) -> pd.Series:
    """Hours per robot per weekday by months since go-live (0..6+), as a share of the mature level."""
    rm = robot_months(b)
    h = rm[(rm.month <= origin)].copy()
    live_share = h.days / h.month.map(lambda m: m.days_in_month)              # part-month at go-live
    h["per_wd"] = h.hours / (h.month.map(weekdays) * live_share)
    c = h.groupby(h.age.clip(upper=6)).per_wd.mean()
    c = c.reindex(range(7)).ffill()
    return c / c[6]


def customer_rates(b: Business, origin: pd.Period) -> pd.Series:
    """Recent hours per weekday of a mature robot, per customer (last 3 months, robots live 3+ months)."""
    rm = robot_months(b)
    h = rm[(rm.month <= origin) & (rm.month > origin - 3) & (rm.age >= 3)].copy()
    h["per_wd"] = h.hours / h.month.map(weekdays)
    return h.groupby("customer_id").per_wd.mean()


def driver_forecast(b: Business, invoices: pd.DataFrame, origin: pd.Period, horizon: int = 6,
                    include_open_pipeline: bool = False) -> pd.DataFrame:
    ramp = ramp_curve(b, origin)
    per_wd = customer_rates(b, origin)
    rm = robot_months(b)
    mature_all = float((rm[(rm.month <= origin) & (rm.age >= 3)].hours /
                        rm[(rm.month <= origin) & (rm.age >= 3)].month.map(weekdays)).mean())
    terms = b.contracts.set_index("customer_id")
    hist = invoices[invoices.month <= origin]
    credit_rate = float(-hist.sla_credit.sum() / hist.usage_charge.sum())
    a = b.assignments.copy()
    a["start"] = a["from"].dt.to_period("M")
    first = a.groupby("robot_id").start.min()
    live = a[(a.start <= origin) & (a.to.dt.to_period("M") > origin)]
    signed = b.pipeline[(b.pipeline.stage == "closed_won")
                        & (b.pipeline.go_live_month.map(lambda x: pd.Period(x, "M")) > origin)
                        & (b.pipeline.signed_month.map(lambda x: pd.Period(x, "M")) <= origin)]
    rows = []
    for k in range(1, horizon + 1):
        m = origin + k
        wd = weekdays(m)
        hours = {}
        for cid, g in live.groupby("customer_id"):
            lvl = per_wd.get(cid, mature_all)
            hours[cid] = sum(lvl * ramp[min((m - first[r]).n, 6)] * wd for r in g.robot_id)
        for s in signed.itertuples():
            gl = pd.Period(s.go_live_month, "M")
            if gl <= m:
                age = (m - gl).n
                hours[s.customer_id] = hours.get(s.customer_id, 0) + s.robots * mature_all * ramp[min(age, 6)] * wd * (
                    0.5 if age == 0 else 1.0)                                    # go-lives land mid-month
        usage = sum(rate_usage(h, terms.loc[c])["usage_charge"] for c, h in hours.items())
        pipe = 0.0
        if include_open_pipeline:
            for o in b.pipeline[b.pipeline.stage.isin(STAGE_P)].itertuples():
                gl = pd.Period(o.go_live_month, "M")
                if gl <= m:
                    pipe += STAGE_P[o.stage] * o.robots * mature_all * ramp[min((m - gl).n, 6)] * wd * o.rate_per_hour
        robots = len(live) + sum(s.robots for s in signed.itertuples() if pd.Period(s.go_live_month, "M") <= m)
        rows.append({"month": m, "usage": usage + pipe, "revenue": (usage + pipe) * (1 - credit_rate),
                     "robots": robots, "hours": sum(hours.values()), "credit_rate": credit_rate,
                     "hours_by_customer": hours})
    return pd.DataFrame(rows)


def monthly_revenue(invoices: pd.DataFrame) -> pd.Series:
    return (invoices.usage_charge + invoices.sla_credit).groupby(invoices.month).sum()


def naive_forecasts(invoices: pd.DataFrame, origin: pd.Period, horizon: int = 6) -> pd.DataFrame:
    g = monthly_revenue(invoices)
    h = g[g.index <= origin]
    last6 = h.iloc[-6:]
    slope = np.polyfit(np.arange(6), last6.values, 1)[0]
    return pd.DataFrame({"month": [origin + k for k in range(1, horizon + 1)],
                         "run_rate": [h.iloc[-1]] * horizon,
                         "linear_trend": [h.iloc[-1] + slope * k for k in range(1, horizon + 1)]})


def backtest(b: Business, invoices: pd.DataFrame, origin: pd.Period) -> dict:
    act = monthly_revenue(invoices)
    drv = driver_forecast(b, invoices, origin)
    nv = naive_forecasts(invoices, origin)
    out = {"months": [str(m) for m in drv.month], "actual": [float(act[m]) for m in drv.month]}
    for name, s in [("driver_plan", drv.revenue), ("run_rate", nv.run_rate), ("linear_trend", nv.linear_trend)]:
        f = np.asarray(s, float)
        a = np.asarray(out["actual"])
        out[name] = {"forecast": f.tolist(), "mape": float(np.mean(np.abs(f - a) / a)),
                     "total_error_share": float((f.sum() - a.sum()) / a.sum())}
    return out
