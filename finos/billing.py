"""Usage-to-invoice: from the raw robot event stream to invoices that reconcile to what the robots did.

Pipeline steps (each can be switched off to measure what it is worth):
  dedupe         drop retried uploads (same robot + sequence number)
  clock          move each event to the right day/month using the fleet's measured clock offset per robot
  as_of_map      attribute usage to the customer the robot worked for *at that time* (effective-dated map)
  late_trueup    events uploaded after the invoice run are re-rated into their service month and billed as a
                 prior-period adjustment on the next invoice, not dropped
  config_check   compare the billing system's rates/minimums with the signed contract (CRM) before rating
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .sim import INVOICE_DAY, MONTHS, SLA_CREDIT, SLA_UPTIME, Business

ALL_FIXES = ("dedupe", "clock", "as_of_map", "late_trueup", "config_check")


def cutoff(month: pd.Period) -> pd.Timestamp:
    return (month + 1).to_timestamp() + pd.Timedelta(days=INVOICE_DAY - 1)


def rate_usage(hours: float, terms) -> dict:
    """Charge for one customer-month: minimum commitment, then tiered rates."""
    billable = max(hours, float(terms["min_hours"]))
    t = float(terms["tier_hours"])
    charge = min(billable, t) * float(terms["rate_per_hour"]) + max(billable - t, 0) * float(terms["tier_rate"])
    return {"hours": round(hours, 2), "billable_hours": round(billable, 2), "usage_charge": round(charge, 2)}


def uptime_table(b: Business) -> pd.DataFrame:
    """Robot operations data: scheduled robot-days and robot-days down, per customer-month (from heartbeats)."""
    u = b.usage_true
    g = u.groupby(["customer_id", "month"]).agg(sched=("scheduled", "sum"), down=("down", "sum")).reset_index()
    g["uptime"] = 1 - g.down / g.sched.clip(lower=1)
    return g


def new_robots(b: Business) -> pd.DataFrame:
    """Deployment fees: billed once, in the month a robot first goes live at a contracted customer."""
    first = b.assignments.sort_values("from").groupby("robot_id").head(1)
    first = first.assign(month=first["from"].dt.to_period("M"))
    return first.groupby(["customer_id", "month"]).size().rename("new_robots").reset_index()


def invoice_lines(b: Business, hours: pd.DataFrame, terms: pd.DataFrame, upt: pd.DataFrame) -> pd.DataFrame:
    """Rate customer-month hours. `hours`: customer_id, month, hours."""
    t = terms.set_index("customer_id")
    upt = upt.set_index(["customer_id", "month"])
    nr = new_robots(b).set_index(["customer_id", "month"]).new_robots
    out = []
    for r in hours.itertuples():
        k = t.loc[r.customer_id]
        if r.month < pd.Period(k.start_month, "M"):
            continue
        line = rate_usage(r.hours, k)
        up = upt.uptime.get((r.customer_id, r.month), 1.0)
        line["sla_credit"] = round(-SLA_CREDIT * line["usage_charge"], 2) if up < SLA_UPTIME else 0.0
        line["deploy_fees"] = float(nr.get((r.customer_id, r.month), 0)) * float(k.deploy_fee_per_robot)
        out.append({"customer_id": r.customer_id, "month": r.month, "uptime": round(up, 4), **line})
    df = pd.DataFrame(out)
    df["total"] = (df.usage_charge + df.sla_credit + df.deploy_fees).round(2)
    return df


def _active_grid(b: Business) -> pd.DataFrame:
    """Every contracted customer-month (so a month with zero usage still bills its minimum)."""
    rows = [(c.customer_id, m) for c in b.contracts.itertuples() for m in MONTHS if m >= pd.Period(c.start_month, "M")]
    return pd.DataFrame(rows, columns=["customer_id", "month"])


def truth_invoices(b: Business) -> pd.DataFrame:
    h = b.usage_true.groupby(["customer_id", "month"]).hours.sum().reset_index()
    h = _active_grid(b).merge(h, how="left").fillna({"hours": 0.0})
    return invoice_lines(b, h, b.contracts, uptime_table(b))


# ------------------------------------------------------------------ the pipeline
@dataclass
class BillingRun:
    invoices: pd.DataFrame            # one row per customer-month as invoiced at the run (service month = month)
    adjustments: pd.DataFrame         # prior-period adjustments billed later
    exceptions: list = field(default_factory=list)
    config_mismatches: list = field(default_factory=list)


def prepare_events(b: Business, fixes=ALL_FIXES) -> pd.DataFrame:
    e = b.events.copy()
    if "dedupe" in fixes:
        e = e.sort_values("received_ts").drop_duplicates("event_id", keep="first")
    if "clock" in fixes:
        off = e.robot_id.map(b.robots.set_index("robot_id").measured_offset_h)  # measured by the fleet (NTP)
        e["ts"] = e.device_ts - pd.to_timedelta(off, unit="h")
    else:
        e["ts"] = e.device_ts
    e["month"] = e.ts.dt.to_period("M")
    e["ts"] = e.ts.astype("datetime64[ns]")            # pandas 3 may mix ns and us resolutions; merge_asof needs one
    if "as_of_map" in fixes:
        a = b.assignments.sort_values("from").assign(**{"from": lambda d: d["from"].astype("datetime64[ns]")})
        e = pd.merge_asof(e.sort_values("ts"), a[["robot_id", "from", "customer_id"]].rename(columns={"from": "ts_from"}),
                          left_on="ts", right_on="ts_from", by="robot_id", direction="backward")
    else:                                                                       # today's map, applied to history
        cur = b.assignments.sort_values("from").groupby("robot_id").tail(1).set_index("robot_id").customer_id
        e["customer_id"] = e.robot_id.map(cur)
    return e


def run_billing(b: Business, fixes=ALL_FIXES) -> BillingRun:
    terms = b.billing_config.copy()
    mismatches = []
    if "config_check" in fixes:
        cols = ["rate_per_hour", "tier_rate", "tier_hours", "min_hours"]
        m = b.billing_config.set_index("contract_id")[cols].ne(b.contracts.set_index("contract_id")[cols])
        for k, row in m.iterrows():
            for c in cols:
                if row[c]:
                    mismatches.append({"contract_id": k, "field": c,
                                       "billing_system": float(b.billing_config.set_index("contract_id").at[k, c]),
                                       "signed": float(b.contracts.set_index("contract_id").at[k, c])})
        terms = b.contracts.copy()                                              # rate from the signed terms
    e = prepare_events(b, fixes)
    upt = uptime_table(b)
    grid = _active_grid(b)
    invoices, adjustments, exceptions = [], [], []
    billed = {}                                                                 # (customer, month) -> rated line
    for m in MONTHS:
        cut = cutoff(m)
        seen = e[(e.received_ts < cut) & (e.month <= m)]
        # this month's usage, as known at the run
        h = seen[seen.month == m].groupby("customer_id").hours.sum()
        g = grid[grid.month == m].assign(hours=lambda d: d.customer_id.map(h).fillna(0.0))
        inv = invoice_lines(b, g, terms, upt)
        invoices.append(inv)
        for r in inv.itertuples():
            billed[(r.customer_id, r.month)] = r
        # reasonableness control: more than 24 billed hours on a robot-day
        day = seen[seen.month == m].assign(d=lambda d: d.ts.dt.date).groupby(["robot_id", "d"]).hours.sum()
        for (rid, d), hrs in day[day > 24].items():
            exceptions.append({"month": str(m), "robot_id": rid, "date": str(d), "hours": round(float(hrs), 2)})
        # prior-period true-ups for events that arrived after an earlier run
        if "late_trueup" in fixes and m > MONTHS[0]:
            prev_cut = cutoff(m - 1)
            late = e[(e.received_ts >= prev_cut) & (e.received_ts < cut) & (e.month < m)]
            for (cid, sm), _ in late.groupby(["customer_id", "month"]):
                hh = e[(e.customer_id == cid) & (e.month == sm) & (e.received_ts < cut)].hours.sum()
                old = billed.get((cid, sm))
                if old is None:
                    continue
                new = invoice_lines(b, pd.DataFrame({"customer_id": [cid], "month": [sm], "hours": [hh]}), terms, upt)
                if new.empty:
                    continue
                nw = new.iloc[0]
                delta = round(nw.total - old.total, 2)
                if abs(delta) >= 0.01:
                    adjustments.append({"customer_id": cid, "service_month": sm, "billed_month": m, "amount": delta,
                                        "hours_added": round(nw.hours - old.hours, 2)})
                billed[(cid, sm)] = nw
    return BillingRun(pd.concat(invoices, ignore_index=True), pd.DataFrame(adjustments), exceptions, mismatches)


def evaluate(b: Business, run: BillingRun, truth: pd.DataFrame | None = None) -> dict:
    """Compare billing to the truth, by service month (invoice + any later adjustment) and as first invoiced.
    The last month is excluded: its late uploads would be trued up after the simulation ends."""
    truth = truth_invoices(b) if truth is None else truth
    t = truth.set_index(["customer_id", "month"]).total
    inv = run.invoices.set_index(["customer_id", "month"]).total
    if len(run.adjustments):
        adj = run.adjustments.groupby(["customer_id", "service_month"]).amount.sum()
        adj.index.names = ["customer_id", "month"]
    else:
        adj = pd.Series(0.0, index=inv.index)
    final = inv.add(adj, fill_value=0.0)
    keep = t.index.get_level_values("month") < MONTHS[-1]
    t, inv, final = t[keep], inv.reindex(t.index[keep]).fillna(0), final.reindex(t.index[keep]).fillna(0)
    d_first, d_final = inv - t, final - t
    return {
        "invoices": int(len(t)), "true_billings": float(t.sum()),
        "first_invoice_exact_share": float((d_first.abs() < 1).mean()),
        "final_exact_share": float((d_final.abs() < 1).mean()),
        "overbilled": float(d_final.clip(lower=0).sum()), "underbilled": float(-d_final.clip(upper=0).sum()),
        "abs_error": float(d_final.abs().sum()), "abs_error_share": float(d_final.abs().sum() / t.sum()),
        "worst_invoice_error": float(d_final.abs().max()),
        "adjustments": int(len(run.adjustments)), "adjustment_total": float(run.adjustments.amount.sum()) if len(run.adjustments) else 0.0,
        "exceptions_over_24h": len(run.exceptions), "config_mismatches": len(run.config_mismatches),
    }
