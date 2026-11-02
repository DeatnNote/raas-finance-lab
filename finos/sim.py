"""A simulated robot-labor company with known truth.

Tutor-style business: customers pay for robot labor by the productive hour. Each contract has
  - a rate per hour, with a lower rate above a monthly tier threshold,
  - a monthly minimum commitment (hours), billed even if the robots work less,
  - a downtime credit: if fleet uptime at the customer is below 95% in a month, 10% off the usage charge,
  - a one-time deployment fee per robot, recognised over the contract term.

Robots emit session events (robot, sequence number, device clock time, upload time, productive hours). The
truth is the usage that really happened, at the right time, at the customer the robot was really working for.
What the raw event stream does to that truth (all planted, all counted):
  - retries: the same event uploaded twice (same robot + sequence number)
  - clock skew: some robots' clocks run hours fast or slow, so sessions near midnight land on the wrong day,
    and near month end in the wrong month
  - late uploads: robots that lose connectivity upload days later, sometimes after the invoice run
  - redeployment: robots move between customers; the current robot->customer map is wrong for history
  - billing configuration typos: the billing system's rate differs from the signed contract (CRM) for a few

Also simulated: customer payments (missing references, combined payments, short pays), vendor bills and card
spend (planted duplicates, miscoding, split purchases), and the CRM pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

START = pd.Timestamp("2025-01-01")
N_MONTHS = 18                                   # 2025-01 .. 2026-06
END = START + pd.DateOffset(months=N_MONTHS)
MONTHS = pd.period_range(START, periods=N_MONTHS, freq="M")
INVOICE_DAY = 3                                 # invoices for month m run on day 3 of month m+1
SLA_UPTIME, SLA_CREDIT = 0.95, 0.10
DEPLOY_FEE = 6_000.0


@dataclass
class Business:
    customers: pd.DataFrame       # CRM accounts
    contracts: pd.DataFrame       # signed terms (CRM)
    billing_config: pd.DataFrame  # terms as entered in the billing system (may contain typos)
    robots: pd.DataFrame
    assignments: pd.DataFrame     # robot -> customer with effective dates (the truth)
    usage_true: pd.DataFrame      # robot-day truth: hours, customer, up/down
    events: pd.DataFrame          # raw event stream
    pipeline: pd.DataFrame        # CRM opportunities (for the operating plan)
    payments: pd.DataFrame | None = None
    bank: pd.DataFrame | None = None
    bills: pd.DataFrame | None = None
    planted: dict | None = None


def month_of(ts) -> pd.Series:
    return pd.to_datetime(ts).dt.to_period("M")


def simulate(seed: int = 7) -> Business:
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END - pd.Timedelta(days=1), freq="D")
    # ---------------- customers and contracts
    n_cust = 36
    seg = rng.choice(["SMB", "Mid-market", "Enterprise"], n_cust, p=[0.5, 0.35, 0.15])
    customers = pd.DataFrame({"customer_id": [f"C{i:03d}" for i in range(n_cust)],
                              "name": [f"Customer {i:03d}" for i in range(n_cust)], "segment": seg})
    start_m = np.sort(rng.integers(0, 14, n_cust))
    start_m[:8] = 0                                              # an initial book of business
    n_rob = np.where(seg == "SMB", rng.integers(1, 4, n_cust), np.where(seg == "Mid-market",
                     rng.integers(3, 8, n_cust), rng.integers(8, 16, n_cust)))
    rate = np.round(rng.uniform(21, 29, n_cust) - np.where(seg == "Enterprise", 2.5, 0), 2)
    hours_day = rng.uniform(7, 15, n_cust)                       # productive hours per robot-day when running
    term = rng.choice([12, 24, 36], n_cust, p=[0.4, 0.4, 0.2])
    tier_hours = (n_rob * 22 * 10).round(-1)                     # monthly hours before the lower rate
    min_hours = (n_rob * 22 * rng.uniform(5, 9, n_cust)).round(-1)
    contracts = pd.DataFrame({
        "contract_id": [f"K{i:03d}" for i in range(n_cust)], "customer_id": customers.customer_id,
        "start_month": [str(MONTHS[m]) for m in start_m], "term_months": term, "robots": n_rob,
        "rate_per_hour": rate, "tier_hours": tier_hours, "tier_rate": np.round(rate * 0.85, 2),
        "min_hours": min_hours, "deploy_fee_per_robot": DEPLOY_FEE})
    billing_config = contracts.copy()
    typo = rng.choice(n_cust, 3, replace=False)
    billing_config.loc[typo[0], "rate_per_hour"] = round(contracts.rate_per_hour[typo[0]] * 0.9, 2)  # keyed wrong
    billing_config.loc[typo[1], "tier_rate"] = contracts.rate_per_hour[typo[1]]                       # no discount
    billing_config.loc[typo[2], "min_hours"] = 0.0                                                    # minimum dropped

    # ---------------- robots and their (true) deployment history
    rows, robots = [], []
    rid = 0
    for i, c in contracts.iterrows():
        go = START + pd.DateOffset(months=int(start_m[i])) + pd.Timedelta(days=int(rng.integers(0, 20)))
        for _ in range(int(c.robots)):
            skew = float(rng.choice([0, 0, 0, 0, 0, 0, 0, rng.uniform(4, 11), -rng.uniform(4, 11)]))
            robots.append({"robot_id": f"R{rid:04d}", "model": "Cassie", "clock_skew_h": round(skew, 2),
                           "flaky_uplink": bool(rng.random() < 0.12)})
            rows.append({"robot_id": f"R{rid:04d}", "customer_id": c.customer_id, "from": go, "to": END})
            rid += 1
    assignments = pd.DataFrame(rows)
    robots = pd.DataFrame(robots)
    # the fleet measures each robot's clock offset at heartbeats (NTP), to within about 6 minutes
    robots["measured_offset_h"] = (robots.clock_skew_h + rng.normal(0, 0.1, len(robots)) * (robots.clock_skew_h != 0)).round(3)
    # redeployments: 8 robots move to another customer mid-month
    movers = rng.choice(len(assignments), 8, replace=False)
    extra = []
    for j in movers:
        a = assignments.loc[j]
        when = a["from"] + pd.Timedelta(days=int(rng.integers(60, 200)))
        if when >= END - pd.Timedelta(days=30):
            continue
        when = when.normalize() + pd.Timedelta(days=int(rng.integers(5, 25)) - when.day + 1)
        dest = rng.choice([c for c in contracts.customer_id if c != a.customer_id
                           and START + pd.DateOffset(months=int(start_m[int(c[1:])])) < when])
        assignments.loc[j, "to"] = when
        extra.append({"robot_id": a.robot_id, "customer_id": dest, "from": when, "to": END})
    assignments = pd.concat([assignments, pd.DataFrame(extra)], ignore_index=True)

    # ---------------- robot-day truth
    cust_hours = dict(zip(contracts.customer_id, hours_day))
    U = []
    for a in assignments.itertuples():
        d = days[(days >= a._3.normalize()) & (days < a.to.normalize())]
        if len(d) == 0:
            continue
        n = len(d)
        wk = d.dayofweek < 5 + (rng.random() < 0.3)              # some sites run Saturdays
        ramp = np.clip(0.55 + 0.45 * np.arange(n) / 45, 0, 1)    # 6-week ramp after go-live
        p_down = 0.006 + (0.03 if rng.random() < 0.15 else 0.0)  # a few unreliable robots
        down = rng.random(n) < p_down
        # outages last a few days
        for k in np.where(down)[0]:
            down[k:k + int(rng.integers(1, 4))] = True
        h = np.where(wk & ~down, cust_hours[a.customer_id] * ramp * rng.uniform(0.75, 1.15, n), 0.0)
        U.append(pd.DataFrame({"robot_id": a.robot_id, "customer_id": a.customer_id, "date": d,
                               "hours": h.round(2), "scheduled": wk, "down": down & wk}))
    usage_true = pd.concat(U, ignore_index=True)
    usage_true["month"] = usage_true.date.dt.to_period("M")

    # ---------------- raw events: up to two sessions a day
    rb = robots.set_index("robot_id")
    ev = usage_true[usage_true.hours > 0].copy()
    two = rng.random(len(ev)) < 0.5
    first = np.where(two, (ev.hours * rng.uniform(0.35, 0.65, len(ev))).round(2), ev.hours)
    shift_start = pd.to_timedelta(rng.uniform(5, 9, len(ev)), unit="h")
    e1 = pd.DataFrame({"robot_id": ev.robot_id.values, "true_ts": (ev.date + shift_start).values,
                       "hours": first, "true_customer": ev.customer_id.values})
    e2 = pd.DataFrame({"robot_id": ev.robot_id.values[two],
                       "true_ts": (ev.date + shift_start + pd.to_timedelta(first + 0.5, unit="h")).values[two],
                       "hours": (ev.hours.values - first)[two].round(2), "true_customer": ev.customer_id.values[two]})
    # night shifts: a share of sessions start late in the evening (where clock skew moves them across days)
    events = pd.concat([e1, e2], ignore_index=True)
    night = rng.random(len(events)) < 0.25
    events.loc[night, "true_ts"] = events.true_ts[night].dt.normalize() + pd.to_timedelta(
        rng.uniform(20.5, 23.9, night.sum()), unit="h")
    events = events.sort_values(["robot_id", "true_ts"]).reset_index(drop=True)
    events["seq"] = events.groupby("robot_id").cumcount()
    events["event_id"] = events.robot_id + "-" + events.seq.astype(str)
    skew = events.robot_id.map(rb.clock_skew_h)
    events["device_ts"] = events.true_ts + pd.to_timedelta(skew, unit="h")
    flaky = events.robot_id.map(rb.flaky_uplink).to_numpy()
    delay_h = np.where(flaky & (rng.random(len(events)) < 0.15), rng.uniform(24, 24 * 12, len(events)),
                       rng.uniform(0.05, 2, len(events)))
    events["received_ts"] = events.true_ts + pd.to_timedelta(delay_h + np.maximum(skew.clip(lower=0) * 0, 0), unit="h")
    events["received_ts"] = events[["received_ts", "true_ts"]].max(axis=1)
    # retries: duplicate uploads
    dup = events.sample(frac=0.015, random_state=seed).copy()
    dup["received_ts"] = dup.received_ts + pd.to_timedelta(rng.uniform(0.01, 30, len(dup)), unit="h")
    events["retry"] = False
    dup["retry"] = True
    events = pd.concat([events, dup], ignore_index=True).sort_values("received_ts").reset_index(drop=True)

    # ---------------- CRM pipeline: opportunities that become the future go-lives (for the operating plan)
    opp = []
    for i, c in contracts.iterrows():
        sm = int(start_m[i])
        opp.append({"opp_id": f"O{i:03d}", "customer_id": c.customer_id, "robots": int(c.robots),
                    "rate_per_hour": float(c.rate_per_hour), "go_live_month": str(MONTHS[sm]),
                    "stage": "closed_won", "signed_month": str(MONTHS[max(sm - 2, 0)]), "won": True})
    # sales promised two late deals a month earlier than they actually went live
    late = [i for i in range(n_cust) if start_m[i] >= 13][:2]
    for i in late:
        opp[i]["go_live_month"] = str(MONTHS[int(start_m[i]) - 1])
    for j in range(24):                                          # open and lost opportunities
        sm = int(rng.integers(10, 18))
        won = rng.random() < 0.35
        opp.append({"opp_id": f"O{100 + j:03d}", "customer_id": f"P{j:03d}", "robots": int(rng.integers(2, 10)),
                    "rate_per_hour": round(float(rng.uniform(21, 29)), 2), "go_live_month": str(MONTHS[sm]),
                    "stage": rng.choice(["qualified", "proposal", "negotiation"]), "signed_month": None, "won": won})
    pipeline = pd.DataFrame(opp)
    return Business(customers, contracts, billing_config, robots, assignments, usage_true, events, pipeline)
