"""Budget-vs-actual variance: an exact driver decomposition, and narratives whose numbers are checked.

Usage revenue = robots x hours per robot x realised rate per hour x (1 - SLA credit rate). Moving one driver
at a time from plan to actual splits the variance into fleet, utilisation, rate and credit effects that add up
exactly to the total.

The narrative is written either by a template or by Claude. Either way, every number in it is checked against
the computed facts before it can go into the close package: the writer explains, it never computes.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .forecast import robot_months


def decompose(plan: dict, actual: dict) -> dict:
    Rp, Hp, Pp, Cp = plan["robots"], plan["hours"] / plan["robots"], plan["usage"] / plan["hours"], plan["credit_rate"]
    Ra, Ha, Pa, Ca = actual["robots"], actual["hours"] / actual["robots"], actual["usage"] / actual["hours"], actual["credit_rate"]
    fx = {"fleet": (Ra - Rp) * Hp * Pp * (1 - Cp), "utilisation": Ra * (Ha - Hp) * Pp * (1 - Cp),
          "rate": Ra * Ha * (Pa - Pp) * (1 - Cp), "credits": Ra * Ha * Pa * (Cp - Ca)}
    rev_p, rev_a = Rp * Hp * Pp * (1 - Cp), Ra * Ha * Pa * (1 - Ca)
    return {"plan_revenue": rev_p, "actual_revenue": rev_a, "variance": rev_a - rev_p,
            "effects": fx, "drivers": {"robots": [Rp, Ra], "hours_per_robot": [Hp, Ha], "rate_per_hour": [Pp, Pa],
                                       "credit_rate": [Cp, Ca]}}


def month_facts(b, invoices: pd.DataFrame, plan_row: pd.Series) -> dict:
    m = plan_row["month"]
    inv = invoices[invoices.month == m]
    rm = robot_months(b)
    robots = int(rm[rm.month == m].robot_id.nunique())
    actual = {"robots": robots, "hours": float(inv.hours.sum()), "usage": float(inv.usage_charge.sum()),
              "credit_rate": float(-inv.sla_credit.sum() / inv.usage_charge.sum())}
    plan = {"robots": float(plan_row["robots"]), "hours": float(plan_row["hours"]), "usage": float(plan_row["usage"]),
            "credit_rate": float(plan_row["credit_rate"])}
    d = decompose(plan, actual)
    # customers behind the utilisation effect
    ph = pd.Series(plan_row["hours_by_customer"])
    ah = inv.set_index("customer_id").hours
    gap = ah.reindex(ph.index.union(ah.index)).fillna(0) - ph.reindex(ph.index.union(ah.index)).fillna(0)
    top = gap.abs().sort_values(ascending=False).head(3).index
    d["top_customers_hours_vs_plan"] = {c: round(float(gap[c]), 0) for c in top}
    credited = inv[inv.sla_credit < 0]
    d["sla_credit_customers"] = sorted(credited.customer_id.tolist())
    d["month"] = str(m)
    return d


# ------------------------------------------------------------------ narratives
def _k(x: float) -> str:
    return f"${abs(x) / 1000:,.1f}K"


def template_narrative(f: dict) -> dict:
    e = f["effects"]
    order = sorted(e, key=lambda k: -abs(e[k]))
    words = {"fleet": "fewer/more robots live than planned", "utilisation": "hours per robot vs. plan",
             "rate": "realised rate per hour (tiers and minimums)", "credits": "SLA downtime credits"}
    drivers = [{"driver": k, "amount": round(e[k], 2),
                "explanation": f"{words[k]}: {'+' if e[k] >= 0 else '-'}{_k(e[k])}"} for k in order]
    v = f["variance"]
    head = (f"{f['month']}: usage revenue {_k(f['actual_revenue'])} vs. plan {_k(f['plan_revenue'])}, "
            f"{'+' if v >= 0 else '-'}{_k(v)} ({100 * v / f['plan_revenue']:+.1f}%).")
    return {"headline": head, "drivers": drivers, "watch_items": [
        f"Largest hours gaps vs. plan: " + ", ".join(f"{c} {h:+,.0f} h" for c, h in f["top_customers_hours_vs_plan"].items())]}


NUM = re.compile(r"(?<![A-Za-z0-9.])([+-]?)\$?(\d[\d,]*\.?\d*)\s*(K|%|h)?")   # not inside ids like C026


def allowed_numbers(f: dict) -> list:
    """Every number the narrative may quote, in the units it may quote them in."""
    vals = [f["plan_revenue"], f["actual_revenue"], f["variance"], *f["effects"].values()]
    out = []
    for v in vals:
        out += [abs(v), abs(v) / 1000]
    out.append(abs(100 * f["variance"] / f["plan_revenue"]))
    for a, b in f["drivers"].values():
        out += [a, b, abs(b - a), 100 * a, 100 * b, abs(100 * (b - a))]
        if a:
            out.append(abs(100 * (b - a) / a))
    out += [abs(h) for h in f["top_customers_hours_vs_plan"].values()]
    out += [float(f["month"][:4]), float(f["month"][5:7]), float(len(f["sla_credit_customers"]))]
    return out


def check_narrative(n: dict, f: dict) -> list:
    """Problems with a narrative: numbers that are not in the facts, driver amounts that don't match the
    decomposition, or drivers that don't add up to the variance."""
    problems = []
    ok = allowed_numbers(f)
    text = json.dumps({k: v for k, v in n.items() if k != "drivers"}) + json.dumps(
        [d.get("explanation", "") for d in n.get("drivers", [])])
    for sign, num, unit in NUM.findall(text):
        x = float(num.replace(",", ""))
        # customer ids like C012 and robot ids are not quantities
        if x < 10 and unit == "":
            continue
        tol = 0.051 if unit in ("K", "%") else max(0.51, 0.005 * x)
        if not any(abs(x - a) <= tol for a in ok):
            problems.append(f"number not in facts: {sign}{num}{unit}")
    eff = f["effects"]
    for d in n.get("drivers", []):
        k = d.get("driver")
        if k not in eff:
            problems.append(f"unknown driver {k}")
        elif abs(float(d.get("amount", 0)) - eff[k]) > 1.0:
            problems.append(f"driver {k}: amount {d.get('amount')} vs computed {eff[k]:.2f}")
    if abs(sum(float(d.get("amount", 0)) for d in n.get("drivers", [])) - f["variance"]) > 2.0:
        problems.append("drivers do not add up to the variance")
    return problems


TOOL = {"name": "write_variance", "description": "Write the month's revenue variance commentary.",
        "input_schema": {"type": "object", "required": ["headline", "drivers", "watch_items"], "properties": {
            "headline": {"type": "string"},
            "drivers": {"type": "array", "items": {"type": "object", "required": ["driver", "amount", "explanation"],
                        "properties": {"driver": {"type": "string", "enum": ["fleet", "utilisation", "rate", "credits"]},
                                       "amount": {"type": "number"}, "explanation": {"type": "string"}}}},
            "watch_items": {"type": "array", "items": {"type": "string"}}}}}
SYSTEM = ("You write budget-vs-actual commentary for a robot-labor company's monthly close. Use ONLY the numbers "
          "in the facts JSON; never compute new totals, ratios or percentages that are not given. Driver amounts "
          "must be copied exactly from facts.effects. Write dollar amounts as $X.XK. Be specific and brief.")


class ClaudeWriter:
    """Variance commentary from Claude, with the number check and one retry; falls back to the template."""

    def __init__(self, client=None, model: str = "claude-haiku-4-5"):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model = client, model
        self.log = []

    def _ask(self, f: dict, feedback: str | None):
        msg = "Facts:\n" + json.dumps(f, default=float)
        if feedback:
            msg += "\n\nYour previous draft was rejected by the number check:\n" + feedback
        r = self.client.messages.create(model=self.model, max_tokens=800, system=SYSTEM, tools=[TOOL],
                                        tool_choice={"type": "tool", "name": "write_variance"},
                                        messages=[{"role": "user", "content": msg}])
        block = next(c for c in r.content if getattr(c, "type", "") == "tool_use")
        return block.input, getattr(r, "usage", None)

    def __call__(self, f: dict) -> dict:
        feedback = None
        for attempt in (1, 2):
            n, usage = self._ask(f, feedback)
            problems = check_narrative(n, f)
            self.log.append({"month": f["month"], "attempt": attempt, "problems": problems,
                             "input_tokens": getattr(usage, "input_tokens", 0),
                             "output_tokens": getattr(usage, "output_tokens", 0)})
            if not problems:
                return {**n, "writer": "claude", "attempts": attempt}
            feedback = "\n".join(problems)
        return {**template_narrative(f), "writer": "template_fallback", "rejected_claude": n, "problems": problems}
