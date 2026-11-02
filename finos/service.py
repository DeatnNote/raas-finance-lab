"""FastAPI service over the finance OS.  uvicorn finos.service:app

  GET  /invoices/{customer_id}/{month}   invoice lines for a service month, with later adjustments
  GET  /close                            close tie-outs and exception counts
  GET  /variance/{month}                 plan vs. actual with the driver decomposition and checked commentary
  POST /economics                        robot unit economics for a deal (any RobotCase fields)
"""
from __future__ import annotations

from dataclasses import fields, replace

import pandas as pd
from fastapi import FastAPI, HTTPException

from .billing import run_billing
from .close import run_close, score_spend
from .economics import RobotCase, hours_for_payback, summary
from .forecast import driver_forecast
from .sim import MONTHS, simulate
from .variance import check_narrative, month_facts, template_narrative


def create_app(state: dict | None = None) -> FastAPI:
    app = FastAPI(title="RaaS Finance OS")
    S = state or {}

    @app.on_event("startup")
    def _load():
        if not S:
            b = simulate()
            run = run_billing(b)
            S.update(b=b, run=run, close=run_close(b, run), plan=driver_forecast(b, run.invoices, MONTHS[11]))

    @app.get("/healthz")
    def health():
        return {"ok": bool(S), "invoices": int(len(S["run"].invoices)) if S else 0}

    @app.get("/invoices/{customer_id}/{month}")
    def invoice(customer_id: str, month: str):
        inv = S["run"].invoices
        row = inv[(inv.customer_id == customer_id) & (inv.month == pd.Period(month, "M"))]
        if row.empty:
            raise HTTPException(404, "no invoice for that customer and month")
        adj = S["run"].adjustments
        later = adj[(adj.customer_id == customer_id) & (adj.service_month == pd.Period(month, "M"))] if len(adj) else adj
        r = row.iloc[0]
        return {"customer_id": customer_id, "service_month": month,
                "lines": {k: float(r[k]) for k in ("hours", "billable_hours", "usage_charge", "sla_credit", "deploy_fees", "total")},
                "uptime": float(r.uptime),
                "later_adjustments": [{"billed_month": str(a.billed_month), "amount": float(a.amount),
                                       "hours_added": float(a.hours_added)} for a in later.itertuples()]}

    @app.get("/close")
    def close():
        c = S["close"]
        return {"checks": c.checks, "spend_review": score_spend(c.bills, c.spend_flags),
                "cash_to_review": int((c.cash_app.how == "review").sum())}

    @app.get("/variance/{month}")
    def variance(month: str):
        p = S["plan"]
        row = p[p.month == pd.Period(month, "M")]
        if row.empty:
            raise HTTPException(404, "month not in the plan (2026-01..2026-06)")
        f = month_facts(S["b"], S["run"].invoices, row.iloc[0])
        n = template_narrative(f)
        return {"facts": {k: f[k] for k in ("plan_revenue", "actual_revenue", "variance", "effects")},
                "commentary": n, "number_check_problems": check_narrative(n, f)}

    @app.post("/economics")
    def economics(body: dict):
        allowed = {f.name for f in fields(RobotCase)}
        bad = set(body) - allowed
        if bad:
            raise HTTPException(422, f"unknown fields: {sorted(bad)}")
        c = replace(RobotCase(), **body)
        return {**summary(c), "hours_for_24m_payback": hours_for_payback(c, 24)}

    return app


app = create_app()
