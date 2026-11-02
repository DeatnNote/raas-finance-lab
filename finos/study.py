"""Every number in the README.  python -m finos.study [--llm]  ->  results/results.json"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import datalayer
from .billing import ALL_FIXES, evaluate, run_billing, truth_invoices
from .close import run_close, score_cash, score_spend
from .economics import RobotCase, fleet_plan, hours_for_payback, sensitivity, summary
from .forecast import backtest, driver_forecast, robot_months
from .sim import MONTHS, simulate
from .variance import ClaudeWriter, check_narrative, month_facts, template_narrative

OUT = Path("results")


def contract_economics(con, case: RobotCase) -> list:
    rows = con.execute("SELECT * FROM v_customer_unit_economics").df()
    out = []
    for r in rows.itertuples():
        c = replace(case, rate_per_hour=r.revenue_per_robot_month / r.hours_per_robot_month,
                    hours_per_month=r.hours_per_robot_month)
        s = summary(c)
        out.append({"customer_id": r.customer_id, "segment": r.segment, "hours_per_robot_month": r.hours_per_robot_month,
                    "revenue_per_robot_month": r.revenue_per_robot_month, "contract_min_hours_per_robot": r.min_hours_per_robot,
                    "payback_months": s["payback_months"], "irr": s["irr"],
                    "hours_needed_for_24m_payback": hours_for_payback(c, 24)})
    return out


def run(llm: bool = False) -> dict:
    t0 = time.time()
    b = simulate()
    truth = truth_invoices(b)
    R = {"business": {"customers": len(b.customers), "robots": len(b.robots), "months": [str(MONTHS[0]), str(MONTHS[-1])],
                      "raw_events": int(len(b.events)), "retried_events": int(b.events.retry.sum()),
                      "robots_with_clock_skew": int((b.robots.clock_skew_h != 0).sum()),
                      "robots_with_flaky_uplink": int(b.robots.flaky_uplink.sum()),
                      "redeployed_robots": int(b.assignments.robot_id.duplicated().sum()),
                      "billing_config_typos": 3, "true_billings": float(truth.total.sum()),
                      "sla_credited_invoices": int((truth.sla_credit < 0).sum()),
                      "sla_credits": float(truth.sla_credit.sum())}}
    # 1) billing: legacy vs pipeline, and what each fix is worth
    pipe = run_billing(b)
    R["billing"] = {"pipeline": evaluate(b, pipe, truth), "legacy": evaluate(b, run_billing(b, ()), truth)}
    R["billing"]["without_each_fix"] = {f: evaluate(b, run_billing(b, tuple(x for x in ALL_FIXES if x != f)), truth)
                                        for f in ALL_FIXES}
    R["billing"]["config_mismatches"] = pipe.config_mismatches
    R["billing"]["usage_exceptions"] = pipe.exceptions
    # 2) close
    c = run_close(b, pipe)
    R["close"] = {"checks": c.checks, "cash": score_cash(c.deposits, c.cash_app), "spend": score_spend(c.bills, c.spend_flags),
                  "journals": len(c.ledger.keys), "ledger_lines": len(c.ledger.entries), "bills": int(len(c.bills)),
                  "invoices_issued": int(len(c.invoices))}
    # ledger controls
    from .ledger import Ledger, LedgerError
    L = Ledger()
    L.post("k1", "2026-01-31", [("1100", 100, 0), ("4000", 0, 100)])
    dup = L.post("k1", "2026-01-31", [("1100", 100, 0), ("4000", 0, 100)])
    ctl = {"idempotent_repost_returns_same_journal": dup == "J0000001" and len(L.entries) == 2}
    for name, kw in [("unbalanced_rejected", dict(lines=[("1100", 100, 0), ("4000", 0, 90)])),
                     ("self_approved_manual_rejected", dict(lines=[("6400", 30_000, 0), ("2000", 0, 30_000)], source="manual",
                                                           preparer="ana", approver="ana"))]:
        try:
            L.post(name, "2026-01-31", **kw)
            ctl[name] = False
        except LedgerError:
            ctl[name] = True
    c.ledger.entries[100]["debit"] += 1.0
    ctl["tampering_detected"] = not c.ledger.verify_chain()
    c.ledger.entries[100]["debit"] -= 1.0
    R["close"]["ledger_controls"] = ctl
    # 3) plan and variance
    origins = MONTHS[6:12]
    bts = {str(o): backtest(b, pipe.invoices, o) for o in origins}
    R["forecast"] = {"by_origin": bts, "mean_mape": {k: float(np.mean([bt[k]["mape"] for bt in bts.values()]))
                                                     for k in ("driver_plan", "run_rate", "linear_trend")}}
    plan = driver_forecast(b, pipe.invoices, MONTHS[11])
    facts = [month_facts(b, pipe.invoices, row) for _, row in plan.iterrows()]
    R["variance"] = [{k: v for k, v in f.items()} for f in facts]
    R["narratives_template"] = [template_narrative(f) for f in facts]
    R["narrative_checks_template"] = [check_narrative(n, f) for n, f in zip(R["narratives_template"], facts)]
    if llm:
        w = ClaudeWriter()
        R["narratives_claude"] = [w(f) for f in facts]
        R["claude_log"] = w.log
    # 4) economics
    con = datalayer.build(str(OUT / "finance.duckdb"), b, pipe, c, robot_months(b))
    case = RobotCase()
    R["economics"] = {"case": asdict(case), "base": summary(case), "financed_80pct": summary(replace(case, loan_share=0.8)),
                      "sensitivity": sensitivity(case),
                      "hours_for_payback": {"24": hours_for_payback(case, 24), "36": hours_for_payback(case, 36)},
                      "contracts": contract_economics(con, case)}
    sched = [5] * 6 + [10] * 6 + [20] * 12
    fe = {}
    for ls in (0.0, 0.8):
        f = fleet_plan(case, sched, ls)
        fe[str(ls)] = {"peak_cash_need": float(-f.cumulative.min()), "month_of_peak": int(f.cumulative.idxmin()),
                       "robots": int(sum(sched))}
    R["economics"]["fleet_plan_24m"] = fe
    R["datalayer"] = {t: int(con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in
                      ["customers", "contracts", "invoices", "gl", "bills", "deposits", "robot_months"]}
    con.close()
    R["seconds"] = time.time() - t0
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also write variance commentary with Claude (needs ANTHROPIC_API_KEY)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    R = run(a.llm)
    name = "results_llm.json" if a.llm else "results.json"
    (OUT / name).write_text(json.dumps(R, indent=2, default=str))
    print(json.dumps({"billing": {k: R["billing"][k] for k in ("pipeline", "legacy")}, "forecast": R["forecast"]["mean_mape"],
                      "economics": R["economics"]["base"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
