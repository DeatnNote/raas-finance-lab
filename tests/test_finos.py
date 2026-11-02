import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from finos.billing import ALL_FIXES, evaluate, rate_usage, run_billing, truth_invoices
from finos.close import apply_cash, review_spend, run_close, score_spend, simulate_bills
from finos.economics import RobotCase, cash_flows, hours_for_payback, irr, payback_months, summary
from finos.forecast import driver_forecast
from finos.ledger import Ledger, LedgerError
from finos.service import create_app
from finos.sim import MONTHS, simulate
from finos.variance import ClaudeWriter, check_narrative, decompose, month_facts, template_narrative


@pytest.fixture(scope="module")
def world():
    b = simulate()
    run = run_billing(b)
    return b, run, truth_invoices(b)


def test_rating_minimum_tiers_and_credit():
    t = {"min_hours": 100, "tier_hours": 200, "rate_per_hour": 25.0, "tier_rate": 20.0}
    assert rate_usage(50, t)["usage_charge"] == 2500.0                 # minimum commitment
    assert rate_usage(250, t)["usage_charge"] == 200 * 25 + 50 * 20     # tier
    assert rate_usage(250, t)["billable_hours"] == 250


def test_pipeline_bills_every_service_month_correctly(world):
    b, run, truth = world
    ev = evaluate(b, run, truth)
    assert ev["final_exact_share"] == 1.0 and ev["abs_error"] < 1.0
    assert ev["config_mismatches"] == 3


def test_legacy_billing_is_wrong_and_each_fix_matters(world):
    b, run, truth = world
    assert evaluate(b, run_billing(b, ()), truth)["abs_error_share"] > 0.02
    for f in ALL_FIXES:
        ev = evaluate(b, run_billing(b, tuple(x for x in ALL_FIXES if x != f)), truth)
        assert ev["abs_error"] > 1_000, f


def test_late_uploads_are_trued_up_not_lost(world):
    b, run, truth = world
    assert len(run.adjustments) > 0 and (run.adjustments.billed_month > run.adjustments.service_month).all()
    no = run_billing(b, tuple(x for x in ALL_FIXES if x != "late_trueup"))
    ev = evaluate(b, no, truth)
    assert ev["overbilled"] < 1 and ev["underbilled"] > 1_000         # dropping late events only ever underbills


def test_ledger_controls():
    L = Ledger()
    j = L.post("a", "2026-01-31", [("1100", 100, 0), ("4000", 0, 100)])
    assert L.post("a", "2026-01-31", [("1100", 100, 0), ("4000", 0, 100)]) == j and len(L.entries) == 2
    with pytest.raises(LedgerError):
        L.post("b", "2026-01-31", [("1100", 100, 0), ("4000", 0, 99)])
    with pytest.raises(LedgerError):
        L.post("c", "2026-01-31", [("6400", 30_000, 0), ("2000", 0, 30_000)], source="manual", preparer="x", approver="x")
    L.post("d", "2026-01-31", [("6400", 30_000, 0), ("2000", 0, 30_000)], source="manual", preparer="x", approver="y")
    assert L.verify_chain()
    L.entries[0]["debit"] = 1_000
    assert not L.verify_chain()


def test_close_ties_out(world):
    b, run, _ = world
    c = run_close(b, run)
    assert all(x["pass"] for x in c.checks), [x for x in c.checks if not x["pass"]]


def test_cash_application_never_guesses_across_customers(world):
    b, run, _ = world
    c = run_close(b, run)
    m = c.deposits.merge(c.cash_app, on="deposit_id")
    cust = dict(zip(c.invoices.invoice_id, c.invoices.customer_id))
    for r in m.itertuples():
        if r.applied and r.truth:
            assert {cust[i] for i in r.applied} == {cust[i] for i in r.truth}


def test_spend_review_catches_planted_problems_without_flagging_recurring_bills():
    bills = simulate_bills()
    s = score_spend(bills, review_spend(bills))
    assert s["possible_split_purchase"]["caught"] == s["possible_split_purchase"]["planted"]
    assert s["unusual_account"]["caught"] == s["unusual_account"]["planted"]
    assert s["possible_duplicate"]["false_flags"] == 0                  # monthly subscriptions are not duplicates


def test_economics():
    c = RobotCase(hours_per_month=200, rate_per_hour=25, maintenance=1000, support=1000, robot_cost=70_000,
                  install_cost=10_000, deploy_fee=0, salvage_share=0)
    cf = cash_flows(c)
    assert cf[1] == 3000 and np.isclose(payback_months(cf), 80_000 / 3000)
    assert np.isclose(np.sum(cf / (1 + ((1 + irr(cf)) ** (1 / 12) - 1)) ** np.arange(len(cf))), 0, atol=1)
    h = hours_for_payback(c, 24)
    assert payback_months(cash_flows(replace(c, hours_per_month=h))) <= 24 < payback_months(cash_flows(replace(c, hours_per_month=h - 1)))
    assert summary(replace(c, loan_share=0.8))["levered_upfront_cash"] < summary(c)["upfront_cash"]


def test_variance_decomposition_adds_up_and_bad_narratives_are_rejected(world):
    b, run, _ = world
    plan = driver_forecast(b, run.invoices, MONTHS[11])
    for _, row in plan.iterrows():
        f = month_facts(b, run.invoices, row)
        assert np.isclose(sum(f["effects"].values()), f["variance"], atol=0.01)
        assert check_narrative(template_narrative(f), f) == []
    n = template_narrative(f)
    n["headline"] += " Utilisation fell 7.5% because two sites paused."
    assert any("7.5%" in p for p in check_narrative(n, f))


def test_claude_writer_retries_then_falls_back(world):
    b, run, _ = world
    f = month_facts(b, run.invoices, driver_forecast(b, run.invoices, MONTHS[11]).iloc[0])
    good = template_narrative(f)
    bad = {**good, "headline": good["headline"] + " Up 99.9% on last year."}
    replies = [bad, good]

    class Msgs:
        def create(self, **kw):
            assert kw["tool_choice"]["name"] == "write_variance" and "ONLY the numbers" in kw["system"]
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=replies.pop(0))],
                                   usage=SimpleNamespace(input_tokens=1000, output_tokens=200))
    w = ClaudeWriter(client=SimpleNamespace(messages=Msgs()))
    out = w(f)
    assert out["writer"] == "claude" and out["attempts"] == 2 and w.log[0]["problems"]
    replies[:] = [bad, bad]
    assert w(f)["writer"] == "template_fallback"


def test_api(world):
    b, run, _ = world
    state = {"b": b, "run": run, "close": run_close(b, run), "plan": driver_forecast(b, run.invoices, MONTHS[11])}
    with TestClient(create_app(state)) as c:
        r = c.get(f"/invoices/{run.invoices.customer_id.iloc[0]}/{run.invoices.month.iloc[0]}")
        assert r.status_code == 200 and r.json()["lines"]["total"] > 0
        assert all(x["pass"] for x in c.get("/close").json()["checks"])
        v = c.get("/variance/2026-03").json()
        assert v["number_check_problems"] == []
        e = c.post("/economics", json={"hours_per_month": 250}).json()
        assert e["payback_months"] < RobotCase().life_months
        assert c.post("/economics", json={"nope": 1}).status_code == 422
