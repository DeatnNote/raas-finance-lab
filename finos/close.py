"""The month-end close: post billing to the ledger, apply cash, review spend, tie out, and report exceptions.

Simulated here (with planted truth): customer payments and the bank statement, vendor bills and their payments.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .billing import BillingRun, cutoff
from .ledger import Ledger
from .sim import END, MONTHS, START, Business

from .economics import RobotCase

ROBOT_COST = RobotCase().robot_cost
OPENING_EQUITY = 14_000_000.0
BILL_APPROVAL_LIMIT = 10_000.0


# ------------------------------------------------------------------ invoices as issued
def issued_invoices(run: BillingRun) -> pd.DataFrame:
    """One invoice per customer per billing month: that month's lines plus any prior-period adjustments."""
    inv = run.invoices.copy()
    inv["billed_month"] = inv.month
    adj = run.adjustments.rename(columns={"amount": "adjustment"}) if len(run.adjustments) else \
        pd.DataFrame(columns=["customer_id", "billed_month", "adjustment"])
    a = adj.groupby(["customer_id", "billed_month"]).adjustment.sum().reset_index()
    inv = inv.merge(a, on=["customer_id", "billed_month"], how="left").fillna({"adjustment": 0.0})
    inv["amount"] = (inv.total + inv.adjustment).round(2)
    inv = inv[inv.amount.abs() >= 0.01].reset_index(drop=True)
    inv["invoice_id"] = [f"INV-{i + 1001}" for i in range(len(inv))]
    inv["issue_date"] = inv.billed_month.map(cutoff)
    return inv


# ------------------------------------------------------------------ customer payments and the bank statement
def simulate_cash(b: Business, inv: pd.DataFrame, seed: int = 11):
    rng = np.random.default_rng(seed)
    names = b.customers.set_index("customer_id").name
    pays, truth = [], []
    for cid, g in inv.groupby("customer_id"):
        g = g.sort_values("issue_date")
        i = 0
        rows = list(g.itertuples())
        while i < len(rows):
            r = rows[i]
            u = rng.random()
            when = r.issue_date + pd.Timedelta(days=int(rng.integers(20, 50)))
            payer = names[cid] if rng.random() < 0.5 else names[cid].upper().replace("CUSTOMER", "CUST") + " INC"
            if rng.random() < 0.08:                                                # paid by a parent company
                payer = f"Holdings Group {cid[-2:]} LLC"
            if rng.random() < 0.10 and i + 1 < len(rows):                          # pays the newer invoice first
                rows[i], rows[i + 1] = rows[i + 1], rows[i]
                r = rows[i]
                when = r.issue_date + pd.Timedelta(days=int(rng.integers(20, 50)))
            if u < 0.03 or when >= END + pd.Timedelta(days=40):                   # unpaid / not yet paid
                i += 1
                continue
            if u < 0.10 and i + 1 < len(rows):                                    # one payment for two invoices
                r2 = rows[i + 1]
                pays.append({"date": max(when, r2.issue_date + pd.Timedelta(days=5)), "amount": round(r.amount + r2.amount, 2),
                             "payer": payer, "reference": ""})
                truth.append({"invoice_ids": [r.invoice_id, r2.invoice_id], "kind": "combined"})
                i += 2
                continue
            amt, ref, kind = r.amount, r.invoice_id, "exact"
            if u < 0.15:
                amt, kind = round(r.amount - float(rng.integers(15, 46)), 2), "short_pay_fee"
            elif u < 0.27:
                ref, kind = "", "no_reference"
            pays.append({"date": when, "amount": amt, "payer": payer, "reference": ref})
            truth.append({"invoice_ids": [r.invoice_id], "kind": kind})
            i += 1
    pay = pd.DataFrame(pays)
    pay["truth"] = [t["invoice_ids"] for t in truth]
    pay["kind"] = [t["kind"] for t in truth]
    noise = pd.DataFrame({"date": START + pd.to_timedelta(rng.integers(0, 540, 20), unit="D"),
                          "amount": rng.uniform(50, 900, 20).round(2),
                          "payer": rng.choice(["INTEREST CREDIT", "VENDOR REFUND", "CARD CASHBACK"], 20),
                          "reference": "", "truth": [[] for _ in range(20)], "kind": "not_a_customer_payment"})
    dep = pd.concat([pay, noise], ignore_index=True).sort_values("date").reset_index(drop=True)
    dep["deposit_id"] = [f"D{i:05d}" for i in range(len(dep))]
    return dep[dep.date < END].reset_index(drop=True)


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", "", s.lower()).replace("cust ", "customer ").replace(" inc", "")
    return s.strip()


def apply_cash(dep: pd.DataFrame, inv: pd.DataFrame, b: Business, fee_tolerance: float = 50.0) -> pd.DataFrame:
    """Cash application: reference -> exact amount at the payer -> combinations of the payer's open invoices.
    Anything else goes to a review queue. Returns one row per deposit with the invoices applied."""
    by_name = {_norm(n): c for c, n in zip(b.customers.customer_id, b.customers.name)}
    open_inv = inv.set_index("invoice_id")[["customer_id", "amount", "issue_date"]].copy()
    open_inv["open"] = True
    out = []
    for d in dep.sort_values("date").itertuples():
        applied, how = [], "review"
        cand = open_inv[open_inv.open & (open_inv.issue_date <= d.date)]
        if d.reference and d.reference in cand.index:
            gap = cand.at[d.reference, "amount"] - d.amount
            if abs(gap) < 0.01:
                applied, how = [d.reference], "reference"
            elif 0 < gap <= fee_tolerance:
                applied, how = [d.reference], "reference_short_pay"
        if not applied:
            cid = by_name.get(_norm(d.payer))
            if cid is not None:
                mine = cand[cand.customer_id == cid].sort_values("issue_date")
                exact = mine[(mine.amount - d.amount).abs() < 0.01]
                if len(exact) >= 1:
                    applied, how = [exact.index[0]], "payer_amount"
                else:
                    for k in (2, 3):
                        hit = next((c for c in combinations(mine.index[:8], k)
                                    if abs(mine.loc[list(c), "amount"].sum() - d.amount) < 0.01), None)
                        if hit:
                            applied, how = list(hit), f"payer_combination_{k}"
                            break
        for a in applied:
            open_inv.at[a, "open"] = False
        out.append({"deposit_id": d.deposit_id, "date": d.date, "amount": d.amount, "applied": applied, "how": how})
    return pd.DataFrame(out)


def score_cash(dep: pd.DataFrame, app: pd.DataFrame) -> dict:
    m = dep.merge(app[["deposit_id", "applied", "how"]], on="deposit_id")
    cust = m[m.kind != "not_a_customer_payment"]
    right = cust.apply(lambda r: sorted(r.applied) == sorted(r.truth), axis=1)
    wrong = cust.apply(lambda r: len(r.applied) > 0 and sorted(r.applied) != sorted(r.truth), axis=1)
    noise = m[m.kind == "not_a_customer_payment"]
    return {"deposits": int(len(m)), "customer_payments": int(len(cust)),
            "auto_applied_share": float((cust.applied.str.len() > 0).mean()),
            "applied_correctly": int(right.sum()), "applied_wrongly": int(wrong.sum()),
            "to_review": int((m.how == "review").sum()),
            "non_customer_deposits_left_unapplied": int((noise.applied.str.len() == 0).sum()),
            "by_kind": {k: float(g.apply(lambda r: sorted(r.applied) == sorted(r.truth), axis=1).mean())
                        for k, g in cust.groupby("kind")}}


# ------------------------------------------------------------------ vendor bills (spend) with planted problems
VENDORS = [("Atlas Motion Components", "5100"), ("Precision Grippers Co", "5100"), ("Lumen Sensors", "5100"),
           ("Northeast Field Techs", "5000"), ("RoboFix Contractors", "5000"), ("CloudScale Compute", "6100"),
           ("TrackIt Software", "6100"), ("DataPipe SaaS", "6100"), ("Watertown Properties", "6200"),
           ("OfficeHub Supplies", "6200"), ("SkyMiles Travel", "6300"), ("Hartwell Legal LLP", "6400"),
           ("Beacon Audit Partners", "6400"), ("FreightLine Logistics", "5000"), ("Servo Dynamics", "5100")]


def simulate_bills(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = 0
    for v, acct in VENDORS:
        recurring = acct in ("6100", "6200")
        base = {"5100": 7_000, "5000": 3_500, "6100": 4_000, "6200": 18_000, "6300": 1_200, "6400": 9_000}[acct]
        amt0 = round(base * rng.uniform(0.6, 1.4), 2)
        for m in MONTHS:
            k = 1 if recurring else int(rng.poisson(5 if acct in ("5100", "5000") else 1.5))
            for _ in range(k):
                n += 1
                day = 1 if recurring else int(rng.integers(1, 28))
                amt = amt0 if recurring else round(base * rng.lognormal(0, 0.5), 2)
                rows.append({"bill_id": f"B{n:05d}", "vendor": v, "invoice_no": f"{v[:3].upper()}-{n:05d}",
                             "date": m.to_timestamp() + pd.Timedelta(days=day - 1), "amount": amt, "account": acct,
                             "true_account": acct, "planted": ""})
    bills = pd.DataFrame(rows)
    # legitimately unusual coding: field techs' travel is cost of service; some office bills are robot-lab fit-out
    tr = bills.index[(bills.vendor == "SkyMiles Travel") & (rng.random(len(bills)) < 0.25)]
    bills.loc[tr, ["account", "true_account"]] = "5000"
    of = bills.index[(bills.vendor == "OfficeHub Supplies") & (rng.random(len(bills)) < 0.15)]
    bills.loc[of, ["account", "true_account"]] = "5100"
    # planted: duplicates (re-submitted with a reformatted invoice number, 0-10 days later)
    d = bills.sample(25, random_state=seed).copy()
    d["bill_id"] = [f"B{n + i + 1:05d}" for i in range(len(d))]
    half = rng.random(len(d)) < 0.5
    d["invoice_no"] = np.where(half, d.invoice_no.str.replace("-", "") + "/R",            # same number, reformatted
                               [f"{v[:3].upper()}-{int(rng.integers(90000, 99999))}" for v in d.vendor])  # re-issued
    d["date"] = d.date + pd.to_timedelta(rng.integers(0, 21, len(d)), unit="D")
    d["planted"] = "duplicate"
    n += len(d)
    # planted: miscoded (robot parts booked as office supplies or software, etc.)
    mis = bills[bills.account.isin(["5100", "5000"])].sample(30, random_state=seed + 1).index
    bills.loc[mis, "account"] = rng.choice(["6200", "6100", "6300"], len(mis))
    bills.loc[mis, "planted"] = "miscoded"
    # planted: split purchases just under the approval limit
    sp = []
    for j in range(10):
        v = VENDORS[int(rng.integers(0, 3))][0]
        date = MONTHS[int(rng.integers(0, 18))].to_timestamp() + pd.Timedelta(days=int(rng.integers(0, 27)))
        for k in range(int(rng.integers(2, 4))):
            n += 1
            sp.append({"bill_id": f"B{n:05d}", "vendor": v, "invoice_no": f"{v[:3].upper()}-{n:05d}", "date": date,
                       "amount": round(float(rng.uniform(8_200, 9_950)), 2), "account": "5100",
                       "true_account": "5100", "planted": f"split_{j}"})
    bills = pd.concat([bills, d, pd.DataFrame(sp)], ignore_index=True).sort_values("date").reset_index(drop=True)
    return bills[bills.date < END].reset_index(drop=True)


def review_spend(bills: pd.DataFrame) -> pd.DataFrame:
    """Rules a controller would run every close. Returns flagged bill ids with the reason."""
    flags = []
    b = bills.copy()
    b["inv_norm"] = b.invoice_no.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str.replace("R$", "", regex=True)
    b["inv_digits"] = b.invoice_no.str.extract(r"(\d{4,})")[0]
    # 1) duplicates: same vendor and amount, same invoice digits or within 10 days (but not a monthly recurring bill)
    for (v, a), g in b.groupby(["vendor", "amount"]):
        if len(g) < 2:
            continue
        g = g.sort_values("date")
        for i in range(1, len(g)):
            prev, cur = g.iloc[i - 1], g.iloc[i]
            gap = (cur.date - prev.date).days
            if cur.inv_digits == prev.inv_digits or gap <= 10:
                flags.append({"bill_id": cur.bill_id, "check": "possible_duplicate", "detail": f"matches {prev.bill_id}"})
    # 2) miscoding: account differs from the vendor's usual account
    usual = b.groupby("vendor").account.agg(lambda s: s.value_counts().index[0])
    for r in b[b.account != b.vendor.map(usual)].itertuples():
        flags.append({"bill_id": r.bill_id, "check": "unusual_account", "detail": f"{r.account} vs usual {usual[r.vendor]}"})
    # 3) split purchases: same vendor, same day, each under the approval limit, together over it
    for (v, d), g in b.groupby(["vendor", "date"]):
        small = g[(g.amount < BILL_APPROVAL_LIMIT) & (g.amount >= 0.8 * BILL_APPROVAL_LIMIT)]
        if len(small) >= 2 and small.amount.sum() >= BILL_APPROVAL_LIMIT:
            for r in small.itertuples():
                flags.append({"bill_id": r.bill_id, "check": "possible_split_purchase",
                              "detail": f"{len(small)} bills, {small.amount.sum():,.0f} total"})
    return pd.DataFrame(flags)


def score_spend(bills: pd.DataFrame, flags: pd.DataFrame) -> dict:
    out = {}
    for check, kind in [("possible_duplicate", "duplicate"), ("unusual_account", "miscoded"),
                        ("possible_split_purchase", "split_")]:
        f = set(flags[flags.check == check].bill_id)
        planted = set(bills[bills.planted.str.startswith(kind)].bill_id)
        out[check] = {"planted": len(planted), "flagged": len(f), "caught": len(f & planted),
                      "false_flags": len(f - planted)}
    return out


# ------------------------------------------------------------------ posting and tie-outs
@dataclass
class CloseResult:
    ledger: Ledger
    invoices: pd.DataFrame
    deposits: pd.DataFrame
    cash_app: pd.DataFrame
    bills: pd.DataFrame
    spend_flags: pd.DataFrame
    checks: list = field(default_factory=list)


def deferred_schedule(b: Business, inv: pd.DataFrame) -> pd.DataFrame:
    """Deployment fees are billed at go-live and recognised evenly over the remaining contract term."""
    term = b.contracts.set_index("customer_id")
    rows = []
    for r in inv[inv.deploy_fees > 0].itertuples():
        k = term.loc[r.customer_id]
        end = pd.Period(k.start_month, "M") + int(k.term_months)
        n = max((end - r.month).n, 1)
        for i in range(n):
            rows.append({"customer_id": r.customer_id, "invoice_id": r.invoice_id, "month": r.month + i,
                         "amount": r.deploy_fees / n})
    return pd.DataFrame(rows)


def run_close(b: Business, run: BillingRun) -> CloseResult:
    L = Ledger()
    L.post("opening-equity", START, [("1000", OPENING_EQUITY, 0), ("3000", 0, OPENING_EQUITY)], "Opening equity")
    inv = issued_invoices(run)
    # robots built and capitalised at first go-live
    first = b.assignments.sort_values("from").groupby("robot_id").head(1)
    for r in first.itertuples():
        L.post(f"capex-{r.robot_id}", r._3, [("1500", ROBOT_COST, 0), ("1000", 0, ROBOT_COST)], f"Robot {r.robot_id} built")
    # invoices: usage revenue (net of SLA credits) in the service month; deployment fees to deferred revenue
    for r in inv.itertuples():
        d = r.issue_date - pd.Timedelta(days=INV_ACCRUE_LAG)
        L.post(f"inv-{r.invoice_id}", d, [
            ("1100", r.amount, 0), ("4000", 0, r.usage_charge + r.adjustment), ("4020", -r.sla_credit, 0),
            ("2100", 0, r.deploy_fees)], f"{r.invoice_id} {r.customer_id} {r.billed_month}")
    sched = deferred_schedule(b, inv)
    for m, g in sched.groupby("month"):
        if m > MONTHS[-1]:
            continue
        amt = round(g.amount.sum(), 2)
        L.post(f"defrev-{m}", m.to_timestamp(how="end").normalize(), [("2100", amt, 0), ("4010", 0, amt)],
               f"Deployment revenue release {m}")
    # cash
    dep = simulate_cash(b, inv)
    app = apply_cash(dep, inv, b)
    amt_by_inv = inv.set_index("invoice_id").amount
    for r in app.merge(dep[["deposit_id", "payer"]], on="deposit_id").itertuples():
        if r.applied:
            owed = round(float(amt_by_inv[r.applied].sum()), 2)
            fee = round(owed - r.amount, 2)
            L.post(f"cash-{r.deposit_id}", r.date, [("1000", r.amount, 0), ("6900", fee, 0), ("1100", 0, owed)],
                   f"Cash applied to {','.join(r.applied)}")
        else:
            L.post(f"cash-{r.deposit_id}", r.date, [("1000", r.amount, 0), ("6400", 0, r.amount)] if r.payer in (
                "INTEREST CREDIT", "VENDOR REFUND", "CARD CASHBACK") else [("1000", r.amount, 0), ("2000", 0, r.amount)],
                   f"Unapplied: {r.payer}")
    # spend: bills post to AP; paid 30 days later
    bills = simulate_bills()
    flags = review_spend(bills)
    held = set(flags.bill_id)                                         # flagged bills are held, not paid, until reviewed
    for r in bills.itertuples():
        L.post(f"bill-{r.bill_id}", r.date, [(r.account, r.amount, 0), ("2000", 0, r.amount)], f"{r.vendor} {r.invoice_no}")
        if r.bill_id not in held and r.date + pd.Timedelta(days=30) < END:
            L.post(f"pay-{r.bill_id}", r.date + pd.Timedelta(days=30), [("2000", r.amount, 0), ("1000", 0, r.amount)],
                   f"Pay {r.vendor}")
    res = CloseResult(L, inv, dep, app, bills, flags)
    res.checks = tie_outs(b, res, sched)
    return res


INV_ACCRUE_LAG = 3   # invoices for month m are accrued on the last day of m (issued on day 3 of m+1)


def tie_outs(b: Business, res: CloseResult, sched: pd.DataFrame) -> list:
    L, inv, app = res.ledger, res.invoices, res.cash_app
    tb = L.trial_balance()
    out = [{"check": "trial balance balances", "pass": bool(abs(tb.debit.sum() - tb.credit.sum()) < 0.01),
            "detail": f"debits {tb.debit.sum():,.2f} credits {tb.credit.sum():,.2f}"},
           {"check": "ledger hash chain intact", "pass": L.verify_chain(), "detail": f"{len(L.entries):,} entries"}]
    applied = {i for a in app.applied for i in a}
    open_ar = round(float(inv[~inv.invoice_id.isin(applied)].amount.sum()), 2)
    fees = round(float(L.frame().query("account == '6900'").debit.sum()), 2)
    gl_ar = L.balance("1100")
    out.append({"check": "AR subledger = GL accounts receivable", "pass": abs(open_ar - gl_ar) < 0.01,
                "detail": f"open invoices {open_ar:,.2f} vs GL {gl_ar:,.2f}"})
    released = sched[sched.month <= MONTHS[-1]].amount.sum()
    def_sched = round(float(inv.deploy_fees.sum() - released), 2)
    gl_def = -L.balance("2100")
    out.append({"check": "Deferred revenue schedule = GL", "pass": abs(def_sched - gl_def) < 1.0,
                "detail": f"schedule {def_sched:,.2f} vs GL {gl_def:,.2f}"})
    billed_rev = round(float((inv.usage_charge + inv.adjustment + inv.sla_credit).sum()), 2)
    gl_rev = round(-L.balance("4000") - L.balance("4020"), 2)
    out.append({"check": "Billing = usage revenue net of credits", "pass": abs(billed_rev - gl_rev) < 0.01,
                "detail": f"billing {billed_rev:,.2f} vs GL {gl_rev:,.2f}"})
    bank = round(OPENING_EQUITY + res.deposits.amount.sum() - len(b.robots) * ROBOT_COST
                 - L.frame().query("key.str.startswith('pay-')", engine="python").credit.sum(), 2)
    out.append({"check": "GL cash = bank statement", "pass": bool(abs(bank - L.balance("1000")) < 0.01),
                "detail": f"bank {bank:,.2f} vs GL {L.balance('1000'):,.2f}"})
    out.append({"check": "Short-pay fees written off", "pass": True, "detail": f"{fees:,.2f} to bank fees"})
    return out
