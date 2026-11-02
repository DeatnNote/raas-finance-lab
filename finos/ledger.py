"""An append-only double-entry ledger with the controls auditors and lenders ask for.

  - every journal must balance (debits = credits) or it is rejected
  - every journal carries an idempotency key: posting the same key twice returns the original, never a duplicate
  - entries are hash-chained: editing any past entry breaks verification
  - manual journals at or above the approval limit need an approver who is not the preparer (maker-checker)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import pandas as pd

ACCOUNTS = {
    "1000": "Cash", "1100": "Accounts receivable", "1500": "Robot fleet (at cost)", "2000": "Accounts payable",
    "2100": "Deferred revenue", "3000": "Equity", "4000": "Usage revenue", "4010": "Deployment revenue",
    "4020": "SLA credits", "5000": "Field service (COGS)", "5100": "Robot parts & components",
    "6100": "Cloud & software", "6200": "Office & facilities", "6300": "Travel", "6400": "Professional services",
    "6900": "Bank fees",
}
APPROVAL_LIMIT = 25_000.0


class LedgerError(Exception):
    pass


@dataclass
class Ledger:
    entries: list = field(default_factory=list)
    keys: dict = field(default_factory=dict)
    _last_hash: str = "GENESIS"

    def post(self, key: str, date, lines: list, memo: str = "", source: str = "system",
             preparer: str = "system", approver: str | None = None) -> str:
        """lines: [(account, debit, credit), ...]. Returns the journal id."""
        if key in self.keys:
            return self.keys[key]                                    # idempotent
        dr = round(sum(l[1] for l in lines), 2)
        cr = round(sum(l[2] for l in lines), 2)
        if abs(dr - cr) > 0.005:
            raise LedgerError(f"{key}: unbalanced journal (debits {dr} vs credits {cr})")
        for a, _, _ in lines:
            if a not in ACCOUNTS:
                raise LedgerError(f"{key}: unknown account {a}")
        if source == "manual" and dr >= APPROVAL_LIMIT and (approver is None or approver == preparer):
            raise LedgerError(f"{key}: manual journal of {dr:,.2f} needs an independent approver")
        jid = f"J{len(self.keys) + 1:07d}"
        for a, d, c in lines:
            if abs(d) < 0.005 and abs(c) < 0.005:
                continue
            rec = {"journal_id": jid, "key": key, "date": str(pd.Timestamp(date).date()), "account": a,
                   "debit": round(d, 2), "credit": round(c, 2), "memo": memo, "source": source,
                   "preparer": preparer, "approver": approver, "prev_hash": self._last_hash}
            rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
            self._last_hash = rec["hash"]
            self.entries.append(rec)
        self.keys[key] = jid
        return jid

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.entries)
        df["date"] = pd.to_datetime(df.date)
        return df

    def balance(self, account: str, as_of=None) -> float:
        df = self.frame()
        if as_of is not None:
            df = df[df.date <= pd.Timestamp(as_of)]
        df = df[df.account == account]
        return round(float(df.debit.sum() - df.credit.sum()), 2)

    def trial_balance(self, as_of=None) -> pd.DataFrame:
        df = self.frame()
        if as_of is not None:
            df = df[df.date <= pd.Timestamp(as_of)]
        tb = df.groupby("account")[["debit", "credit"]].sum().round(2)
        tb["name"] = tb.index.map(ACCOUNTS)
        return tb

    def verify_chain(self) -> bool:
        prev = "GENESIS"
        for e in self.entries:
            rec = {k: v for k, v in e.items() if k != "hash"}
            if rec["prev_hash"] != prev:
                return False
            if hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest() != e["hash"]:
                return False
            prev = e["hash"]
        return True
