"""The finance data layer: one DuckDB database joining CRM, contracts, robot operations, billing, the ledger,
spend and cash, with views finance actually asks for."""
from __future__ import annotations

import duckdb
import pandas as pd

VIEWS = {
    "v_revenue_by_month": """
        SELECT month, SUM(usage_charge) AS usage, SUM(sla_credit) AS credits, SUM(deploy_fees) AS deploy_fees,
               SUM(total) AS billed, SUM(hours) AS hours
        FROM invoices GROUP BY month ORDER BY month""",
    "v_customer_unit_economics": """
        WITH rm AS (SELECT customer_id, month, COUNT(DISTINCT robot_id) AS robots, SUM(hours) AS hours
                    FROM robot_months GROUP BY customer_id, month
                    HAVING MIN(age) >= 3 AND MIN(days) >= 28),   -- months when every robot there is mature
             inv AS (SELECT customer_id, month, usage_charge + sla_credit AS revenue FROM invoices)
        SELECT c.customer_id, c.segment, k.rate_per_hour, k.min_hours / k.robots AS min_hours_per_robot,
               AVG(rm.hours / rm.robots) AS hours_per_robot_month, AVG(inv.revenue / rm.robots) AS revenue_per_robot_month
        FROM rm JOIN inv USING (customer_id, month) JOIN customers c USING (customer_id)
             JOIN contracts k USING (customer_id)
        GROUP BY ALL ORDER BY revenue_per_robot_month""",
    "v_ar_aging": """
        SELECT i.customer_id, i.invoice_id, i.amount, i.issue_date,
               date_diff('day', i.issue_date, TIMESTAMP '2026-07-01') AS days_outstanding
        FROM issued_invoices i LEFT JOIN applied a ON a.invoice_id = i.invoice_id
        WHERE a.invoice_id IS NULL ORDER BY days_outstanding DESC""",
    "v_trial_balance": """
        SELECT account, ROUND(SUM(debit) - SUM(credit), 2) AS balance FROM gl GROUP BY account ORDER BY account""",
}


def build(path: str, b, run, close, robot_months: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path)
    inv = run.invoices.assign(month=run.invoices.month.astype(str))
    rm = robot_months.assign(month=robot_months.month.astype(str))
    applied = pd.DataFrame({"invoice_id": [i for a in close.cash_app.applied for i in a]})
    tables = {"customers": b.customers, "contracts": b.contracts, "robots": b.robots, "assignments": b.assignments,
              "invoices": inv, "issued_invoices": close.invoices.assign(month=close.invoices.month.astype(str),
                                                                          billed_month=close.invoices.billed_month.astype(str)),
              "applied": applied, "gl": close.ledger.frame(), "bills": close.bills, "robot_months": rm,
              "deposits": close.deposits.drop(columns=["truth"])}
    for name, df in tables.items():
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    for name, sql in VIEWS.items():
        con.execute(f"CREATE OR REPLACE VIEW {name} AS {sql}")
    return con
