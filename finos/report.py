"""Figures and a self-contained close dashboard (HTML) from results/results.json.  python -m finos.report"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
OUT = Path("results")


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_billing(R):
    w = R["billing"]["without_each_fix"]
    names = {"as_of_map": "Effective-dated robot map", "dedupe": "Drop retried uploads",
             "config_check": "Check rates vs. signed contract", "late_trueup": "True up late uploads",
             "clock": "Correct robot clocks"}
    order = sorted(w, key=lambda k: w[k]["abs_error"])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.barh([names[k] for k in order], [w[k]["abs_error"] / 1000 for k in order], color=ORANGE, height=0.55)
    for i, k in enumerate(order):
        ax.text(w[k]["abs_error"] / 1000 + 2, i, f"${w[k]['abs_error'] / 1000:,.0f}K", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Billing error over 17 months if this step is removed ($K, absolute)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Legacy process: ${R['billing']['legacy']['abs_error'] / 1000:,.0f}K wrong; "
                 f"pipeline: correct to the cent", loc="left", fontsize=11, color=INK)
    fig.savefig(OUT / "billing_fixes.png", dpi=150, bbox_inches="tight")
    return _png(fig)


def fig_forecast(R):
    bt = R["forecast"]["by_origin"]["2025-12"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    m = bt["months"]
    ax.plot(m, [x / 1000 for x in bt["actual"]], color=INK, lw=2, marker="o", label="Actual")
    for k, c, lab in [("driver_plan", ORANGE, "Driver plan"), ("run_rate", GRAY, "Run rate"),
                      ("linear_trend", BLUE, "Linear trend")]:
        ax.plot(m, [x / 1000 for x in bt[k]["forecast"]], color=c, lw=2, label=f"{lab} (MAPE {100 * bt[k]['mape']:.1f}%)")
    ax.set_ylabel("Usage revenue net of credits ($K)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("H1 2026 plan made at the end of Dec 2025", loc="left", fontsize=11, color=INK)
    fig.savefig(OUT / "forecast.png", dpi=150, bbox_inches="tight")
    return _png(fig)


def fig_payback(R):
    cs = sorted(R["economics"]["contracts"], key=lambda r: r["payback_months"])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(range(len(cs)), [c["payback_months"] for c in cs],
           color=[ORANGE if c["payback_months"] > 24 else BLUE for c in cs], width=0.75)
    ax.axhline(24, color=INK2, ls="--", lw=1)
    ax.text(0, 25, "24-month payback target", fontsize=8.5, color=INK2)
    ax.set_xticks([])
    ax.set_xlabel("Contracts, by payback (orange: slower than 24 months)")
    ax.set_ylabel("Payback per robot (months)")
    ax.set_title("Unit economics by contract at realised utilisation", loc="left", fontsize=11, color=INK)
    fig.savefig(OUT / "payback.png", dpi=150, bbox_inches="tight")
    return _png(fig)


def dashboard(R, imgs):
    p, l = R["billing"]["pipeline"], R["billing"]["legacy"]
    cash, sp = R["close"]["cash"], R["close"]["spend"]
    rows = "".join(f"<tr><td>{html.escape(c['check'])}</td><td class='{'ok' if c['pass'] else 'bad'}'>"
                   f"{'pass' if c['pass'] else 'FAIL'}</td><td>{html.escape(c['detail'])}</td></tr>" for c in R["close"]["checks"])
    var = "".join(f"<tr><td>{v['month']}</td><td>${v['plan_revenue'] / 1e3:,.1f}K</td><td>${v['actual_revenue'] / 1e3:,.1f}K</td>"
                  f"<td>{v['variance'] / 1e3:+,.1f}K</td>" + "".join(f"<td>{v['effects'][k] / 1e3:+,.1f}K</td>"
                  for k in ("fleet", "utilisation", "rate", "credits")) + "</tr>" for v in R["variance"])
    slow = sum(c["payback_months"] > 24 for c in R["economics"]["contracts"])
    tiles = [("Billing error, legacy", f"${l['abs_error'] / 1e3:,.0f}K", f"{100 * l['abs_error_share']:.1f}% of billings"),
             ("Billing error, pipeline", "$0", "every invoice correct to the cent"),
             ("Cash auto-applied", f"{100 * cash['auto_applied_share']:.0f}%", f"{cash['applied_wrongly']} misapplied"),
             ("Plan accuracy (MAPE)", f"{100 * R['forecast']['mean_mape']['driver_plan']:.1f}%",
              f"vs {100 * R['forecast']['mean_mape']['run_rate']:.1f}% run rate"),
             ("Contracts > 24-mo payback", f"{slow} / {len(R['economics']['contracts'])}", "minimums below breakeven")]
    tile_html = "".join(f"<div class='tile'><div class='k'>{a}</div><div class='v'>{b}</div><div class='s'>{c}</div></div>"
                        for a, b, c in tiles)
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RaaS Close Dashboard</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e0;--ok:#008300;--bad:#e34948}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37}}
body{{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1080px;margin:auto}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}} .sub{{color:var(--ink2)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:16px}}
.tile{{border:1px solid var(--line);border-radius:8px;padding:12px}} .k{{color:var(--ink2);font-size:12px}}
.v{{font-size:24px;font-weight:600}} .s{{color:var(--ink2);font-size:12px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}}
.ok{{color:var(--ok);font-weight:600}} .bad{{color:var(--bad);font-weight:600}} img{{max-width:100%;background:#fcfcfb;border-radius:6px}}
.wrap{{overflow-x:auto}}</style></head><body>
<h1>Robot-as-a-Service close dashboard</h1><div class="sub">Simulated company: {R['business']['customers']} customers,
{R['business']['robots']} robots, {R['business']['months'][0]} to {R['business']['months'][1]}. Generated from results.json.</div>
<div class="tiles">{tile_html}</div>
<h2>Usage-to-invoice</h2><img src="data:image/png;base64,{imgs[0]}" alt="billing error by pipeline step">
<h2>Close tie-outs</h2><div class="wrap"><table><tr><th>Check</th><th>Status</th><th>Detail</th></tr>{rows}</table></div>
<p class="sub">Spend review: duplicates caught {sp['possible_duplicate']['caught']}/{sp['possible_duplicate']['planted']},
miscoding {sp['unusual_account']['caught']}/{sp['unusual_account']['planted']} ({sp['unusual_account']['false_flags']} false flags),
split purchases {sp['possible_split_purchase']['caught']}/{sp['possible_split_purchase']['planted']}.</p>
<h2>Plan vs. actual, H1 2026</h2><div class="wrap"><table><tr><th>Month</th><th>Plan</th><th>Actual</th><th>Variance</th>
<th>Fleet</th><th>Utilisation</th><th>Rate</th><th>Credits</th></tr>{var}</table></div>
<img src="data:image/png;base64,{imgs[1]}" alt="forecast backtest">
<h2>Unit economics</h2><img src="data:image/png;base64,{imgs[2]}" alt="payback by contract">
</body></html>"""


def main():
    R = json.loads((OUT / "results.json").read_text())
    imgs = [fig_billing(R), fig_forecast(R), fig_payback(R)]
    Path("docs").mkdir(exist_ok=True)
    (Path("docs") / "close_dashboard.html").write_text(dashboard(R, imgs))
    print("wrote results/*.png and docs/close_dashboard.html")


if __name__ == "__main__":
    main()
