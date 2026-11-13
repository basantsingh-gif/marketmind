"""Robustness checks for the MarketMind event study (run AFTER marketmind.py).

Uses the cached FinBERT scores, so it takes seconds. Prints:
  1. IC by horizon (placebo: a real news signal should fade at +2/+3 days)
  2. IC by calendar year (is the signal stable or concentrated in one period?)
  3. Hold-out strategy: position balance, long/short day returns, costs,
     permutation test on Sharpe, and a skip-a-day (stale signal) version.

Usage: python robustness.py --per_day 10
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

import marketmind as mm


def sharpe(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_day", type=int, default=10)
    ap.add_argument("--prices", default="data/upload_DJIA_table.csv")
    a = ap.parse_args()

    sc = pd.read_csv(f"data/headline_scores_{a.per_day}.csv", parse_dates=["Date"])
    daily = sc.groupby("Date")["score"].mean().rename("sent").reset_index()

    p = mm.load_prices(a.prices)[["Date", "Adj Close"]].copy()
    p["ret1"] = p["Adj Close"].pct_change()          # return INTO day t (t-1 close -> t close)
    horizons = {"same_day(t-1->t)": -1, "next_day(t->t+1) [tested]": 0, "t+1->t+2": 1, "t+2->t+3": 2, "t+3->t+4": 3}
    for name, k in horizons.items():
        p[name] = p["ret1"].shift(-(k + 1))
    df = daily.merge(p, on="Date", how="inner").sort_values("Date").reset_index(drop=True)

    out = {}
    print("\n1) Rank IC by horizon (sentiment on day t vs return window)")
    out["ic_by_horizon"] = {}
    for name in horizons:
        d = df.dropna(subset=[name])
        ic, pv = stats.spearmanr(d["sent"], d[name])
        out["ic_by_horizon"][name] = {"ic": round(float(ic), 4), "p": round(float(pv), 4), "n": int(len(d))}
        print(f"   {name:28s} IC={ic:+.4f}  p={pv:.4f}  n={len(d)}")

    tcol = "next_day(t->t+1) [tested]"
    d0 = df.dropna(subset=[tcol]).reset_index(drop=True)
    d0["r0"] = d0[tcol]

    print("\n2) Rank IC by calendar year (next-day return)")
    out["ic_by_year"] = {}
    for y, g in d0.groupby(d0["Date"].dt.year):
        if len(g) > 30:
            ic, pv = stats.spearmanr(g["sent"], g["r0"])
            out["ic_by_year"][int(y)] = {"ic": round(float(ic), 4), "p": round(float(pv), 4), "n": int(len(g))}
            print(f"   {y}: IC={ic:+.4f}  p={pv:.4f}  n={len(g)}")

    print("\n3) Hold-out strategy (same 70/30 chronological split as marketmind.py)")
    k = int(len(d0) * 0.7)
    tr, te = d0.iloc[:k], d0.iloc[k:].copy()
    thr = float(tr["sent"].median())
    te["pos"] = np.where(te["sent"] > thr, 1, -1)
    te["strat"] = te["pos"] * te["r0"]
    share_long = float((te["pos"] == 1).mean())
    long_ret = float(te.loc[te["pos"] == 1, "r0"].mean())
    short_ret = float(te.loc[te["pos"] == -1, "r0"].mean())
    cost = 0.0005 * te["pos"].diff().abs().fillna(0)  # 5 bps per unit of turnover
    sh_g, sh_n = sharpe(te["strat"]), sharpe(te["strat"] - cost)
    turnover = float(te["pos"].diff().abs().fillna(0).mean() / 2)

    rng = np.random.default_rng(0)
    sims = np.array([sharpe(np.where(rng.permutation(te["sent"].values) > thr, 1, -1) * te["r0"].values)
                     for _ in range(2000)])
    p_sharpe = float((np.sum(sims >= sh_g) + 1) / (len(sims) + 1))

    stale_pos = np.where(d0["sent"].shift(1).iloc[k:].values > thr, 1, -1)
    sh_stale = sharpe(stale_pos * te["r0"].values)

    out["holdout"] = {
        "n_days": int(len(te)), "share_days_long": round(share_long, 3),
        "mean_daily_ret_when_long_pct": round(100 * long_ret, 4),
        "mean_daily_ret_when_short_pct": round(100 * short_ret, 4),
        "daily_flip_rate": round(turnover, 3),
        "sharpe_gross": round(sh_g, 3), "sharpe_net_5bps": round(sh_n, 3),
        "perm_p_value_sharpe": round(p_sharpe, 4),
        "sharpe_using_yesterdays_sentiment": round(sh_stale, 3),
    }
    for kk, v in out["holdout"].items():
        print(f"   {kk}: {v}")

    with open("results/robustness.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/robustness.json")


if __name__ == "__main__":
    main()
