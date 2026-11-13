"""MarketMind: does FinBERT news sentiment predict next-day DJIA returns?

Data (Kaggle "Daily News for Stock Market Prediction"):
  Combined_News_DJIA.csv   -> Date, Label, Top1..Top25 (daily headlines)
  upload_DJIA_table.csv    -> Date, Open, High, Low, Close, Volume, Adj Close

Design choices (to avoid look-ahead bias):
  * Sentiment on date t is tested against the return from t's close to the NEXT
    trading day's close (never the same-day return).
  * The trading threshold is fit on the first 70% of days only; all strategy
    metrics are computed on the final 30% (chronological split, no shuffling).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def _clean(s):
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s[:2] in ("b'", 'b"') and len(s) > 3:
        s = s[2:-1]
    return s or None


def load_headlines(news_csv, per_day=25):
    df = pd.read_csv(news_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    tops = [c for c in df.columns if c.startswith("Top")][:per_day]
    long = df.melt(id_vars="Date", value_vars=tops, value_name="headline")
    long["headline"] = long["headline"].map(_clean)
    long = long.dropna(subset=["headline"])
    return long[["Date", "headline"]].sort_values("Date").reset_index(drop=True)


def load_prices(price_csv):
    px = pd.read_csv(price_csv)
    px["Date"] = pd.to_datetime(px["Date"])
    return px.sort_values("Date").reset_index(drop=True)


# ----------------------------------------------------------------------------
# FinBERT scoring (score = P(positive) - P(negative), range [-1, 1])
# ----------------------------------------------------------------------------
def score_headlines(long, cache_path, batch_size=64):
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, parse_dates=["Date"])
        if len(cached) == len(long):
            print(f"Loaded cached scores: {cache_path}")
            return cached
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    print(f"Scoring {len(long):,} headlines with ProsusAI/finbert (device={device})")
    clf = pipeline(
        "text-classification", model="ProsusAI/finbert", top_k=None,
        device=device, truncation=True, max_length=64,
    )
    texts = long["headline"].tolist()
    scores = []
    chunk = 2048
    for i in range(0, len(texts), chunk):
        out = clf(texts[i:i + chunk], batch_size=batch_size)
        for probs in out:
            d = {p["label"].lower(): p["score"] for p in probs}
            scores.append(d.get("positive", 0.0) - d.get("negative", 0.0))
        print(f"  {min(i + chunk, len(texts)):,}/{len(texts):,}")
    long = long.copy()
    long["score"] = scores
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    long.to_csv(cache_path, index=False)
    return long


# ----------------------------------------------------------------------------
# Event study
# ----------------------------------------------------------------------------
def sharpe(r):
    r = np.asarray(r, dtype=float)
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else float("nan")


def evaluate(scored, prices, split=0.7, n_perm=10000, seed=0):
    daily = scored.groupby("Date")["score"].mean().rename("sentiment").reset_index()

    px = prices[["Date", "Adj Close"]].copy()
    px["ret_next"] = px["Adj Close"].shift(-1) / px["Adj Close"] - 1  # t close -> t+1 close
    px["ret_prev"] = px["Adj Close"] / px["Adj Close"].shift(1) - 1   # diagnostic only

    df = daily.merge(px, on="Date", how="inner").dropna(subset=["ret_next", "ret_prev"])
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)

    # Full-sample rank IC + permutation test
    ic, p_sp = stats.spearmanr(df["sentiment"], df["ret_next"])
    rs, rr = stats.rankdata(df["sentiment"]), stats.rankdata(df["ret_next"])
    obs = np.corrcoef(rs, rr)[0, 1]
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(rng.permutation(rs), rr)[0, 1] for _ in range(n_perm)])
    p_perm = float((np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1))

    # Diagnostic: is sentiment just reacting to yesterday's market move?
    ic_prev, p_prev = stats.spearmanr(df["sentiment"], df["ret_prev"])

    # Chronological split
    k = int(n * split)
    train, test = df.iloc[:k], df.iloc[k:].copy()
    thr = float(train["sentiment"].median())
    test["pos"] = np.where(test["sentiment"] > thr, 1, -1)
    test["strat"] = test["pos"] * test["ret_next"]
    ic_test, p_test = stats.spearmanr(test["sentiment"], test["ret_next"])

    acc = float((np.sign(test["pos"]) == np.sign(test["ret_next"])).mean())
    base = float((test["ret_next"] > 0).mean())

    res = {
        "n_days": int(n),
        "date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "n_headlines": int(len(scored)),
        "rank_ic_full": round(float(ic), 4),
        "p_value_spearman": round(float(p_sp), 4),
        "p_value_permutation": round(p_perm, 4),
        "rank_ic_vs_prev_day_return": round(float(ic_prev), 4),
        "p_value_vs_prev_day_return": round(float(p_prev), 4),
        "test_n_days": int(len(test)),
        "test_date_range": [str(test["Date"].min().date()), str(test["Date"].max().date())],
        "rank_ic_test": round(float(ic_test), 4),
        "p_value_test": round(float(p_test), 4),
        "threshold_from_train_median": round(thr, 4),
        "test_directional_accuracy": round(acc, 4),
        "test_up_day_base_rate": round(base, 4),
        "test_sharpe_strategy": round(sharpe(test["strat"]), 3),
        "test_sharpe_buy_and_hold": round(sharpe(test["ret_next"]), 3),
        "test_total_return_strategy": round(float((1 + test["strat"]).prod() - 1), 4),
        "test_total_return_buy_and_hold": round(float((1 + test["ret_next"]).prod() - 1), 4),
        "notes": "No transaction costs. Index-level, single asset. Permutation test ignores autocorrelation.",
    }
    return res, test


def plot(test, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(test["Date"], (1 + test["strat"]).cumprod(), label="Sentiment long/short")
    ax.plot(test["Date"], (1 + test["ret_next"]).cumprod(), label="Buy & hold DJIA")
    ax.set_title("Out-of-sample (chronological hold-out), no costs")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="data/Combined_News_DJIA.csv")
    ap.add_argument("--prices", default="data/upload_DJIA_table.csv")
    ap.add_argument("--per_day", type=int, default=25, help="headlines per day to score (Top1..TopN)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    long = load_headlines(args.news, args.per_day)
    scored = score_headlines(long, os.path.join("data", f"headline_scores_{args.per_day}.csv"))
    res, test = evaluate(scored, load_prices(args.prices))

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    plot(test, os.path.join(args.out, "cumulative_returns.png"))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
