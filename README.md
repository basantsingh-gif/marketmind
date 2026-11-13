# MarketMind: does FinBERT news sentiment predict next-day DJIA returns?

FinBERT sentiment on daily Reddit WorldNews headlines, tested against next-day DJIA returns
with a leak-free event study and robustness checks. Honest result: a weak, regime-dependent signal.

## Method
1. Score the top-10 headlines per day with `ProsusAI/finbert` (score = P(positive) - P(negative)).
2. Daily sentiment = mean headline score.
3. Align sentiment on day t with the **next** trading day's return (t close to t+1 close).
4. Spearman rank IC and a permutation test; chronological 70/30 split; threshold = train median sentiment;
   long/short on the hold-out vs buy-and-hold.
5. `robustness.py`: placebo horizons, IC by year, transaction costs, permutation test on Sharpe, stale-signal test.

Data: Kaggle "Daily News for Stock Market Prediction" (`Combined_News_DJIA.csv`, `upload_DJIA_table.csv`).

## Results
19,890 headlines, 1,987 trading days (2008-08-11 to 2016-06-30). Hold-out: 597 days (2014-02-19 to 2016-06-30).

| Metric | Value |
|---|---|
| Rank IC, full sample | 0.047 (p = 0.035; permutation p = 0.035) |
| Rank IC, hold-out | 0.138 (p = 0.0007) |
| Hold-out Sharpe, strategy (gross, no costs) | 1.87 |
| Hold-out Sharpe, strategy (net, 5 bps per unit turnover) | 0.97 |
| Hold-out Sharpe, buy-and-hold | 0.41 |
| Permutation p-value for hold-out Sharpe | 0.004 |
| Hold-out total return, strategy (gross) vs buy-and-hold | 79.8% vs 11.9% |
| Directional accuracy vs up-day base rate | 53.1% vs 53.1% |
| Mean daily return on long days vs short days | +0.119% vs -0.084% |
| Sharpe using yesterday's sentiment (stale signal) | -0.56 |

## Robustness
| Check | Result |
|---|---|
| IC vs same-day return | +0.036 (p = 0.108) |
| IC at t+1 to t+2 / t+2 to t+3 / t+3 to t+4 | -0.025 / +0.037 / -0.009 (all p > 0.10) |
| IC by year | 2008-10: about 0 or negative; 2011-14: +0.03 to +0.10 (not significant); 2015: +0.14 (p = 0.02); 2016: +0.23 (p = 0.01, n = 125) |

## Interpretation
Sentiment shows a statistically significant but small next-day relationship in the full sample,
concentrated in 2015-2016. It does not hold across 2008-2013, and it halves after a 5 bps cost
assumption. The result is promising but unvalidated, not a tradable strategy.

## Limitations
- Reddit WorldNews is general news, not a financial newswire; headline selection by upvotes may include hindsight.
- Single index, daily frequency, 2008-2016, one regime split. No slippage or market impact.
- I ran several tests (5 horizons, 9 years, multiple metrics); isolated p < 0.05 results can occur by chance.
- Permutation tests ignore autocorrelation. Used top-10 headlines/day (`--per_day 10`).
- The training half contains almost no signal, so the hold-out result rests on one favourable period.

## Run
```bash
pip install -r requirements.txt
python marketmind.py --per_day 10
python robustness.py --per_day 10
```
`commentary.py` is an experimental LLM comment generator and was not evaluated.
