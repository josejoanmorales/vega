# Vega — Full Product Strategy

**Repo:** https://github.com/josejoanmorales/vega
**Status:** Strategy locked 2026-07-05. Internal phase not yet started.
**Name:** Vega — the option greek measuring sensitivity to volatility. The product's edge is knowing how exposed every idea is to a changing market, and proving it.

---

## 1. What Vega is

Vega is an evidence-based market analyst assistant for swing trading US equities/ETFs and major crypto. It generates a daily pre-market briefing, ranked trade ideas with full risk specs, and a weekly self-graded review — and it forward-tests every recommendation in a paper account from day one, building an unfakeable, auditable track record.

**Path:** internal tool for Jose first (same pattern as Caral) → SaaS only after the track record earns it.

## 2. Strategic thesis

No system reliably predicts prices. Retail "AI stock pickers" fail because they chase prediction, overfit to history, and have no verifiable record. Vega's accuracy is won in three places where accuracy is actually winnable:

1. **Accuracy of evidence** — every number traceable to a live data source; the LLM never produces a figure from memory.
2. **Accuracy of validation** — no signal influences a recommendation until it survives a rigorous, registered backtest.
3. **Accuracy of the track record** — every call auto-executed in a paper account, graded weekly by the system itself.

This is the Caral thesis applied to markets: **verified vs. reported.** Competitors report how smart they are. Vega proves it with a ledger it cannot fake. When Vega goes SaaS, the graded ledger *is* the marketing — and the compliance story.

## 3. Product definition

- **User (internal phase):** Jose — active investor with a US-accessible broker, swing horizon.
- **Scope v1:** swing trades (5–20 trading days), **long/flat only** (no shorts, no options, no margin). "Flat" is a position: in hostile regimes the correct high-accuracy output is *no trade*.
- **Assets:** US equities & ETFs + major crypto, run as separate sleeves with separate benchmarks — but risk-managed as **one portfolio** (see §5), because crypto and tech equities correlate heavily in risk-off.
- **Form factors, sequenced:** daily pre-market briefing → weekly self-graded review → on-demand analyst chat → real-time alerts. Not built simultaneously.

### Tradable universe (fixed, versioned)

- Equities: S&P 500 + Nasdaq-100 constituents, minimum median dollar-volume filter.
- ETFs: a curated list of ~30 liquid index/sector/commodity ETFs.
- Crypto: top ~20 by market cap, ex-stablecoins.

Rationale: signal quality and free-data reliability collapse outside liquid names, and constraining the universe to current large/mid caps bounds (though does not eliminate) survivorship bias — all performance claims are scoped to this universe.

## 4. What "accuracy" means — the metric hierarchy

- **Primary:** Sharpe ratio and max drawdown of the paper portfolio vs. buy-and-hold SPY (equities sleeve) and BTC (crypto sleeve), **plus an exposure-adjusted comparison** — Vega is often partly in cash, so it is also measured against a risk-matched blend (e.g., X% SPY / (1−X)% T-bills matching its average exposure). Otherwise cash-heavy discipline reads as underperformance.
- **Secondary:** calibration — when Vega says 70% confidence, is it right ~70% of the time? Tracked per signal family via calibration curves.
- **Guardrail:** evidence integrity — 100% of quantitative claims in briefings cite a verifiable data point.
- **Reported but never optimized:** hit-rate (gameable by low-conviction calls).

### Statistical honesty (non-negotiable)

A swing system produces roughly 20–60 closed trades in 3 months — far too few for a statistically meaningful live Sharpe. Therefore:

- **Backtests carry the statistical weight early.** Live paper trading exists to detect *decay* (live performance falling below the backtest's expected band), not to prove alpha in month two.
- **Performance verdicts require ≥ ~100 closed trades or ~12 months**, whichever comes first. Earlier checkpoints judge process, not returns (see §9).

## 5. Trading doctrine

This section is the largest single accuracy lever. Entries get the attention; **sizing and exits determine the P&L.**

### Position sizing & portfolio risk

- **Volatility-scaled sizing:** position size derived from ATR-based stop distance so every trade risks the same fraction of equity.
- **Fixed fractional risk:** 0.5–1.0% of portfolio equity at risk per trade (1R).
- **Max portfolio heat:** total open risk capped (e.g., 6R across all positions).
- **Correlation-aware exposure:** heat is computed across both sleeves with correlation penalties — five longs in NVDA, SMH, QQQ, BTC and ETH is one trade wearing five costumes.
- **Gap-risk sizing:** stops do not protect against overnight gaps — the dominant risk of swing trading. Sizing assumes a gap-through-stop scenario (stress the stop at 2–3× its distance) so a single gap cannot exceed ~2R of damage.

### Exit specification — mandatory on every recommendation

No exit spec, no recommendation. Every call ships with:

1. **Hard stop** (price level, ATR-derived).
2. **Time stop** — the thesis has N days to work; expiry means exit regardless of P&L.
3. **Profit-taking / trailing rule** (e.g., partial at +2R, trail the rest).
4. **Thesis invalidation condition** — the specific observable fact that means the idea is wrong (distinct from the price stop).

### Event risk rules

- **Never hold through an earnings date** unless the signal is explicitly an earnings play. Enforced automatically via the earnings calendar.
- No new entries immediately ahead of scheduled high-impact macro events (FOMC, CPI); the calendar gates entry timing.

### Regime gate

A deliberately **dumb** regime layer — 200-day moving average status, VIX bands, simple breadth, crypto fear/greed — gates signal firing and portfolio heat. It is kept simple on purpose: clever regime models are the most overfit component of retail systems. In hostile regimes Vega reduces or refuses exposure and says so.

### Human override ledger

When Jose (later: any user) deviates from a Vega call — skips it, resizes it, exits early — the override is logged. Vega periodically reports whether human overrides added or destroyed value. The system grades the human too.

## 6. System architecture pillars

### Pillar 1 — Deterministic quant core; LLM at the edges

The single most important architectural decision. Signals, backtests, sizing, exits, and portfolio math live in a deterministic Python engine. The LLM (Claude) does synthesis: interpreting news, writing theses, explaining *why* — always grounded in numbers handed to it by the engine. **The LLM is never permitted to produce a price, percentage, or statistic from memory.** This kills the #1 failure mode of LLM finance tools: confident hallucinated numbers.

### Pillar 2 — Signal registry with a promotion lifecycle

Every signal is a registered hypothesis that earns its way into production:

```
candidate → backtested (walk-forward, out-of-sample) → paper-live → trusted → retired
```

- A signal must have a **written economic rationale before it is ever tested** — no data mining expeditions.
- Only `paper-live` and `trusted` signals influence recommendations.
- Auto-retirement: a live signal whose performance decays below its backtest confidence band is demoted without debate.
- **v1 signals are price/volume-based only.** Free fundamental data (yfinance snapshots) is *current-state*, not point-in-time — using it in backtests injects lookahead bias. Fundamentals may serve as static universe filters only, until point-in-time data is affordable.
- **News/sentiment is never an entry signal in v1.** At free-API latency the move has already happened; retail-latency news carries near-zero alpha. News and sentiment serve as *context, veto filters, and event-risk management* only.

### Pillar 3 — Backtesting protocol

Where accuracy is won or lost. The four sins engineered out from day one:

1. **Lookahead bias** — signals may only use data available at decision time; point-in-time discipline enforced structurally.
2. **Survivorship bias** — bounded by the fixed liquid universe; claims scoped accordingly.
3. **Ignored costs** — commission, spread, and slippage modeled on every simulated trade; crypto spreads modeled wider.
4. **Overfitting / multiple testing** — walk-forward validation with a locked out-of-sample holdout, **plus a backtest registry**: every backtest run ever executed is recorded, so the number of hypotheses tried is known and acceptance thresholds tighten as tests accumulate (deflated-Sharpe spirit). A signal tested twenty ways and cherry-picked is rejected by construction.

### Pillar 4 — Forward verification: the ledger

- Every recommendation written to an append-only ledger: ticker, direction, thesis, confidence, horizon, full exit spec, invalidation condition, signal attribution.
- Simultaneously executed in a **free Alpaca paper account**.
- **Paper fills lie:** paper executions are optimistically filled. A conservative slippage haircut is overlaid on all paper results; paper P&L is treated as directionally honest, never exact. (Live-vs-paper fill deltas become their own tracked metric once real money follows.)
- The **weekly review** grades Vega against its own ledger: returns vs. benchmarks (raw and exposure-adjusted), calibration curves, per-signal-family attribution, decay checks, override analysis.

### Pillar 5 — Data layer

| Need | Free source (Phase 0–2) |
|---|---|
| Equity/ETF prices & history | Alpaca Market Data free tier + yfinance cross-check |
| Consolidated volume | yfinance (Alpaca free = IEX feed, ~2–3% of volume — never trust IEX volume alone) |
| Crypto prices & history | CoinGecko free + Binance public API |
| Macro / rates / regime | FRED (fully free) |
| News (context/veto only) | Finnhub free tier, Alpha Vantage news-sentiment, RSS |
| Sentiment (context only) | CNN Fear & Greed, Crypto Fear & Greed, StockTwits public |
| Earnings & econ calendar | Finnhub free tier |
| Paper trading | Alpaca paper account (free) |

Rules:

- **Dual-source price cross-validation** — one bad candle silently corrupts a backtest; discrepancies quarantine the symbol for the day.
- **Immutable snapshots** — every API response persisted locally (parquet + DuckDB); every backtest reproducible from stored data.
- **Paid upgrade (≤$50/mo: Polygon starter or FMP)** triggers only when a specific, named data limitation is provably costing accuracy.

## 7. Roadmap

| Phase | Content | Gate to advance |
|---|---|---|
| **0 — Foundation** (wks 1–2) | Data layer + cross-validation + snapshot store; Alpaca paper account; ledger schema; briefing v1 (regime + watchlist + calendar — descriptive, no calls) | Data pipeline runs 5 consecutive days clean |
| **1 — Signals & backtest engine** (wks 3–8) | Backtest protocol + backtest registry; first 3–5 price/volume signal families through the lifecycle; briefing v2 with ranked calls incl. full exit specs; auto-paper execution | ≥2 signal families reach `paper-live` legitimately |
| **2 — Self-grading & chat** (wks 9–12) | Weekly review with calibration + attribution + override ledger; on-demand analyst chat over the same engine (chat reads the engine, never freelances) | 3-month process checkpoint passes (§9) |
| **3 — Alerts & paid data** (gated) | Thesis-break + signal-trigger alerts; targeted $50/mo data upgrade | A named data gap justifies the spend; trusted signals exist worth alerting on |
| **4 — Intraday experiment** (gated) | Intraday as a *new signal family* entering the same pipeline — not a pivot. Requires paid real-time data; expect most of the swing edge not to transfer | Swing track record positive at 6-month interim review |
| **5 — SaaS** (gated) | Productize; positioning: *"the only analyst that shows you its graded ledger"* | ≥12 months / ≥100 trades of verified track record (§9) |

**Live-money notes (pre-Phase 4):** PDT rule applies below $25k on a margin account if trade frequency creeps up — cash account or frequency cap avoids it; crypto trades are taxable events (MX/US treatment differs) — the ledger doubles as the tax record.

## 8. SaaS gate & regulatory posture

- **Positioning:** research/education tool with verifiable evidence and a public graded ledger — not personalized investment advice. Ledger transparency is simultaneously the differentiator and the compliance story.
- **Hard gate:** no SaaS until the 12-month/100-trade verdict is in. If the verdict is "no alpha," the pivot (§9) still ships as a product.

## 9. Checkpoints, success metrics, and pre-committed kill criteria

Pre-committed now, because overfitting also happens to founders.

- **3-month checkpoint (process, not returns):** data pipeline integrity ≥99% clean days; 100% of calls carried full exit specs and were paper-executed; calibration curve exists and isn't degrading; no live signal outside its backtest band without demotion. *Returns are explicitly not judged here — the sample is meaningless.*
- **6-month interim:** live signal families still inside backtest bands; calibration error shrinking; exposure-adjusted performance not materially negative. Intraday experiment unlocks only if this passes.
- **12-month / ≥100-trade verdict:** paper portfolio beats its exposure-adjusted benchmark on Sharpe with acceptable drawdown → scale toward SaaS. If not:
- **The pivot:** reposition from "alpha generator" to **evidence-grade research copilot** — verified data, graded ledger, regime context, risk discipline, no return claims. Still differentiated, still sellable, ~80% of the build is shared.

## 10. Risk register

| Risk | Mitigation |
|---|---|
| Overfitting / multiple testing | Promotion lifecycle, locked holdouts, backtest registry, economic-rationale-first, auto-retirement |
| Free-data quality | Dual-source cross-validation, quarantine, immutable snapshots; evidence-triggered paid upgrade |
| Lookahead via fundamentals | v1 price/volume-only signals; fundamentals as static filters only |
| Optimistic paper fills | Conservative slippage haircut on all paper results; live-vs-paper delta tracked later |
| Overnight gap risk | Gap-stressed sizing; earnings/event rules; heat cap |
| Cross-asset correlation | Single-portfolio heat with correlation penalties across sleeves |
| LLM hallucination | Hard quant/LLM boundary; every figure tool-sourced; evidence-integrity guardrail metric |
| Regime-model overfit | Deliberately dumb regime layer |
| Founder bias | Pre-committed checkpoints/kill criteria; override ledger grades the human |
| Statistical self-deception | No return verdicts before 100 trades / 12 months |
