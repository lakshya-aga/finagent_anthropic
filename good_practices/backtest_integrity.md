# Backtest Integrity

## The Cardinal Sins of Backtesting

### 1. Lookahead Bias
Using information that would not have been available at the time of the trading decision.

**Common sources:**
- Full-sample normalization (z-score with full mean/std instead of expanding window)
- Using revised economic data before revision date
- Forward-filling prices before actual publication
- Using point-in-time adjusted data as if it were available historically

**Fix:** Every computation at time t must use ONLY data available at time t.

### 2. Survivorship Bias
Only including securities that survived until the end of the sample.

**Impact:** Overstates returns by 1-3% per year in equity strategies.

**Fix:** Use point-in-time universe membership. Include delisted securities.

### 3. Data Snooping
Testing many strategies and only reporting the best one.

**The problem:** With 100 independent strategy variants, expect ~5 to be "significant" at 5% level purely by chance.

**Fix:**
- Apply Bonferroni correction or BHY procedure
- Use Harvey, Liu & Zhu (2016) haircut: required Sharpe ≈ 3.0 for new factors
- Always have an out-of-sample holdout period (minimum 20% of data)
- Report ALL strategies tested, not just winners

### 4. Transaction Costs
**Typical costs:**
- US large-cap equities: 5-10 bps per trade
- US small-cap: 20-50 bps
- Emerging markets: 30-100 bps
- High-frequency: 0.5-2 bps but with market impact

**Rule:** Always model transaction costs. A strategy that doesn't survive costs isn't a strategy.

### 5. Market Impact
For strategies with significant AUM, your trading moves the market.

**Rule of thumb:** Square-root model: impact ∝ σ * √(V/ADV) where V = trade volume, ADV = average daily volume.
