# Statistical Testing for Trading Strategies

## Multiple Testing Problem

When you test N strategies, the probability of finding at least one "significant" result by chance is:
P(at least one false positive) = 1 - (1 - α)^N

With α=0.05 and N=20 strategies: P ≈ 64%

### Corrections
1. **Bonferroni**: Divide α by N. Simple but conservative.
2. **Benjamini-Hochberg-Yekutieli (BHY)**: Controls False Discovery Rate. Less conservative, more appropriate for finance.
3. **Harvey, Liu & Zhu (2016)**: For factor discovery, need t-stat > 3.0 (not 2.0).

## Sharpe Ratio Caveats

- Sharpe ratio is NOT normally distributed for non-normal returns
- Annualized Sharpe: SR_annual = SR_daily * √252 — assumes i.i.d. returns (usually wrong)
- For autocorrelated returns, use Lo (2002) correction
- A "good" Sharpe depends on the strategy type:
  - Stat arb: 2.0+ (high-frequency data)
  - Momentum: 0.5-1.0
  - Value: 0.3-0.7

## Out-of-Sample Testing

**Minimum requirements:**
- Reserve 20-30% of data for out-of-sample
- Never touch OOS data until final evaluation
- If OOS Sharpe < 50% of in-sample Sharpe → likely overfit
- Consider walk-forward optimization as a more robust alternative

## Stationarity
- Test for structural breaks (Chow test, CUSUM)
- Rolling parameter stability analysis
- Regime-switching models if parameters vary
