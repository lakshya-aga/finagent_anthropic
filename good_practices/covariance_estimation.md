# Covariance Matrix Estimation

## Shrinkage Is Mandatory for Large Universes

When estimating a covariance matrix from N assets and T observations:
- The sample covariance matrix has N*(N+1)/2 free parameters
- For N=500 assets, that's 125,250 parameters
- Many "relationships" will appear significant purely by chance

**Rule: Always apply shrinkage to covariance matrices when N/T > 0.1**

### Recommended Approaches
1. **Ledoit-Wolf shrinkage** (constant correlation target) — good default
2. **Oracle Approximating Shrinkage (OAS)** — better for normal distributions
3. **Factor model covariance** — when you have a good factor model

### Common Mistakes
- Using raw sample covariance for portfolio optimization → extreme weights
- Using full-sample covariance for rolling analysis → lookahead bias
- Not handling missing data before estimation → biased correlations

### Library Function
Use `shrink_covariance()` from the tools library with appropriate target.
