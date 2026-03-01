# psychaj: Psychometric Network Analysis in JavaScript — Technical Report

## A Standalone Library for Pearson, Partial Correlation, GLASSO, mlVAR, graphicalVAR, NCT, Bootstrap, Community Detection, and PageRank

**Date:** 2026-03-01
**Version:** 0.1.0
**Repository:** [github.com/mohsaqr/psychaj](https://github.com/mohsaqr/psychaj)
**Source files:** 31 (6 entrypoints: core, estimation, models, validation, graph)
**Tests:** 104/104 passing across 17 test files
**Runtime dependency:** carm (multiple regression only)
**Cross-validated against:** R qgraph 1.9.8, glasso 1.11, bootnet 1.7.1, mlVAR 0.5.2, graphicalVAR 0.3.4, NetworkComparisonTest 2.2.2, corpcor 1.6.10

---

## Table of Contents

1. [Design Principles and Architecture](#1-design-principles-and-architecture)
2. [Core: Statistical Foundations](#2-core-statistical-foundations)
3. [Core: Pearson Correlation Matrix](#3-core-pearson-correlation-matrix)
4. [Core: Partial Correlation via Pseudoinverse](#4-core-partial-correlation-via-pseudoinverse)
5. [Core: Within-Between Decomposition](#5-core-within-between-decomposition)
6. [Core: Linear Algebra Primitives](#6-core-linear-algebra-primitives)
7. [Core: Seeded Random Number Generator](#7-core-seeded-random-number-generator)
8. [Estimation: GLASSO Coordinate Descent](#8-estimation-glasso-coordinate-descent)
9. [Estimation: EBIC-GLASSO Path Selection](#9-estimation-ebic-glasso-path-selection)
10. [Estimation: Precision to Partial Correlation](#10-estimation-precision-to-partial-correlation)
11. [Models: Multilevel VAR (mlVAR)](#11-models-multilevel-var-mlvar)
12. [Models: Graphical VAR](#12-models-graphical-var)
13. [Models: PDC and PCC](#13-models-pdc-and-pcc)
14. [Validation: Network Comparison Test (NCT)](#14-validation-network-comparison-test-nct)
15. [Validation: Bootstrap GLASSO](#15-validation-bootstrap-glasso)
16. [Validation: CS-Coefficient (Centrality Stability)](#16-validation-cs-coefficient-centrality-stability)
17. [Validation: P-Value Adjustment](#17-validation-p-value-adjustment)
18. [Graph: Centrality Measures](#18-graph-centrality-measures)
19. [Graph: PageRank](#19-graph-pagerank)
20. [Graph: Community Detection](#20-graph-community-detection)
21. [Graph: Network Metrics](#21-graph-network-metrics)
22. [Performance Optimizations](#22-performance-optimizations)
23. [Cross-Validation: R Equivalence](#23-cross-validation-r-equivalence)
24. [Public API Reference](#24-public-api-reference)
25. [References](#25-references)

---

## 1. Design Principles and Architecture

### 1.1 Motivation

Dynalytics Desktop implements two families of network analysis:

- **Sequence-based** (TNA/FTNA/CTNA/ATNA/WTNA): transitions between discrete states → lives in the `tnaj` package
- **Variable-based** (Pearson, pcor, GLASSO, mlVAR, graphicalVAR, NCT, bootstrap): correlations/regressions on continuous numeric matrices → previously embedded in Desktop's `src/analysis/`

These families share zero algorithms and zero data structures. The variable-based code operates on plain `number[][]` matrices and has no conceptual dependency on tnaj's `Matrix`, `TNA`, or `SequenceData` types. Extracting it into a standalone package:

1. **Eliminates coupling**: psychaj takes `number[][]` inputs and returns `number[][]` outputs. No tnaj types cross the boundary.
2. **Enables reuse**: Any JavaScript/TypeScript project can use psychaj without pulling in tnaj, cytoscape, or any other Desktop dependency.
3. **Simplifies testing**: Each algorithm is tested in isolation against R reference implementations.

### 1.2 Package Structure

```
psychaj/
  src/
    index.ts                    # root barrel re-exports everything
    core/
      types.ts                  # ~25 interfaces — zero runtime code
      stats.ts                  # statistical distribution functions
      pearson.ts                # Pearson correlation matrix
      partial-corr.ts           # partial correlation (Jacobi eigen + pseudoinverse)
      decompose.ts              # within-between decomposition
      linalg.ts                 # Cholesky-based inversion and log-determinant
      rng.ts                    # SeededRNG (xoshiro128**)
    estimation/
      glasso.ts                 # GLASSO coordinate descent (FHT 2008)
      ebic-glasso.ts            # EBIC path selection with warm-start
      theta-to-pcor.ts          # precision matrix → partial correlations
    models/
      mlvar.ts                  # multilevel VAR (OLS within-estimator)
      graphicalvar.ts           # graphical VAR (Rothman 2010 block coordinate descent)
      pdc.ts                    # partial directed correlations
      pcc.ts                    # partial contemporaneous correlations
    validation/
      nct.ts                    # Network Comparison Test (permutation)
      bootstrap.ts              # case-resampling bootstrap with CIs
      cs-coefficient.ts         # centrality stability (callback pattern)
      p-adjust.ts               # Bonferroni, Holm, BH adjustment
    graph/
      centrality.ts             # in/out strength, betweenness, closeness
      pagerank.ts               # power iteration (no cytoscape)
      community.ts              # 6 algorithms (Louvain, walktrap, etc.)
      metrics.ts                # network-level summary statistics
  tests/                        # 17 files mirroring src/
```

### 1.3 Build System

- **Bundler**: tsup with 6 entrypoints, dual ESM/CJS output, `.d.ts` declarations
- **Target**: ES2021, strict TypeScript, bundler module resolution
- **Tests**: vitest
- **Install**: `prepare` script runs `tsup` on `npm install` from GitHub, so `dist/` is available immediately

### 1.4 Dependency Policy

**Single runtime dependency**: `carm` (for `multipleRegression()` in mlVAR). All other algorithms are implemented from scratch. No cytoscape, no tnaj, no D3, no external matrix libraries.

---

## 2. Core: Statistical Foundations

`src/core/stats.ts` provides distribution functions used throughout the package.

### 2.1 Log-Gamma (Lanczos Approximation)

```
lgamma(x) = -tmp + ln(2.5066282746310005 × ser / x)
```

where `tmp = (x + 5.5) - (x + 0.5) × ln(x + 5.5)` and `ser = 1 + Σ_{j=0}^{5} c_j / (x + j + 1)` with Lanczos coefficients `c = [76.18, -86.51, 24.01, -1.23, 1.21×10⁻³, -5.40×10⁻⁶]`. Accurate to ~15 significant digits for x > 0.

### 2.2 Regularized Incomplete Gamma P(a, x)

Two branches for numerical stability:
- **x < a + 1**: Series expansion (converges rapidly for small x)
- **x ≥ a + 1**: Continued fraction via modified Lentz's method

Used by `chiSqCDF(x, df) = P(df/2, x/2)`.

### 2.3 Regularized Incomplete Beta I_x(a, b)

Continued fraction (Numerical Recipes §6.4) with reflection for stability:
- **x < (a+1)/(a+b+2)**: Direct evaluation
- **Otherwise**: `1 - I_{1-x}(b, a)`

Used by:
- `fDistCDF(x, d1, d2) = I_z(d1/2, d2/2)` where `z = d1·x / (d1·x + d2)`
- `tDistCDF(t, df) = 1 - ½ I_x(df/2, ½)` where `x = df / (df + t²)`

### 2.4 Student's t Quantile (Inverse CDF)

Bisection search on `tDistCDF`: 64 iterations in [-1000, 1000] gives ~10⁻⁹ precision. Used for confidence intervals in mlVAR.

### 2.5 Normal CDF

Abramowitz & Stegun 7.1.26 error function approximation (max error ~1.5×10⁻⁷):

```
normalCDF(z) = ½(1 + erf(z/√2))
```

### 2.6 Percentile (R Quantile Type 7)

Linear interpolation matching R's default `quantile(x, p, type=7)`:

```
pos = p × (n - 1)
result = sorted[⌊pos⌋] + (sorted[⌈pos⌉] - sorted[⌊pos⌋]) × (pos - ⌊pos⌋)
```

---

## 3. Core: Pearson Correlation Matrix

`computePearsonMatrix(matrix: number[][]): number[][]`

Matches R `cor()` exactly.

```
means[j] = (1/n) Σ_i x[i][j]
sds[j]   = √((1/(n-1)) Σ_i (x[i][j] - means[j])²)          // ddof=1
cov(a,b) = (1/(n-1)) Σ_i (x[i][a] - means[a])(x[i][b] - means[b])
R[a][b]  = cov(a,b) / (sds[a] × sds[b])
```

**Implementation details:**
- Uses `Float64Array` for means and SDs (avoids boxing overhead)
- Pre-computes `1/(n-1)` to replace division with multiplication in the inner loop
- Hoists `means[a]` and `sds[a]` out of the `b` loop
- Returns identity for `n < 2` or `p = 0`
- Zero-variance columns produce `r = 0` (guarded by `denom < 1e-12`)

**Verified against R:** Exact match to machine epsilon on all test cases.

---

## 4. Core: Partial Correlation via Pseudoinverse

`computePartialCorrMatrix(R: number[][]): number[][]`

Matches R `corpcor::cor2pcor()`.

### 4.1 Jacobi Eigenvalue Decomposition

For symmetric `A`, iteratively zero the largest off-diagonal element via Givens rotations:

```
For each sweep:
  Find (p,q) = argmax |A[i][j]| for i < j
  If maxOff < 1e-14: converged
  Compute rotation: τ = (A[q][q] - A[p][p]) / (2A[p][q])
                    t = sign(τ) / (|τ| + √(1 + τ²))
                    c = 1/√(1 + t²), s = t·c
  Apply rotation to A and accumulate eigenvectors V
```

Convergence: max `n × 30` sweeps. Returns eigenvalues (diagonal of rotated A) and eigenvectors (columns of V).

### 4.2 Moore-Penrose Pseudoinverse

```
A⁺ = V D⁺ Vᵀ
```

where `D⁺[k] = 1/λ_k` if `|λ_k| > tol`, else `0`. Tolerance: `max(n × ε × maxEig × 10, maxEig × 10⁻⁴)`. The raised floor (10⁻⁴ × max) is necessary because Jacobi accumulates ~10⁻⁷ error on the structural zero eigenvalue of rank-deficient correlation matrices (e.g., compositional data where proportions sum to 1), whereas LAPACK gives ~10⁻¹⁶.

### 4.3 Partial Correlation

```
Theta = pseudoInverse(R)
pcor[i][j] = -Theta[i][j] / √(Theta[i][i] × Theta[j][j])    for i ≠ j
```

**Verified against R:** Matches `corpcor::cor2pcor()` including rank-deficient cases.

---

## 5. Core: Within-Between Decomposition

`decomposeWithinBetween(data: number[][], actorIds: string[])`

For repeated-measures data:

- **Between**: Per-actor means (one row per unique actor, insertion order)
- **Within**: Each row minus its actor's mean (person-mean centering)

Used by corrnet for within-person vs between-person network decomposition.

---

## 6. Core: Linear Algebra Primitives

### 6.1 Cholesky-Based Log-Determinant

```
L Lᵀ = M  (Cholesky factorization)
log|M| = 2 Σ_i log(L[i][i])
```

Returns `-Infinity` if `M` is not positive definite (any `L[i][i]² ≤ 0`).

### 6.2 Cholesky-Based Matrix Inversion

```
L Lᵀ = M
L⁻¹ via forward substitution
M⁻¹ = L⁻ᵀ L⁻¹
```

Uses `max(s, 1e-15)` in the Cholesky diagonal for numerical safety. Result is symmetric by construction.

---

## 7. Core: Seeded Random Number Generator

`SeededRNG` implements xoshiro128** (Blackman & Vigna, 2018), identical to tnaj's `SeededRNG`.

### 7.1 State Initialization

```
seed → splitmix32(seed) → s0
s0 → splitmix32(s0) → s1
s1 → splitmix32(s1) → s2
s2 → splitmix32(s2) → s3
```

where `splitmix32` is Stafford's finalizer:

```
z = (z + 0x9E3779B9) >>> 0
z = imul(z ^ (z >>> 16), 0x85EBCA6B) >>> 0
z = imul(z ^ (z >>> 13), 0xC2B2AE35) >>> 0
return (z ^ (z >>> 16)) >>> 0
```

### 7.2 Core Generator

```
result = imul(rotl(imul(s1, 5), 7), 9) >>> 0
t = (s1 << 9) >>> 0
s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = rotl(s3, 11)
```

All arithmetic uses `>>> 0` for unsigned 32-bit semantics. Period: 2¹²⁸ - 1.

### 7.3 Derived Methods

| Method | Description |
|--------|-------------|
| `random()` | `next() / 2³²` → [0, 1) |
| `randInt(max)` | `⌊random() × max⌋` |
| `shuffle(arr)` | Fisher-Yates (in-place) |
| `permutation(n)` | Shuffle [0, ..., n-1] |
| `choice(n, size)` | With replacement |
| `choiceWithoutReplacement(n, size)` | Partial Fisher-Yates |

**Verified:** Produces identical sequences to tnaj's `SeededRNG` for the same seed.

---

## 8. Estimation: GLASSO Coordinate Descent

`runGlasso(R, rho, ...)` — Friedman, Hastie & Tibshirani (2008).

### 8.1 Algorithm

```
Input: R (p×p correlation matrix), ρ (regularization parameter)
Initialize: W = R (or warm-start), Beta = 0 (or warm-start)

Repeat until convergence:
  For each column j = 0..p-1:
    Partition: W₁₁ = W[-j,-j], s₁₂ = R[-j,j]
    Extract: β = Beta[j][-j]
    Coordinate descent (Lasso on W₁₁ β = s₁₂ with L1 penalty ρ):
      For each k = 0..p-2:
        partial = s₁₂[k] - Σ_{l≠k} W₁₁[k,l] × β[l]
        β[k] = softThreshold(partial, ρ) / W₁₁[k,k]
    Update: w₁₂ = W₁₁ β, W[-j,j] = w₁₂, W[j,-j] = w₁₂
    Store: Beta[j][-j] = β

Build Theta from W and Beta:
  θ[j,j] = 1 / (W[j,j] - Σ_{a≠j} W[a,j] × Beta[j,a])
  θ[j,a] = -Beta[j,a] × θ[j,j]
Symmetrize: Theta = (Theta + Thetaᵀ) / 2
```

### 8.2 Key Design Choices

- **penalize.diagonal = FALSE**: The diagonal of W stays at `R[j,j]` (not `R[j,j] + ρ`), matching `qgraph::EBICglasso` default
- **Outer update w₁₂ = W₁₁ β**: NOT `w₁₂ = β`. Using `W₁₁ β` keeps w₁₂ bounded by `s₁₂ ± ρ`, preventing divergence for near-singular R
- **Warm-starting**: Previous lambda's converged W/Beta initialize the next lambda. Critical for path stability.

### 8.3 Internal Structure

The implementation splits into two functions:

- `glassoCD()`: Core coordinate descent returning only `{W, Beta}` — no Theta allocation. Uses `Float64Array` pre-allocated work buffers (`W11`, `s12`, `beta`, `w12`) outside the column loop. Supports `inPlace` mode to skip W/Beta copy on warm-start.
- `buildTheta()`: Constructs and symmetrizes Theta from converged W and Beta.
- `runGlasso()`: Convenience wrapper calling both.

This split enables `ebicGlasso` to scan 100 lambdas without allocating Theta for each one (see §9).

**Verified against R:** Matches `glasso::glasso()` within 1×10⁻⁵.

---

## 9. Estimation: EBIC-GLASSO Path Selection

`ebicGlasso(R, n, gamma, nLambda, ...)` — matches `qgraph::EBICglasso()` within 1×10⁻⁴.

### 9.1 Lambda Grid

```
λ_max = max |R[i,j]| for i < j
λ_min = 0.01 × λ_max
λ[k] = exp(log(λ_max) + (log(λ_min) - log(λ_max)) × k/(nLambda-1))    k = 0..99
```

100 log-spaced values from most sparse to most dense.

### 9.2 EBIC Score (Theta-Free Computation)

During the path scan, EBIC is computed directly from W and Beta without constructing Theta. This avoids 99 out of 100 Theta allocations.

**Diagonal of Theta:**

```
θ[j,j] = 1 / (W[j,j] - Σ_{a≠j} W[a,j] × Beta[j,a])
```

**Trace of R × Theta (from W/Beta):**

```
tr(R Θ) = Σ_j θ[j,j] × (R[j,j] - Σ_{a≠j} R[a,j] × Beta[j,a])
```

This identity holds because R is symmetric and symmetrization of Theta does not affect the trace of a symmetric × symmetric product.

**Edge count (from Beta):**

```
|E| = #{(i,j) : i < j, |Beta[j,i] × θ[j,j] + Beta[i,j] × θ[i,i]| / 2 > 10⁻⁸}
```

**EBIC:**

```
log ℓ = (n/2)(log|Θ| - tr(RΘ))      where log|Θ| = -log|W|
EBIC = -2 log ℓ + log(n) × |E| + 4γ log(p) × |E|
```

### 9.3 Path Optimization

- **Warm-start**: Each lambda's solution initializes the next via `glassoCD(..., prevW, prevBeta, inPlace=true)`
- **In-place mode**: Only copies W/Beta when a new EBIC minimum is found (saves 99 copies)
- **Refinement**: After path selection, re-runs GLASSO at the best lambda with tight tolerance (`10⁴` outer iters, `10⁻⁸` tol). Skipped when `skipRefinement=true` (bootstrap iterations).

---

## 10. Estimation: Precision to Partial Correlation

```
pcor[i,j] = -Θ[i,j] / √(Θ[i,i] × Θ[j,j])    for i ≠ j
pcor[i,i] = 0
```

Used by GLASSO networks, contemporaneous (PCC), and between-subjects networks.

---

## 11. Models: Multilevel VAR (mlVAR)

`fitMlVAR(rows, opts)` — matches R `mlVAR::mlVAR(temporal="fixed")`.

See the companion [mlVAR Technical Report](../MLVAR-TECHNICAL-REPORT.md) for full algorithmic details. Summary:

1. Sort by (id, day, beep)
2. Person-mean center predictors and outcomes (within-estimator OLS)
3. OLS per outcome with df correction: `df_correct = N - d - n_subjects`
4. Temporal network: `d×d` fixed-effect Beta matrix
5. Contemporaneous: EBIC-GLASSO on OLS residuals
6. Between-subjects: EBIC-GLASSO on person means

---

## 12. Models: Graphical VAR

`fitGraphicalVAR(rows, opts)` — matches R `graphicalVAR::graphicalVAR()`.

See the companion [graphicalVAR Technical Report](../GRAPHICALVAR-TECHNICAL-REPORT.md) for full algorithmic details. Summary:

1. Scale globally, center within-person, build lag pairs
2. Rothman–Levina–Zhu (2010) block coordinate descent alternating Beta step (L1-penalized regression) and Kappa step (GLASSO on residual covariance)
3. EBIC model selection over 2D lambda grid (λ_β × λ_κ)
4. Output: PDC (partial directed correlations) and PCC (partial contemporaneous correlations)

---

## 13. Models: PDC and PCC

### 13.1 Partial Directed Correlations

```
PDC[j,k] = β[j,k] / √(σ[k,k] × κ[j,j] + β[j,k]²)
```

where `σ = κ⁻¹` (residual covariance). Bounded in [-1, 1]. Represents the strength of the temporal effect from variable j to variable k, controlling for all other temporal and contemporaneous effects.

### 13.2 Partial Contemporaneous Correlations

```
PCC = thetaToPcor(κ) = -cov2cor(κ)
```

Identical to the partial correlation conversion in §10. Symmetric, zero diagonal.

---

## 14. Validation: Network Comparison Test (NCT)

`networkComparisonTest(xData, yData, opts)` — matches R `NetworkComparisonTest::NCT()`.

### 14.1 Algorithm

```
1. Estimate networks for groups X and Y: net_X, net_Y
2. Compute observed differences: D_obs[i,j] = net_X[i,j] - net_Y[i,j]
3. Pool data: pooled = [X; Y]
4. Pre-compute lambda path from pooled correlation (for GLASSO method)
5. For each permutation t = 1..iter:
   a. Randomly partition pooled into groups of size n_X and n_Y
   b. Estimate permuted networks: net_pX, net_pY
   c. Compute permuted differences: D_perm[i,j]
   d. Accumulate: exceed_count if |D_perm| ≥ |D_obs|
6. P-values: p[i,j] = (exceed_count + 1) / (iter + 1)
7. Effect sizes: D_obs / sd(D_perm)
8. Apply p-value adjustment (if requested)
```

### 14.2 Implementation Details

- **RNG**: `SeededRNG` (xoshiro128**) for reproducible permutations
- **Paired mode**: Randomly swap within pairs instead of full shuffle
- **Lambda path**: Pre-computed from pooled data, shared across permutations (avoids lambda instability)
- **Pre-computed indices**: Optional `permutationIndices` parameter for exact R equivalence testing
- **Running accumulators**: Uses `Float64Array` for exceed counts, sum of diffs, and sum of squared diffs (avoids storing all permutation results)

---

## 15. Validation: Bootstrap GLASSO

`bootstrapGlasso(rawData, labels, opts)` — matches R `bootnet::bootnet(default="EBICglasso", type="case")`.

### 15.1 Algorithm

```
1. Estimate original network: pcor_orig, lambda_orig = estimatePcor(rawData)
2. For each bootstrap b = 1..iter:
   a. Case-resample: n rows with replacement (SeededRNG)
   b. Estimate bootstrap network: pcor_boot = estimatePcor(sample)
   c. Store edge values: bootStorage[edge][b] = pcor_boot[i][j]
3. Per-edge statistics:
   - bootMean = mean(bootStorage[edge])
   - bootSd = sd(bootStorage[edge])
   - CI = percentile(sorted, [α/2, 1-α/2])  (R quantile type 7)
   - significant = CI excludes 0
4. Pairwise edge differences:
   - For each pair (A, B): diff_samples[t] = A_boot[t] - B_boot[t]
   - CI of diff distribution
   - significant = diff CI excludes 0
```

### 15.2 R Equivalence: Per-Sample EBIC vs Fixed Lambda

By default (`fixLambda: false`), each bootstrap sample re-runs the full 100-lambda EBIC path search, matching R bootnet's behavior exactly. This was verified by:

1. Generating 300 × 12 data with Toeplitz correlation structure in R
2. Running 500 bootstrap iterations in R with `qgraph::EBICglasso()` and fixed permutation indices
3. Replaying the identical 500 samples in TypeScript with the same indices
4. Comparing all 33,000 individual bootstrap edge values

**Result:** Max per-sample Δ = 6.7×10⁻⁶, max CI Δ = 3.3×10⁻⁷, 100% significance agreement (66/66 edges).

### 15.3 Optional Fast Mode

Setting `fixLambda: true` fixes the lambda selected on the original data for all bootstrap samples. This reduces each iteration from ~100 GLASSO solves to 1 (~100× speedup). This does NOT match R bootnet and produces different CIs, but is statistically valid for applications where R equivalence is not required.

---

## 16. Validation: CS-Coefficient (Centrality Stability)

`estimateCS(data, networkFn, opts)` — callback-based centrality stability.

### 16.1 Callback Pattern

Unlike the tnaj-coupled Desktop version, psychaj's CS uses a generic callback:

```typescript
estimateCS(freqMatrix, (subset) => {
  const R = computePearsonMatrix(subset);
  const { Theta } = ebicGlasso(R, subset.length);
  return thetaToPcor(Theta);
}, { measures: ['inStrength', 'outStrength', 'betweenness'] });
```

This decouples CS estimation from any specific network estimation method.

### 16.2 Algorithm

```
For each drop proportion dp ∈ {0.1, 0.2, ..., 0.9}:
  For each iteration t = 1..iter:
    Drop ⌊n × dp⌋ rows without replacement
    Estimate network on remaining rows via networkFn(subset)
    Compute centralities on subset network
    Correlate with original centralities (Pearson or Spearman)
  Track proportion of correlations ≥ threshold (default 0.7)

CS[measure] = max dp where ≥ certainty (default 95%) of correlations ≥ threshold
```

---

## 17. Validation: P-Value Adjustment

`pAdjust(pvals, method)` — matches R `p.adjust()`.

| Method | Formula |
|--------|---------|
| Bonferroni | `p_adj = min(p × n, 1)` |
| Holm | Step-down: sort ascending, `p_adj[k] = cummax(p[k] × (n - k))` |
| BH (Benjamini-Hochberg) | Step-up: sort ascending, `p_adj[k] = cummin(p[k] × n / (k + 1))` from bottom |

---

## 18. Graph: Centrality Measures

`computeCentralities(weights, directed?)` — takes `number[][]`, returns `{inStrength, outStrength, betweenness, closeness}`.

### 18.1 Strength

```
outStrength[i] = Σ_{j≠i} w[i][j]
inStrength[j]  = Σ_{i≠j} w[i][j]
```

### 18.2 Betweenness (Brandes' Algorithm)

Weighted shortest-path betweenness using Dijkstra with distance = 1/weight:

```
For each source s:
  Dijkstra from s (priority: smallest distance first)
  Track σ[v] (number of shortest paths) and pred[v] (predecessors)
  Back-propagate: δ[v] += (σ[v]/σ[w]) × (1 + δ[w]) for each predecessor
  betweenness[v] += δ[v]

For undirected: divide by 2
```

### 18.3 Closeness

```
closeness[s] = (reachable nodes from s) / Σ_{t reachable} dist(s, t)
```

Dijkstra-based with distance = 1/weight. Returns 0 for isolated nodes.

---

## 19. Graph: PageRank

`computePageRank(weights, damping?, maxIter?, tol?)` — own power iteration, no cytoscape.

### 19.1 Algorithm

```
Initialize: pr[i] = 1/n for all i
Compute: outDeg[i] = Σ_{j≠i} w[i][j]

Repeat until convergence:
  danglingSum = Σ_{i: outDeg[i]=0} pr[i]
  For each node i:
    pr_new[i] = (1-d)/n + d × danglingSum/n + d × Σ_{j: w[j][i]>0} pr[j] × w[j][i] / outDeg[j]
  If Σ |pr_new - pr| < tol: stop
  pr = pr_new
```

Default: `damping = 0.85`, `maxIter = 100`, `tol = 1e-8`. Values sum to 1.

Handles dangling nodes (no outgoing edges) by distributing their rank uniformly.

---

## 20. Graph: Community Detection

`detectCommunities(weights, method, directed?)` — 6 algorithms on `number[][]`.

| Method | Algorithm | Reference |
|--------|-----------|-----------|
| `louvain` | Greedy modularity optimization with node moves | Blondel et al. (2008) |
| `walktrap` | Random walks + Ward agglomerative clustering | Pons & Latapy (2005) |
| `fast_greedy` | Agglomerative modularity merging | Clauset et al. (2004) |
| `label_prop` | Label propagation with deterministic tie-breaking | Raghavan et al. (2007) |
| `leading_eigen` | Power iteration on modularity matrix B | Newman (2006) |
| `edge_betweenness` | Girvan-Newman edge removal | Girvan & Newman (2002) |

### 20.1 Walktrap Details

For directed graphs, walktrap uses the original directed weight matrix for transition probabilities:

```
P[i][j] = w[i][j] / Σ_k w[i][k]    (row-stochastic)
P^t = P × P × ... × P               (t-step transition probabilities, default t=4)
```

Ward-style agglomerative clustering on the walk distances:

```
d²(C_A, C_B) = Σ_k (dist_A[k] - dist_B[k])² / deg[k]
ward = (|C_A| × |C_B|) / (|C_A| + |C_B|) × d²
```

Best partition tracked by modularity over the merge dendrogram.

### 20.2 Output

All methods return `{ assignments: number[], nCommunities, method, modularity }` with 0-indexed contiguous community IDs.

---

## 21. Graph: Network Metrics

`computeGraphMetrics(weights, directed?)` — network-level summary.

| Metric | Formula |
|--------|---------|
| Density | edges / max_edges |
| Reciprocity | mutual_edges / total_edges (directed only) |
| Transitivity | 3 × triangles / connected_triples |
| Avg path length | mean finite shortest path (Floyd-Warshall, dist = 1/w) |
| Diameter | max finite shortest path |
| Components | weakly connected (BFS on symmetrized adjacency) |

---

## 22. Performance Optimizations

### 22.1 GLASSO Path Scan

The EBIC path evaluates ~100 GLASSO solutions per call. Three optimizations reduce overhead without changing numerical results:

| Optimization | Mechanism | Savings |
|-------------|-----------|---------|
| Theta-free EBIC | Compute EBIC from W/Beta directly | Avoids 99/100 Theta allocations |
| Float64Array buffers | Pre-allocate W11, s12, beta, w12 as typed arrays | Reduces GC pressure |
| In-place warm-start | Mutate W/Beta between lambdas, copy only at new EBIC minimum | Avoids 99 matrix copies |

Combined effect: **~7-9× speedup** on bootstrap workloads (measured on 200 iterations, p=6-10).

### 22.2 Bootstrap Performance

| Config | Default (per-sample EBIC) | fixLambda mode |
|--------|---------------------------|----------------|
| p=6, 1000 iter | ~0.6s | ~0.1s |
| p=10, 1000 iter | ~3s | ~0.2s |
| p=12, 500 iter | ~2s | ~0.3s |

---

## 23. Cross-Validation: R Equivalence

### 23.1 Methodology

```
┌─────────────────────────────────────┐
│ R generates ground truth            │
│   Fixed data + fixed boot indices   │
│   → JSON with 10-digit precision    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│ TS replays identical samples        │
│   Same data, same indices           │
│   → per-edge per-sample values      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│ Edge-by-edge comparison             │
│   original weight, boot mean/sd/CI  │
│   significance agreement            │
└─────────────────────────────────────┘
```

### 23.2 Results: 300 × 12, 500 Bootstrap Samples

```
Dataset: 300 rows × 12 nodes
Edges: 66 (p*(p-1)/2)
Bootstrap samples: 500
Total values compared: 33,000

Lambda:         R = 0.11725910, TS = 0.11725910, Δ = 4.3×10⁻¹²
Orig pcor:      max Δ = 2.0×10⁻⁷
Per-sample:     max Δ = 6.7×10⁻⁶  (across 33,000 values)
Bootstrap mean: max Δ = 3.8×10⁻⁸
Bootstrap SD:   max Δ = 2.5×10⁻⁸
Bootstrap CI:   max Δ = 3.3×10⁻⁷
Significant edges (R):  6/66
Significant edges (TS): 6/66
Agreement:      66/66 (100%)
```

### 23.3 Results: 60 × 4, 1000 Bootstrap Samples

```
Dataset: 60 rows × 4 nodes
Edges: 6
Bootstrap samples: 1000
Total values compared: 6,000

Lambda:         Δ = 1.4×10⁻¹²
Per-sample:     max Δ = 6.5×10⁻⁵
Bootstrap mean: max Δ = 1.1×10⁻⁶
Bootstrap CI:   max Δ = 1.7×10⁻⁵
Agreement:      6/6 (100%)
```

### 23.4 Source of Residual Differences

The ~10⁻⁵ per-sample differences arise from the coordinate descent convergence path: R uses Fortran `glasso` (LAPACK-backed) while psychaj uses JavaScript. Both converge to the same solution but via slightly different floating-point trajectories. The differences are well below any practically meaningful threshold.

---

## 24. Public API Reference

### 24.1 Core (~15 exports)

```typescript
// Correlation
computePearsonMatrix(matrix: number[][]): number[][]
computePartialCorrMatrix(R: number[][]): number[][]
decomposeWithinBetween(data: number[][], actorIds: string[]): DecompResult

// Linear algebra
invertSymmetric(M: number[][]): number[][]
logDet(M: number[][]): number

// Statistics
lgamma, gammaP, chiSqCDF, betaI, fDistCDF, tDistCDF, tDistInv, normalCDF
percentile(sorted: Float64Array | number[], p: number): number
pearsonCorr(a: Float64Array | number[], b: Float64Array | number[]): number

// RNG
class SeededRNG { constructor(seed: number); random(); randInt(max); shuffle(arr); permutation(n); choice(n, size); choiceWithoutReplacement(n, size); }
```

### 24.2 Estimation (3 exports)

```typescript
runGlasso(R, rho, maxIterOuter?, tolOuter?, maxIterInner?, tolInner?, initW?, initBeta?): { Theta, W, Beta }
ebicGlasso(R, n, gamma?, nLambda?, customLambdas?, skipRefinement?): { Theta, lambda }
thetaToPcor(Theta: number[][]): number[][]
```

### 24.3 Models (5 exports)

```typescript
fitMlVAR(rows: Record<string, string|number>[], opts: MlVAROptions): MlVARResult
computeImpulseResponse(temporal, labels, shockedVarIdx, nSteps?, magnitude?): ImpulseResponse
fitGraphicalVAR(rows: Record<string, string|number>[], opts: GraphicalVAROptions): GraphicalVARResult
computePDC(beta: number[][], kappa: number[][]): number[][]
computePCC(kappa: number[][]): number[][]
```

### 24.4 Validation (5 exports)

```typescript
networkComparisonTest(xData, yData, opts?: NctOptions): NctResult
computeNctLambdaPath(pooledData, nLambda?): number[]
bootstrapGlasso(rawData, labels, opts?: BootstrapGlassoOptions): BootstrapGlassoResult
estimateCS(data, networkFn: (subset: number[][]) => number[][], opts?: StabilityOptions): StabilityResult
pAdjust(pvals: number[], method: 'none'|'bonferroni'|'holm'|'BH'): number[]
```

### 24.5 Graph (4 exports)

```typescript
computeCentralities(weights: number[][], directed?: boolean): CentralityResult
computePageRank(weights: number[][], damping?, maxIter?, tol?): Float64Array
detectCommunities(weights: number[][], method: CommunityMethod, directed?: boolean): CommunityResult
computeGraphMetrics(weights: number[][], directed?: boolean): GraphMetrics
```

### 24.6 Types (~25 interfaces)

`MlVAROptions`, `MlVARCoef`, `MlVARResult`, `ImpulseResponse`, `GraphicalVAROptions`, `GraphicalVARResult`, `NctOptions`, `NctEdgeResult`, `NctResult`, `BootstrapGlassoOptions`, `BootstrapGlassoResult`, `GlassoBootEdge`, `GlassoBootDiff`, `StabilityOptions`, `StabilityResult`, `CommunityResult`, `CentralityResult`, `GraphMetrics`, `EstimationMethod`, `CommunityMethod`

---

## 25. References

- Blackman, D., & Vigna, S. (2018). Scrambled linear pseudorandom number generators. *arXiv:1805.01407*.
- Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics*, P10008.
- Clauset, A., Newman, M. E. J., & Moore, C. (2004). Finding community structure in very large networks. *Physical Review E*, 70(6), 066111.
- Epskamp, S., Borsboom, D., & Fried, E. I. (2018). Estimating psychological networks and their accuracy: A tutorial paper. *Behavior Research Methods*, 50(1), 195–212.
- Epskamp, S., Waldorp, L. J., Mottus, R., & Borsboom, D. (2018). The Gaussian graphical model in cross-sectional and time-series data. *Multivariate Behavioral Research*, 53(4), 453–480.
- Foygel, R., & Drton, M. (2010). Extended Bayesian information criteria for Gaussian graphical models. *Advances in Neural Information Processing Systems*, 23.
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*, 9(3), 432–441.
- Girvan, M., & Newman, M. E. J. (2002). Community structure in social and biological networks. *Proceedings of the National Academy of Sciences*, 99(12), 7821–7826.
- Newman, M. E. J. (2006). Finding community structure in networks using the eigenvectors of matrices. *Physical Review E*, 74(3), 036104.
- Pons, P., & Latapy, M. (2005). Computing communities in large networks using random walks. *Computer and Information Sciences — ISCIS 2005*, 284–293.
- Raghavan, U. N., Albert, R., & Kumara, S. (2007). Near linear time algorithm to detect community structures in large-scale networks. *Physical Review E*, 76(3), 036106.
- Rothman, A. J., Levina, E., & Zhu, J. (2010). Sparse multivariate regression with covariance estimation. *Journal of Computational and Graphical Statistics*, 19(4), 947–962.
- van Borkulo, C. D., et al. (2022). Comparing network structures on three aspects: A permutation test. *Psychological Methods*, 27(2), 275–293.
