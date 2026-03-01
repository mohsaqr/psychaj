/**
 * PageRank centrality via power iteration — no cytoscape dependency.
 */

/**
 * Compute PageRank centrality for a weighted graph.
 *
 * @param weights   n×n weight matrix (weights[i][j] = edge from i to j)
 * @param damping   damping factor (default: 0.85)
 * @param maxIter   maximum iterations (default: 100)
 * @param tol       convergence tolerance (default: 1e-8)
 */
export function computePageRank(
  weights: number[][],
  damping = 0.85,
  maxIter = 100,
  tol = 1e-8,
): Float64Array {
  const n = weights.length;
  const result = new Float64Array(n);
  if (n === 0) return result;

  // Build column-stochastic transition matrix
  // outDeg[i] = sum of weights from node i
  const outDeg = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j) outDeg[i] += weights[i]![j]!;
    }
  }

  // Initialize uniform
  let pr = new Float64Array(n).fill(1 / n);

  for (let iter = 0; iter < maxIter; iter++) {
    const newPr = new Float64Array(n).fill((1 - damping) / n);

    // Dangling node contribution (nodes with no outgoing edges)
    let danglingSum = 0;
    for (let i = 0; i < n; i++) {
      if (outDeg[i]! <= 0) danglingSum += pr[i]!;
    }
    const danglingContrib = damping * danglingSum / n;
    for (let i = 0; i < n; i++) newPr[i] += danglingContrib;

    // Standard PageRank contribution: j → i
    for (let j = 0; j < n; j++) {
      if (outDeg[j]! <= 0) continue;
      for (let i = 0; i < n; i++) {
        if (i === j) continue;
        const w = weights[j]![i]!;
        if (w > 0) {
          newPr[i] += damping * pr[j]! * w / outDeg[j]!;
        }
      }
    }

    // Check convergence
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(newPr[i]! - pr[i]!);
    pr = newPr;
    if (diff < tol) break;
  }

  // Copy to result
  for (let i = 0; i < n; i++) result[i] = pr[i]!;
  return result;
}
