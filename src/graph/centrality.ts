/**
 * Centrality measures for weighted directed/undirected graphs.
 * All functions take number[][] weight matrices — no tnaj dependency.
 */

import type { CentralityResult } from '../core/types';

/**
 * Compute in-strength, out-strength, betweenness, and closeness centralities.
 *
 * @param weights   n×n weight matrix (directed: weights[i][j] = edge from i to j)
 * @param directed  whether the graph is directed (default: true)
 */
export function computeCentralities(
  weights: number[][],
  directed = true,
): CentralityResult {
  const n = weights.length;

  const inStrength = new Float64Array(n);
  const outStrength = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      outStrength[i] += weights[i]![j]!;
      inStrength[j] += weights[i]![j]!;
    }
  }

  // Betweenness centrality (Brandes' algorithm on weighted graph)
  const betweenness = new Float64Array(n);

  for (let s = 0; s < n; s++) {
    // Dijkstra-like shortest path computation
    const dist = new Float64Array(n).fill(Infinity);
    const sigma = new Float64Array(n);  // number of shortest paths
    const pred: number[][] = Array.from({ length: n }, () => []);
    dist[s] = 0;
    sigma[s] = 1;

    const visited = new Uint8Array(n);
    const stack: number[] = [];

    for (let step = 0; step < n; step++) {
      // Find unvisited node with smallest distance
      let u = -1;
      let minDist = Infinity;
      for (let i = 0; i < n; i++) {
        if (!visited[i] && dist[i]! < minDist) {
          minDist = dist[i]!;
          u = i;
        }
      }
      if (u < 0) break;
      visited[u] = 1;
      stack.push(u);

      for (let v = 0; v < n; v++) {
        if (v === u) continue;
        const w = weights[u]![v]!;
        if (w <= 0) continue;
        const edgeDist = 1 / w;  // distance = inverse weight
        const newDist = dist[u]! + edgeDist;

        if (newDist < dist[v]! - 1e-10) {
          dist[v] = newDist;
          sigma[v] = sigma[u]!;
          pred[v] = [u];
        } else if (Math.abs(newDist - dist[v]!) < 1e-10) {
          sigma[v] += sigma[u]!;
          pred[v]!.push(u);
        }

        if (!directed) {
          const w2 = weights[v]![u]!;
          if (w2 > 0) {
            const edgeDist2 = 1 / w2;
            const newDist2 = dist[u]! + edgeDist2;
            if (newDist2 < dist[v]! - 1e-10) {
              dist[v] = newDist2;
              sigma[v] = sigma[u]!;
              pred[v] = [u];
            } else if (Math.abs(newDist2 - dist[v]!) < 1e-10) {
              sigma[v] += sigma[u]!;
              if (!pred[v]!.includes(u)) pred[v]!.push(u);
            }
          }
        }
      }
    }

    // Accumulate betweenness
    const delta = new Float64Array(n);
    while (stack.length > 0) {
      const w = stack.pop()!;
      for (const v of pred[w]!) {
        const frac = sigma[v]! > 0 ? (sigma[v]! / sigma[w]!) * (1 + delta[w]!) : 0;
        delta[v] += frac;
      }
      if (w !== s) {
        betweenness[w] += delta[w]!;
      }
    }
  }

  // For undirected graphs, betweenness is counted twice
  if (!directed) {
    for (let i = 0; i < n; i++) betweenness[i] /= 2;
  }

  // Closeness centrality (inverse of mean shortest path distance)
  const closeness = new Float64Array(n);
  for (let s = 0; s < n; s++) {
    // Dijkstra
    const dist = new Float64Array(n).fill(Infinity);
    dist[s] = 0;
    const vis = new Uint8Array(n);

    for (let step = 0; step < n; step++) {
      let u = -1;
      let minD = Infinity;
      for (let i = 0; i < n; i++) {
        if (!vis[i] && dist[i]! < minD) { minD = dist[i]!; u = i; }
      }
      if (u < 0) break;
      vis[u] = 1;

      for (let v = 0; v < n; v++) {
        if (v === u) continue;
        const w = weights[u]![v]!;
        if (w > 0) {
          const nd = dist[u]! + 1 / w;
          if (nd < dist[v]!) dist[v] = nd;
        }
        if (!directed) {
          const w2 = weights[v]![u]!;
          if (w2 > 0) {
            const nd2 = dist[u]! + 1 / w2;
            if (nd2 < dist[v]!) dist[v] = nd2;
          }
        }
      }
    }

    let sumDist = 0;
    let reachable = 0;
    for (let i = 0; i < n; i++) {
      if (i !== s && dist[i]! < Infinity) {
        sumDist += dist[i]!;
        reachable++;
      }
    }
    closeness[s] = reachable > 0 ? reachable / sumDist : 0;
  }

  return { inStrength, outStrength, betweenness, closeness };
}
