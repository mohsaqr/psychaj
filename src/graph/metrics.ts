/**
 * Network-level graph metrics computed from a weight matrix.
 * All functions take number[][] — no tnaj dependency.
 */

import type { GraphMetrics } from '../core/types';

/**
 * Compute network-level metrics from a weight matrix.
 *
 * @param weights      n×n weight matrix
 * @param directed     if false, treats graph as undirected (default: true)
 */
export function computeGraphMetrics(
  weights: number[][],
  directed = true,
): GraphMetrics {
  const n = weights.length;

  // Count edges, self-loops
  let selfLoops = 0;
  let edgeCount = 0;
  for (let i = 0; i < n; i++) {
    if (weights[i]![i]! > 0) selfLoops++;
    for (let j = 0; j < n; j++) {
      if (i !== j && weights[i]![j]! > 0) edgeCount++;
    }
  }
  const edges = directed ? edgeCount : edgeCount / 2;

  // Density
  const maxEdges = directed ? n * (n - 1) : n * (n - 1) / 2;
  const density = maxEdges > 0 ? edges / maxEdges : 0;

  // Degree
  const avgDegree = n > 0 ? (directed ? edges / n : 2 * edges / n) : 0;

  // Weighted degree
  let totalWeightedDeg = 0;
  for (let i = 0; i < n; i++) {
    let outS = 0, inS = 0;
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        outS += weights[i]![j]!;
        inS += weights[j]![i]!;
      }
    }
    totalWeightedDeg += (outS + inS) / 2;
  }
  const avgWeightedDegree = n > 0 ? totalWeightedDeg / n : 0;

  // Reciprocity (directed only)
  let reciprocity: number | null = null;
  if (directed) {
    let mutual = 0;
    let total = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j && weights[i]![j]! > 0) {
          total++;
          if (weights[j]![i]! > 0) mutual++;
        }
      }
    }
    reciprocity = total > 0 ? mutual / total : 0;
  }

  // Transitivity: global clustering coefficient
  const adj: boolean[][] = Array.from({ length: n }, () => new Array(n).fill(false));
  const deg: number[] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j && (weights[i]![j]! > 0 || weights[j]![i]! > 0)) {
        adj[i]![j] = true;
      }
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (adj[i]![j]) deg[i]++;
    }
  }

  let triangles = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (!adj[i]![j]) continue;
      for (let k = j + 1; k < n; k++) {
        if (adj[i]![k] && adj[j]![k]) triangles++;
      }
    }
  }
  let triples = 0;
  for (let i = 0; i < n; i++) {
    triples += deg[i]! * (deg[i]! - 1) / 2;
  }
  const transitivity = triples > 0 ? (3 * triangles) / triples : 0;

  // Floyd-Warshall for shortest paths
  const INF = Infinity;
  const dist: number[][] = Array.from({ length: n }, () => new Array(n).fill(INF));
  for (let i = 0; i < n; i++) dist[i]![i] = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        const wij = weights[i]![j]!;
        if (wij > 0) dist[i]![j] = 1 / wij;
        if (!directed) {
          const wji = weights[j]![i]!;
          if (wji > 0) {
            const d = 1 / wji;
            if (d < dist[i]![j]!) dist[i]![j] = d;
          }
        }
      }
    }
  }
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const via = dist[i]![k]! + dist[k]![j]!;
        if (via < dist[i]![j]!) dist[i]![j] = via;
      }
    }
  }

  let sumPath = 0;
  let countPath = 0;
  let maxPath = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j && dist[i]![j]! < INF) {
        sumPath += dist[i]![j]!;
        countPath++;
        if (dist[i]![j]! > maxPath) maxPath = dist[i]![j]!;
      }
    }
  }
  const avgPathLength = countPath > 0 ? sumPath / countPath : 0;
  const diameter = maxPath;

  // Weakly connected components
  const visited = new Array(n).fill(false);
  let components = 0;
  let largestComponentSize = 0;

  for (let start = 0; start < n; start++) {
    if (visited[start]) continue;
    components++;
    let size = 0;
    const queue = [start];
    visited[start] = true;
    while (queue.length > 0) {
      const v = queue.shift()!;
      size++;
      for (let u = 0; u < n; u++) {
        if (!visited[u] && adj[v]![u]) {
          visited[u] = true;
          queue.push(u);
        }
      }
    }
    if (size > largestComponentSize) largestComponentSize = size;
  }

  return {
    nodes: n,
    edges,
    density,
    avgDegree,
    avgWeightedDegree,
    reciprocity,
    transitivity,
    avgPathLength,
    diameter,
    components,
    largestComponentSize,
    selfLoops,
  };
}
