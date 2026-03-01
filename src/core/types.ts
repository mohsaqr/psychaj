/**
 * All psychaj interfaces — zero runtime code.
 */

// ═══════════════════════════════════════════════════════════
//  Core
// ═══════════════════════════════════════════════════════════

/** Estimation method for psychometric networks. */
export type EstimationMethod = 'cor' | 'pcor' | 'glasso';

/** Community detection method. */
export type CommunityMethod =
  | 'louvain'
  | 'walktrap'
  | 'fast_greedy'
  | 'label_prop'
  | 'leading_eigen'
  | 'edge_betweenness';

// ═══════════════════════════════════════════════════════════
//  Models
// ═══════════════════════════════════════════════════════════

export interface MlVAROptions {
  vars: string[];
  idvar: string;
  dayvar?: string;
  beepvar?: string;
  lag?: number;
  standardize?: boolean;
  gamma?: number;
  computeIndividual?: boolean;
}

export interface MlVARCoef {
  from: string;
  to: string;
  estimate: number;
  se: number;
  tValue: number;
  pValue: number;
  ci: [number, number];
}

export interface MlVARResult {
  temporal: number[][];
  contemporaneous: number[][];
  between: number[][];
  labels: string[];
  coefs: MlVARCoef[];
  nObs: number;
  nSubjects: number;
  formatted: string;
  subjectIds?: string[];
  perSubjectBetas?: Map<string, number[][]>;
  perSubjectNObs?: Map<string, number>;
}

export interface ImpulseResponse {
  trajectories: number[][];
  labels: string[];
  shockedVar: string;
  spectralRadius: number;
}

export interface GraphicalVAROptions {
  vars: string[];
  idvar: string;
  dayvar?: string;
  beepvar?: string;
  lag?: number;
  gamma?: number;
  nLambda?: number;
  scale?: boolean;
  centerWithin?: boolean;
  penalizeDiagonal?: boolean;
}

export interface GraphicalVARResult {
  temporal: number[][];
  contemporaneous: number[][];
  PDC: number[][];
  PCC: number[][];
  beta: number[][];
  kappa: number[][];
  labels: string[];
  nObs: number;
  nSubjects: number;
  lambda_beta: number;
  lambda_kappa: number;
  EBIC: number;
  formatted: string;
}

// ═══════════════════════════════════════════════════════════
//  Validation
// ═══════════════════════════════════════════════════════════

export interface NctOptions {
  method?: EstimationMethod;
  iter?: number;
  alpha?: number;
  gamma?: number;
  paired?: boolean;
  adjust?: 'none' | 'bonferroni' | 'holm' | 'BH';
  seed?: number;
  nodeNames?: string[];
  permutationIndices?: number[][];
}

export interface NctEdgeResult {
  from: string;
  to: string;
  weight_x: number;
  weight_y: number;
  diff: number;
  effect_size: number;
  p_value: number;
  significant: boolean;
}

export interface NctResult {
  obs_diff: number[][];
  p_values: number[][];
  effect_size: number[][];
  sig_diff: number[][];
  net_x: number[][];
  net_y: number[][];
  summary: NctEdgeResult[];
  nodeNames: string[];
  method: EstimationMethod;
  iter: number;
  alpha: number;
  nX: number;
  nY: number;
}

export interface GlassoBootEdge {
  from: string;
  to: string;
  i: number;
  j: number;
  weight: number;
  bootMean: number;
  bootSd: number;
  ciLower: number;
  ciUpper: number;
  significant: boolean;
  bootSamples: Float64Array;
}

export interface GlassoBootDiff {
  edgeA: string;
  edgeB: string;
  diffMean: number;
  diffCiLower: number;
  diffCiUpper: number;
  significant: boolean;
}

export interface BootstrapGlassoResult {
  edges: GlassoBootEdge[];
  pairwiseDiffs: GlassoBootDiff[];
  labels: string[];
  n: number;
  iter: number;
  method: EstimationMethod;
  gamma: number;
  lambda?: number;
}

export interface BootstrapGlassoOptions {
  method?: EstimationMethod;
  gamma?: number;
  rho?: number;
  iter?: number;
  seed?: number;
  ciLevel?: number;
  /**
   * When true, fix the lambda selected on the original data for all
   * bootstrap samples (faster, ~100x for GLASSO). When false (default),
   * re-run EBIC path search per sample (matches R bootnet). */
  fixLambda?: boolean;
}

export interface StabilityResult {
  csCoefficients: Record<string, number>;
  meanCorrelations: Record<string, number[]>;
  dropProps: number[];
  threshold: number;
  certainty: number;
}

export interface StabilityOptions {
  measures?: string[];
  iter?: number;
  dropProps?: number[];
  threshold?: number;
  certainty?: number;
  seed?: number;
  corrMethod?: 'pearson' | 'spearman';
}

// ═══════════════════════════════════════════════════════════
//  Graph
// ═══════════════════════════════════════════════════════════

export interface CommunityResult {
  assignments: number[];
  nCommunities: number;
  method: string;
  modularity: number;
}

export interface CentralityResult {
  inStrength: Float64Array;
  outStrength: Float64Array;
  betweenness: Float64Array;
  closeness: Float64Array;
}

export interface GraphMetrics {
  nodes: number;
  edges: number;
  density: number;
  avgDegree: number;
  avgWeightedDegree: number;
  reciprocity: number | null;
  transitivity: number;
  avgPathLength: number;
  diameter: number;
  components: number;
  largestComponentSize: number;
  selfLoops: number;
}
