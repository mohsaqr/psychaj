/**
 * Partial Contemporaneous Correlations (PCC).
 * PCC = -cov2cor(kappa); diag = 0
 * Identical to thetaToPcor().
 */

import { thetaToPcor } from '../estimation/theta-to-pcor';

export function computePCC(kappa: number[][]): number[][] {
  return thetaToPcor(kappa);
}
