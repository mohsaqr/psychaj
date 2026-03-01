import { describe, it, expect } from 'vitest';
import { decomposeWithinBetween } from '../../src/core/decompose';

describe('decomposeWithinBetween', () => {
  it('between = actor means, within = residuals', () => {
    const data = [
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ];
    const ids = ['A', 'A', 'B', 'B'];
    const { withinData, betweenData, betweenActorIds, hasRepeatedMeasures } = decomposeWithinBetween(data, ids);

    expect(hasRepeatedMeasures).toBe(true);
    expect(betweenActorIds).toEqual(['A', 'B']);
    expect(betweenData).toEqual([[2, 3], [6, 7]]);

    // Within = data - actor mean
    expect(withinData[0]).toEqual([-1, -1]); // [1,2] - [2,3]
    expect(withinData[1]).toEqual([1, 1]);   // [3,4] - [2,3]
    expect(withinData[2]).toEqual([-1, -1]); // [5,6] - [6,7]
    expect(withinData[3]).toEqual([1, 1]);   // [7,8] - [6,7]
  });

  it('no repeated measures when all IDs unique', () => {
    const data = [[1, 2], [3, 4]];
    const ids = ['A', 'B'];
    const { hasRepeatedMeasures } = decomposeWithinBetween(data, ids);
    expect(hasRepeatedMeasures).toBe(false);
  });
});
