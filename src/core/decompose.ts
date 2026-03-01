/**
 * Within-Between decomposition for repeated-measures data.
 */

export function decomposeWithinBetween(
  data: number[][],
  actorIds: string[],
): {
  withinData: number[][];
  betweenData: number[][];
  betweenActorIds: string[];
  hasRepeatedMeasures: boolean;
} {
  const p = data[0]?.length ?? 0;
  const n = data.length;

  const actorMap = new Map<string, number[]>();
  for (let i = 0; i < n; i++) {
    const id = actorIds[i]!;
    if (!actorMap.has(id)) actorMap.set(id, []);
    actorMap.get(id)!.push(i);
  }

  const uniqueActors = [...actorMap.keys()];
  const hasRepeatedMeasures = uniqueActors.some(id => actorMap.get(id)!.length > 1);

  // Between-person: actor means
  const betweenData: number[][] = uniqueActors.map(id => {
    const indices = actorMap.get(id)!;
    const mean = new Array<number>(p).fill(0);
    for (const idx of indices) {
      for (let j = 0; j < p; j++) mean[j]! += data[idx]![j]!;
    }
    for (let j = 0; j < p; j++) mean[j]! /= indices.length;
    return mean;
  });

  const actorMeanIdx = new Map(uniqueActors.map((id, i) => [id, i]));

  // Within-person: row - actor mean
  const withinData: number[][] = data.map((row, i) => {
    const mean = betweenData[actorMeanIdx.get(actorIds[i]!)!]!;
    return row.map((v, j) => v - mean[j]!);
  });

  return { withinData, betweenData, betweenActorIds: uniqueActors, hasRepeatedMeasures };
}
