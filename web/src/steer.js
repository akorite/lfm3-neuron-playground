export const LAYERS = 24;
export const INTERMEDIATE = 7680;

export function readStats(buffer) {
  const values = new Float32Array(buffer);
  if (values.length !== LAYERS + LAYERS * INTERMEDIATE) {
    throw new Error(`expected ${LAYERS + LAYERS * INTERMEDIATE} floats, got ${values.length}`);
  }
  return {
    thresholds: values.slice(0, LAYERS),
    norms: values.slice(LAYERS),
  };
}

export function delta(stats, layer, neuron, dose) {
  const threshold = stats.thresholds[layer];
  const norm = stats.norms[layer * INTERMEDIATE + neuron];
  if (!(threshold > 0) || !(norm > 0)) throw new Error(`bad stats at L${layer}N${neuron}`);
  return (dose * threshold) / norm;
}

export function steerRow(stats, cells, layer) {
  const row = new Float32Array(INTERMEDIATE);
  for (const cell of cells) {
    if (cell.layer === layer) row[cell.neuron] += delta(stats, cell.layer, cell.neuron, cell.dose);
  }
  return row;
}

export function steerRows(stats, cells) {
  return Array.from({ length: LAYERS }, (_, layer) => steerRow(stats, cells, layer));
}
