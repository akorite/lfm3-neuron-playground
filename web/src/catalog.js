export const LAYER_TYPES = [
  "conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
  "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
  "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
  "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
];

export function layerType(layer) {
  return LAYER_TYPES[layer] ?? "unknown";
}

export function filterEntries(entries, { family = "all", layerTypeFilter = "all" } = {}) {
  return entries.filter((entry) => {
    const familyOk = family === "all" || entry.family === family;
    const typeOk = layerTypeFilter === "all" || layerType(entry.layer) === layerTypeFilter;
    return familyOk && typeOk;
  });
}

export function familyCounts(entries) {
  const counts = {};
  for (const entry of entries) counts[entry.family] = (counts[entry.family] ?? 0) + 1;
  return counts;
}

export function coverage(entries) {
  const covered = new Set(entries.map((entry) => entry.layer));
  return LAYER_TYPES.map((type, layer) => ({
    layer,
    type,
    sampled: covered.has(layer),
  }));
}
