import { describe, expect, it } from "vitest";

import { coverage, familyCounts, filterEntries, layerType } from "../src/catalog.js";

const entries = [
  { family: "personas", layer: 5 },
  { family: "style", layer: 8 },
  { family: "personas", layer: 10 },
];

describe("catalog", () => {
  it("labels layers by type", () => {
    expect(layerType(5)).toBe("conv");
    expect(layerType(8)).toBe("sliding_attention");
    expect(layerType(10)).toBe("full_attention");
  });

  it("filters by family and layer type", () => {
    expect(filterEntries(entries, { family: "personas" })).toHaveLength(2);
    expect(filterEntries(entries, { layerTypeFilter: "sliding_attention" })).toHaveLength(1);
    expect(filterEntries(entries, { family: "personas", layerTypeFilter: "conv" })).toHaveLength(1);
  });

  it("counts families", () => {
    expect(familyCounts(entries)).toEqual({ personas: 2, style: 1 });
  });

  it("marks unsampled layers as coverage gaps", () => {
    const rows = coverage(entries);
    expect(rows).toHaveLength(24);
    expect(rows.find((row) => row.layer === 5).sampled).toBe(true);
    expect(rows.find((row) => row.layer === 0).sampled).toBe(false);
  });
});
