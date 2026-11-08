import { describe, expect, it } from "vitest";

import { INTERMEDIATE, LAYERS, delta, readStats, steerRow } from "../src/steer.js";

function statsBuffer(thresholds, norms) {
  return new Float32Array([...thresholds, ...norms]).buffer;
}

function makeStats() {
  const thresholds = new Float32Array(LAYERS).fill(10);
  const norms = new Float32Array(LAYERS * INTERMEDIATE).fill(2);
  return readStats(statsBuffer(thresholds, norms));
}

describe("steering math", () => {
  it("rejects a buffer that is not the stats layout", () => {
    expect(() => readStats(new Float32Array(10).buffer)).toThrow(/expected/);
  });

  it("applies dose times threshold over column norm", () => {
    const stats = makeStats();
    expect(delta(stats, 11, 42, 1)).toBeCloseTo(5);
    expect(delta(stats, 11, 42, -0.5)).toBeCloseTo(-2.5);
  });

  it("writes one row per layer with zeros elsewhere", () => {
    const stats = makeStats();
    const row = steerRow(stats, [{ layer: 11, neuron: 42, dose: 1 }], 11);
    expect(row[42]).toBeCloseTo(5);
    expect(row[41]).toBe(0);
    expect(row.reduce((sum, value) => sum + Math.abs(value), 0)).toBeCloseTo(5);
  });

  it("sums several neurons on the same layer", () => {
    const stats = makeStats();
    const row = steerRow(
      stats,
      [
        { layer: 11, neuron: 42, dose: 1 },
        { layer: 11, neuron: 42, dose: 0.5 },
      ],
      11,
    );
    expect(row[42]).toBeCloseTo(7.5);
  });
});
