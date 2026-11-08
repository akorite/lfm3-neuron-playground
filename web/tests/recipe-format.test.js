import { describe, expect, it } from "vitest";

import { formatRecipe, parseRecipe } from "../src/recipe-format.js";

describe("recipe text", () => {
  it("round-trips a two-neuron recipe", () => {
    const cells = parseRecipe("L14N4949+1 L13N3956-0.5");
    expect(cells).toEqual([
      { layer: 14, neuron: 4949, dose: 1 },
      { layer: 13, neuron: 3956, dose: -0.5 },
    ]);
    expect(formatRecipe(cells)).toBe("L14N4949+1 L13N3956-0.5");
  });

  it("rejects empty text and out-of-range cells", () => {
    expect(() => parseRecipe("nothing here")).toThrow(/no recipe cells/);
    expect(() => parseRecipe("L24N0+1")).toThrow(/outside the model/);
    expect(() => parseRecipe("L0N7680+1")).toThrow(/outside the model/);
  });
});
