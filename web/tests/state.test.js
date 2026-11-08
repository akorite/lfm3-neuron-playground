import { describe, expect, it } from "vitest";

import { decodeState, encodeState, readHash } from "../src/state.js";

const state = {
  cells: [
    { layer: 14, neuron: 4949, dose: 1 },
    { layer: 13, neuron: 3956, dose: -0.5 },
  ],
  prompt: "Who are you?",
};

describe("url state", () => {
  it("round-trips cells, doses, and prompt", () => {
    expect(decodeState(encodeState(state))).toEqual(state);
  });

  it("reads a hash with a leading marker", () => {
    expect(readHash(`#${encodeState(state)}`)).toEqual(state);
  });

  it("returns null for garbage and empty input", () => {
    expect(decodeState("")).toBeNull();
    expect(decodeState("not-base64")).toBeNull();
    expect(decodeState(encodeState({ cells: [], prompt: "" }))).toBeNull();
  });

  it("drops invalid cells instead of trusting them", () => {
    const token = encodeState({ cells: [{ layer: 99, neuron: 1, dose: 1 }], prompt: "x" });
    expect(decodeState(token)).toBeNull();
  });
});
