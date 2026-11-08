export const LAYERS = 24;
export const INTERMEDIATE = 7680;

const TOKEN = /L(\d+)N(\d+)([+-]\d+(?:\.\d+)?)/g;

export function parseRecipe(text) {
  const cells = [];
  for (const match of String(text).matchAll(TOKEN)) {
    const cell = {
      layer: Number(match[1]),
      neuron: Number(match[2]),
      dose: Number(match[3]),
    };
    if (cell.layer < 0 || cell.layer >= LAYERS || cell.neuron < 0 || cell.neuron >= INTERMEDIATE) {
      throw new Error(`cell outside the model: L${cell.layer}N${cell.neuron}`);
    }
    cells.push(cell);
  }
  if (!cells.length) throw new Error("no recipe cells found");
  return cells;
}

export function formatRecipe(cells) {
  return cells
    .map((cell) => `L${cell.layer}N${cell.neuron}${cell.dose >= 0 ? "+" : ""}${cell.dose}`)
    .join(" ");
}
