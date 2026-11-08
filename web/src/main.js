import { coverage, familyCounts, filterEntries, layerType } from "./catalog.js";
import { formatRecipe, parseRecipe } from "./recipe-format.js";
import { encodeState, readHash } from "./state.js";
import { readStats, steerRows } from "./steer.js";

const LAYERS = 24;
const INTERMEDIATE = 7680;
const SAMPLED = [5, 11, 17, 8, 12, 16, 10, 14, 18];

const state = {
  catalog: { entries: [] },
  heatmap: null,
  stats: null,
  cells: [],
  prompt: "Who are you?",
  model: null,
  busy: false,
};

const el = (id) => document.getElementById(id);

async function loadArtifacts() {
  try {
    const response = await fetch("./catalog.json");
    if (!response.ok) throw new Error(String(response.status));
    state.catalog = await response.json();
  } catch {
    el("finding").textContent = "Catalog not published yet — run the pipeline to generate it.";
  }
  try {
    const response = await fetch("./heatmap.bin");
    if (!response.ok) throw new Error(String(response.status));
    state.heatmap = new Uint8Array(await response.arrayBuffer());
  } catch {
    state.heatmap = null;
  }
  try {
    const response = await fetch("./steer_stats.bin");
    if (response.ok) state.stats = readStats(await response.arrayBuffer());
  } catch {
    state.stats = null;
  }
}

function setFinding() {
  const finding = state.catalog.finding;
  if (!finding) return;
  const parts = Object.entries(finding.by_layer_type).map(
    ([kind, value]) => `${kind.replace("_", " ")} ${Math.round(value.share * 100)}%`,
  );
  el("finding").textContent = `Share of high-effect neurons: ${parts.join(" · ")}`;
}

function recipeCells() {
  try {
    return parseRecipe(el("recipe").value);
  } catch {
    return [];
  }
}

function syncUrl() {
  const token = encodeState({ cells: recipeCells(), prompt: el("prompt").value });
  history.replaceState(null, "", `#${token}`);
}

function renderGallery() {
  const family = el("family").value;
  const layerTypeFilter = el("layer-type").value;
  const entries = filterEntries(state.catalog.entries ?? [], { family, layerTypeFilter });
  const gallery = el("gallery");
  gallery.replaceChildren();

  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No recipes match these filters yet.";
    gallery.append(empty);
    return;
  }

  for (const entry of entries) {
    const card = document.createElement("article");
    card.className = "card";
    const title = document.createElement("h3");
    title.textContent = `${entry.family} · L${entry.layer}N${entry.neuron} ${entry.dose > 0 ? "+" : ""}${entry.dose}`;
    const meta = document.createElement("p");
    meta.className = "muted";
    meta.textContent = `${layerType(entry.layer).replace("_", " ")} · score ${entry.score}`;
    const sample = document.createElement("pre");
    sample.textContent = entry.text_baseline
      ? `baseline: ${entry.text_baseline}\nsteered: ${entry.text}`
      : entry.text;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Use this recipe";
    button.addEventListener("click", () => {
      el("recipe").value = formatRecipe([{ layer: entry.layer, neuron: entry.neuron, dose: entry.dose }]);
      syncUrl();
    });
    card.append(title, meta, sample, button);
    gallery.append(card);
  }
}

function renderHeatmap() {
  const canvas = el("heatmap");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const image = context.createImageData(width, height);
  const rows = state.heatmap?.length === SAMPLED.length * INTERMEDIATE ? SAMPLED : null;
  const sampled = new Set(SAMPLED);

  for (let y = 0; y < height; y += 1) {
    const layer = Math.floor((y * LAYERS) / height);
    const rowIndex = rows ? rows.indexOf(layer) : -1;
    for (let x = 0; x < width; x += 1) {
      const neuron = Math.floor((x * INTERMEDIATE) / width);
      const offset = (y * width + x) * 4;
      let color = [120, 120, 130, 255];
      if (rowIndex >= 0) {
        const value = state.heatmap[rowIndex * INTERMEDIATE + neuron];
        color = [40 + value, 40, 200 - value, 255];
      } else if (sampled.has(layer)) {
        color = [90, 90, 100, 255];
      } else {
        color = [60, 60, 66, 255];
      }
      image.data.set(color, offset);
    }
  }
  context.putImageData(image, 0, 0);
}

function addCellFromCanvas(event) {
  const canvas = el("heatmap");
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
  const layer = Math.floor((y * LAYERS) / canvas.height);
  const neuron = Math.floor((x * INTERMEDIATE) / canvas.width);
  el("recipe").value = formatRecipe([{ layer, neuron, dose: 1 }]);
  syncUrl();
  if (!SAMPLED.includes(layer)) {
    el("status").textContent = `Layer ${layer} was not sampled — no recorded output, but it can still be steered.`;
    return;
  }
  el("status").textContent = `Selected L${layer}N${neuron}.`;
}

async function run() {
  if (state.busy) return;
  const cells = recipeCells();
  if (!cells.length) {
    el("status").textContent = "Enter a recipe first, for example L14N4949+1.";
    return;
  }
  state.busy = true;
  el("run").disabled = true;
  el("baseline").textContent = "";
  el("steered").textContent = "";
  try {
    el("status").textContent = "Loading the 1.38 GB model…";
    state.stats ??= readStats(await (await fetch("./steer_stats.bin")).arrayBuffer());
    const { generate, loadModel } = await import("./onnx.js");
    state.model ??= await loadModel({
      onProgress: (message) => {
        el("status").textContent = message === "ready" ? "Model ready." : message;
      },
    });

    el("status").textContent = "Running baseline…";
    await generate(state.model, {
      prompt: el("prompt").value,
      rows: null,
      onToken: (text) => {
        el("baseline").textContent += text;
      },
    });

    el("status").textContent = "Running steered…";
    const rows = steerRows(state.stats, cells);
    await generate(state.model, {
      prompt: el("prompt").value,
      rows,
      onToken: (text) => {
        el("steered").textContent += text;
      },
    });
    el("status").textContent = "Done.";
  } catch (error) {
    el("status").textContent = String(error.message ?? error);
  } finally {
    state.busy = false;
    el("run").disabled = false;
  }
}

function restoreFromHash() {
  const restored = readHash(location.hash);
  if (!restored) return;
  state.cells = restored.cells;
  el("recipe").value = formatRecipe(restored.cells);
  el("prompt").value = restored.prompt;
}

async function main() {
  await loadArtifacts();
  setFinding();
  renderGallery();
  renderHeatmap();

  const counts = familyCounts(state.catalog.entries ?? []);
  for (const family of Object.keys(counts).sort()) {
    const option = document.createElement("option");
    option.value = family;
    option.textContent = `${family} (${counts[family]})`;
    el("family").append(option);
  }

  const gaps = coverage(state.catalog.entries ?? []).filter((row) => !row.sampled).length;
  if (!state.catalog.entries?.length) {
    el("finding").textContent = `No catalog yet — ${gaps} of 24 layers are unsampled.`;
  }

  el("family").addEventListener("change", renderGallery);
  el("layer-type").addEventListener("change", renderGallery);
  el("recipe").addEventListener("input", syncUrl);
  el("prompt").addEventListener("input", syncUrl);
  el("heatmap").addEventListener("click", addCellFromCanvas);
  el("import-recipe").addEventListener("click", () => {
    try {
      el("recipe").value = formatRecipe(parseRecipe(el("recipe").value));
      el("status").textContent = "Recipe imported.";
      syncUrl();
    } catch (error) {
      el("status").textContent = String(error.message ?? error);
    }
  });
  el("run").addEventListener("click", run);
  el("run").disabled = false;

  restoreFromHash();
}

main();
