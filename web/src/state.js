import { LAYERS, INTERMEDIATE } from "./recipe-format.js";

function toBase64Url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(token) {
  const padded = token.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(padded.padEnd(padded.length + ((4 - (padded.length % 4)) % 4), "="));
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}

export function encodeState({ cells, prompt }) {
  const body = JSON.stringify({ c: cells, p: prompt ?? "" });
  return toBase64Url(new TextEncoder().encode(body));
}

export function decodeState(token) {
  if (!token) return null;
  try {
    const parsed = JSON.parse(new TextDecoder().decode(fromBase64Url(token)));
    if (!Array.isArray(parsed.c)) return null;
    const cells = parsed.c.filter(
      (cell) =>
        Number.isInteger(cell.layer) &&
        Number.isInteger(cell.neuron) &&
        cell.layer >= 0 &&
        cell.layer < LAYERS &&
        cell.neuron >= 0 &&
        cell.neuron < INTERMEDIATE &&
        Number.isFinite(cell.dose),
    );
    if (!cells.length) return null;
    return { cells, prompt: typeof parsed.p === "string" ? parsed.p : "" };
  } catch {
    return null;
  }
}

export function readHash(hash) {
  return decodeState(String(hash ?? "").replace(/^#/, ""));
}
