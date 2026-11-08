const GRAPH_URL = "lfm3-steer.onnx";
const ONNX_REVISION = "57c125dfa29f2d39157e280b1d1f7480b8736d8d";
const TOKENIZER_REVISION = "0dc4268b561499c648c7ba7c9466bf3d8d01517a";
const DATA_URL = `https://huggingface.co/LiquidAI/LFM3-1.7B-ONNX/resolve/${ONNX_REVISION}/onnx/model_q4f16.onnx_data`;
const MAX_PROMPT_TOKENS = 1024;
const CONV_LAYERS = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
const KV_LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22];
const LAYERS = 24;
const INTERMEDIATE = 7680;
const HIDDEN = 2048;
const CHUNK = 256;

export const isSupported = () => typeof navigator !== "undefined" && "gpu" in navigator;

export async function loadModel({ onProgress } = {}) {
  if (!isSupported()) throw new Error("This browser has no WebGPU. Use Chrome 121 or newer.");
  const ort = await import("onnxruntime-web/webgpu");
  ort.env.wasm.numThreads = 1;
  const externalData = [{ path: "model_q4f16.onnx_data", data: DATA_URL }];
  const session = await ort.InferenceSession.create(GRAPH_URL, {
    executionProviders: ["webgpu"],
    externalData,
    extra: { session: { disable_cpu_ep_fallback: "1" } },
  });
  const { AutoTokenizer, env } = await import("@huggingface/transformers");
  env.allowLocalModels = false;
  const tokenizer = await AutoTokenizer.from_pretrained("LiquidAI/LFM3-1.7B", {
    revision: TOKENIZER_REVISION,
  });
  onProgress?.("ready");
  return { ort, session, tokenizer };
}

function f32ToF16(value) {
  const float = new Float32Array(1);
  const bits = new Uint32Array(float.buffer);
  float[0] = value;
  const x = bits[0];
  const sign = (x >> 16) & 0x8000;
  let exponent = (x >> 23) & 0xff;
  let mantissa = x & 0x7fffff;
  if (exponent === 0xff) return sign | (mantissa ? 0x7e00 : 0x7c00);
  let half = exponent - 127 + 15;
  if (half >= 0x1f) return sign | 0x7c00;
  if (half <= 0) {
    if (half < -10) return sign;
    mantissa = (mantissa | 0x800000) >> (1 - half);
    return sign | (mantissa >> 13);
  }
  return sign | (half << 10) | (mantissa >> 13);
}

function steerTensors(ort, rows) {
  const tensors = {};
  for (let layer = 0; layer < LAYERS; layer += 1) {
    const half = new Uint16Array(INTERMEDIATE);
    const row = rows[layer];
    for (let neuron = 0; neuron < INTERMEDIATE; neuron += 1) {
      if (row && row[neuron] !== 0) half[neuron] = f32ToF16(row[neuron]);
    }
    tensors[`steer.${layer}`] = new ort.Tensor("float16", half, [1, 1, INTERMEDIATE]);
  }
  return tensors;
}

function initialCaches() {
  const caches = {};
  for (const layer of CONV_LAYERS) caches[`conv.${layer}`] = { data: new Uint16Array(HIDDEN * 7), length: HIDDEN * 7 };
  for (const layer of KV_LAYERS) {
    caches[`kv.${layer}.key`] = { data: new Uint16Array(0), length: 0 };
    caches[`kv.${layer}.value`] = { data: new Uint16Array(0), length: 0 };
  }
  return caches;
}

function feedFor(ort, ids, position, caches, steer) {
  const length = ids.length;
  const feeds = {
    input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, length]),
    attention_mask: new ort.Tensor("int64", new BigInt64Array(position + length).fill(1n), [1, position + length]),
    position_ids: new ort.Tensor(
      "int64",
      BigInt64Array.from({ length }, (_, index) => BigInt(position + index)),
      [1, length],
    ),
    logits_to_keep: new ort.Tensor("int64", BigInt64Array.from([1n]), [1]),
    ...steer,
  };
  for (const [key, cache] of Object.entries(caches)) {
    if (key.startsWith("conv.")) {
      const layer = key.slice(5);
      feeds[`past_conv.${layer}`] = new ort.Tensor("float16", cache.data, [1, HIDDEN, 7]);
    } else {
      const [, layer, part] = key.split(".");
      const past = cache.length / (4 * 128);
      feeds[`past_key_values.${layer}.${part}`] = new ort.Tensor("float16", cache.data, [1, 4, past, 128]);
    }
  }
  return feeds;
}

function updateCaches(ort, results, caches) {
  for (const layer of CONV_LAYERS) {
    const tensor = results[`present_conv.${layer}`];
    caches[`conv.${layer}`] = { data: tensor.data, length: tensor.data.length };
  }
  for (const layer of KV_LAYERS) {
    for (const part of ["key", "value"]) {
      const tensor = results[`present.${layer}.${part}`];
      if (!tensor) throw new Error(`graph did not return present.${layer}.${part}`);
      caches[`kv.${layer}.${part}`] = { data: tensor.data, length: tensor.data.length };
    }
  }
}

function argmax(logits) {
  let best = 0;
  for (let index = 1; index < logits.length; index += 1) if (logits[index] > logits[best]) best = index;
  return best;
}

const escapeSpecials = (text) => String(text).replace(/<\|/g, "< |");

const wrap = (prompt) =>
  `<|startoftext|><|im_start|>user\n${escapeSpecials(prompt)}<|im_end|>\n<|im_start|>assistant\n<think></think>`;

export async function generate(model, { prompt, rows, maxTokens = 48, onToken }) {
  const { ort, session, tokenizer } = model;
  const encoded = await tokenizer(wrap(prompt), { add_special_tokens: false });
  const ids = Array.from(encoded.input_ids.data, Number);
  if (!ids.length) throw new Error("The prompt encoded to no tokens.");
  if (ids.length > MAX_PROMPT_TOKENS) {
    throw new Error(`Prompt is ${ids.length} tokens; the cap is ${MAX_PROMPT_TOKENS}.`);
  }
  const steer = steerTensors(ort, rows);
  let caches = initialCaches();
  let position = 0;
  let logits = null;

  for (let start = 0; start < ids.length; start += CHUNK) {
    const chunk = ids.slice(start, start + CHUNK);
    const results = await session.run(feedFor(ort, chunk, position, caches, steer));
    updateCaches(ort, results, caches);
    position += chunk.length;
    logits = results.logits.data;
  }

  const produced = [];
  for (let step = 0; step < maxTokens; step += 1) {
    const next = argmax(logits);
    produced.push(next);
    const text = await tokenizer.decode([next], { skip_special_tokens: true });
    onToken?.(text);
    const results = await session.run(feedFor(ort, [next], position, caches, steer));
    updateCaches(ort, results, caches);
    position += 1;
    logits = results.logits.data;
  }
  return produced;
}
