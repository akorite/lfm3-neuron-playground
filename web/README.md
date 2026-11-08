# LFM3-1.7B neuron playground — web app

Static browser app. The catalog and coverage map load from four published
artifacts; the 1.38 GB model downloads only when a visitor runs a recipe.

## Artifacts

Published together at one pinned revision, next to the app:

| File | Produced by | Size |
|---|---|---|
| `lfm3-steer.onnx` | `tools/build_steer_graph.py` | ~266 KB |
| `steer_stats.bin` | `tools/build_stats.py` | ~737 KB |
| `catalog.json` | `research/build_catalog.py` | tens of KB |
| `heatmap.bin` | `research/build_catalog.py` | 69,120 B |
| `recipes.json` | `research/build_catalog.py` | a few KB |

The weight shards are **not** republished: `onnx.js` streams
`model_q4f16.onnx_data` from `LiquidAI/LFM3-1.7B-ONNX` through ort-web's
`externalData` option, keyed on the protobuf `location`, from the immutable
revision `57c125dfa29f2d39157e280b1d1f7480b8736d8d`. The tokenizer comes from
`LiquidAI/LFM3-1.7B` at `0dc4268b561499c648c7ba7c9466bf3d8d01517a`. Both pins
live at the top of `src/onnx.js`; bumping either changes what every published
number means, so bump them together with a re-run of the pipeline.

## Build and deploy

```bash
npm ci            # exact lockfile install
npm test          # unit tests: recipe text, URL state, dose math, filters
npm run build     # emits dist/
```

Serve `dist/` with `web/public/` artifacts copied in, or point `GRAPH_URL`
and the fetches at the published revision. Any static host works; no server
code and no secrets.

## Runtime requirements

- Chrome 121+ or another browser with WebGPU and `shader-f16`.
- The session is created with `session.disable_cpu_ep_fallback`, so a node
  left on the CPU aborts instead of silently running at CPU speed.
- Prefill is chunked at 256 tokens; the graph is pinned to the revision whose
  node names the steering edit targets.

## Spend guard

Every GPU job runs through `colab run` (the VM releases itself) or a session
that is stopped the moment its job ends. An idle A100 burns about 8 CU per
hour.

## License

LFM3 is released under the LFM Open License v1.0. Commercial use is licensed
only for entities with annual revenue under $10 million.
