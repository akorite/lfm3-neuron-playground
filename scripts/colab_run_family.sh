#!/bin/sh
# Verify one family's deep-rollout candidates on the shipped graph.
# Usage: colab_run_family.sh <family> <deep.jsonl> [<cells.json>]
# With no cells file, the deep records are used directly.
set -e
FAMILY="$1"
DEEP="$2"
CELLS="$3"
cd /content/lfm3

if [ -n "$CELLS" ]; then
  python research/deepen.py --cells "$CELLS" --out "$DEEP" --stats web/public/steer_stats.bin --tokens 64
fi

python - "$FAMILY" "$DEEP" > "/content/build_${FAMILY}.log" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

from research.model_info import layer_type
from research.score import fires, load_family, score

family, deep = sys.argv[1], sys.argv[2]
spec = load_family(family)
entries = []
for line in open(deep):
    row = json.loads(line)
    value = score(row["text"], spec)
    if not fires(value, spec):
        continue
    entries.append({
        "family": family, "layer": row["layer"], "neuron": row["neuron"],
        "dose": row["sign"], "layer_type": layer_type(row["layer"]),
        "score": round(value, 4), "text": row["text"],
    })
Path(f"{family}_catalog.json").write_text(json.dumps({
    "entries": entries, "count": len(entries), "families": [family],
    "layer_types": sorted({e["layer_type"] for e in entries}), "finding": {},
}, indent=1))
print(f"{family} candidates", len(entries))
PY

python research/verify.py --graph graph/lfm3-steer.onnx --catalog "${FAMILY}_catalog.json" \
  --out "${FAMILY}_verified.json" --stats web/public/steer_stats.bin --tokens 48 \
  > "/content/verify_${FAMILY}.log" 2>&1
echo done > "/content/${FAMILY}.done"
