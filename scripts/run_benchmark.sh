#!/usr/bin/env bash
# Benchmark : plusieurs architectures en parallèle sur le gros dataset,
# puis évaluation de chacune sur test.txt (jamais vu pendant l'apprentissage).
#
# Usage (depuis la racine du dépôt) :
#   scripts/run_benchmark.sh                 # tout lancer
#   JOBS="a128_64 a256_128_64" scripts/run_benchmark.sh   # seulement certains jobs
# Suivre : tail -f benchmark_runs/<date>/*/train.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---------- Réglages ----------
DATA="${DATA:-dataset/benchmark_77k}"   # généré par scripts/generate_dataset.py
MAX_EPOCHS="${MAX_EPOCHS:-400}"
PATIENCE="${PATIENCE:-25}"
MIN_DELTA="${MIN_DELTA:-1e-6}"
PARALLEL="${PARALLEL:-5}"           # ~1,5 Go de RAM par job
PY="$ROOT/.venv/bin/python"

# nom:couches cachées (séparées par des virgules):learning_rate
ALL_JOBS=(
  "continue_88:keep:keep"          # reprend my_torch_network_1.nn (le modèle à 88 %)
  "a128_64:128,64:0.01"            # même archi que le modèle livré, à neuf (référence)
  "a256_128_64:256,128,64:0.01"
  "a64:64:0.01"
  "a64_64_64:64,64,64:0.01"
  "a128:128:0.01"
)

# ---------- Empêcher la veille ----------
if [[ -z "${BENCHMARK_INHIBITED:-}" ]] && command -v systemd-inhibit >/dev/null; then
  export BENCHMARK_INHIBITED=1
  exec systemd-inhibit --what=sleep:idle:handle-lid-switch \
       --who=my_torch_benchmark --why="Entraînement my_torch" "$0" "$@"
fi

DATA="$(realpath "$DATA")"
[[ -x "$PY" ]] || { echo "Venv manquant : python3 -m venv .venv && .venv/bin/pip install numpy" >&2; exit 1; }
[[ -f "$DATA/train.txt" && -f "$DATA/test.txt" ]] || { echo "Dataset manquant : $DATA" >&2; exit 1; }

make -s >/dev/null
RUN_DIR="$ROOT/benchmark_runs/$(date +%Y-%m-%d_%H%M)"
mkdir -p "$RUN_DIR"
echo "Résultats dans $RUN_DIR"

# Un seul thread BLAS par job : sinon les jobs parallèles se marchent dessus.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

run_job() {
  local spec="$1" name layers lr dir
  IFS=: read -r name layers lr <<<"$spec"
  dir="$RUN_DIR/$name"
  mkdir -p "$dir"
  cd "$dir"

  if [[ "$layers" == keep ]]; then
    cp "$ROOT/my_torch_network_1.nn" start.nn
  else
    "$PY" - "$ROOT/tests/conf_example/chess_network.conf" "$layers" "$lr" <<'EOF'
import json, sys
conf = json.load(open(sys.argv[1]))
conf.update(name="start", layers=[838, *map(int, sys.argv[2].split(",")), 5],
            learning_rate=float(sys.argv[3]))
json.dump(conf, open("start.conf", "w"), indent=2)
EOF
    "$ROOT/my_torch_generator" start.conf 1 >/dev/null
    mv start_1.nn start.nn
  fi

  # max_epochs / patience / min_delta sont lus DANS le .nn : on les écrit dedans.
  "$PY" - start.nn "$MAX_EPOCHS" "$PATIENCE" "$MIN_DELTA" <<'EOF'
import json, sys
path = sys.argv[1]
nn = json.load(open(path))
nn.update(max_epochs=int(sys.argv[2]), patience=int(sys.argv[3]), min_delta=float(sys.argv[4]))
json.dump(nn, open(path, "w"))
EOF

  local start=$SECONDS status=0
  "$PY" "$ROOT/my_torch_analyzer" --train --save trained.nn start.nn "$DATA/train.txt" \
      > train.log 2>&1 || status=$?
  echo "exit=$status duration=$(( (SECONDS - start) / 60 ))min" >> train.log
  [[ -f trained.nn ]] || { echo "$name: échec, voir $dir/train.log"; return; }

  "$PY" "$ROOT/my_torch_analyzer" --predict trained.nn "$DATA/test.txt" > test_predictions.txt 2>/dev/null
  "$PY" - "$DATA/test.txt" test_predictions.txt "$name" > score.txt <<'EOF'
import sys
from collections import Counter
truth = [" ".join(l.split()[6:]) for l in open(sys.argv[1]) if l.strip()]
pred = [l.strip() for l in open(sys.argv[2])]
assert len(truth) == len(pred), f"{len(truth)} labels vs {len(pred)} prédictions"
ok, tot = Counter(), Counter(truth)
for t, p in zip(truth, pred):
    ok[t] += t == p
acc = sum(ok.values()) / len(truth)
print(f"{acc * 100:.2f} %  {sys.argv[3]}")
for label in sorted(tot):
    print(f"         {label:<16} {ok[label] / tot[label] * 100:6.2f} %")
EOF
  cat score.txt
}
export -f run_job
export ROOT RUN_DIR DATA MAX_EPOCHS PATIENCE MIN_DELTA PY

selected=()
for spec in "${ALL_JOBS[@]}"; do
  [[ -z "${JOBS:-}" || " $JOBS " == *" ${spec%%:*} "* ]] && selected+=("$spec")
done

printf '%s\n' "${selected[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'run_job "$1"' _ {}

echo
echo "===== Classement (accuracy sur $DATA/test.txt) ====="
cat "$RUN_DIR"/*/score.txt 2>/dev/null | grep -v '^ ' | sort -rn | tee "$RUN_DIR/summary.txt"
echo "(continue_88 a peut-être déjà vu des positions du test : son score peut être gonflé)"
