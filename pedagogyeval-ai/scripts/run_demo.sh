#!/usr/bin/env bash
# One-command, offline launch of the PedagogyEval AI demo.
#   bash scripts/run_demo.sh            (PORT=8502 bash scripts/run_demo.sh to change port)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for env in .venv venv env; do
  if [ -f "$env/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$env/bin/activate"
    echo "[ok] activated $env"
    break
  fi
done

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42

if [ ! -f models_cache/all-MiniLM-L6-v2/config.json ]; then
  echo "[!!] models_cache/all-MiniLM-L6-v2 is missing. Run once while online:  python scripts/download_model.py"
  echo "     (the app will still start; Model B and semantic matching will show an error notice)"
fi
if [ ! -f results/pilot_metrics.csv ]; then
  echo "[..] results/ is empty: the RESULTS page will ask you to run the notebook."
fi

echo "[ok] offline mode on; launching http://localhost:${PORT:-8501}"
exec python -m streamlit run app/Home.py --server.port "${PORT:-8501}" --server.headless true
