#!/usr/bin/env bash
# temporal validation の一連の実行。このフォルダで bash run_a1.sh。ログは logs/ に出る。
set -euo pipefail
cd "$(dirname "$0")"
# uv がプロジェクトの仮想環境を自動で使う
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
mkdir -p logs
run() { local name=$1; shift; echo "=== $(date '+%F %T') $name"; uv run "$@" 2>&1 | tee "logs/$name.log"; }
run 01_retrain        retrain_final_seeded.py --force
run 02_evaluate      evaluate_external.py
run 03_analyze       analyze_external.py
echo "=== $(date '+%F %T') done"
