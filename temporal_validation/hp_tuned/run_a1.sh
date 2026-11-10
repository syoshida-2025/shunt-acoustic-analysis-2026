#!/usr/bin/env bash
# v4 追加（2026-10-27）: 方針 A1 の一連の実行。このフォルダで `bash run_a1.sh`。ログは logs/ に出る。
set -euo pipefail
cd "$(dirname "$0")"
# uv がプロジェクトの仮想環境を自動で使う
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
mkdir -p logs
run() { local name=$1; shift; echo "=== $(date '+%F %T') $name"; uv run "$@" 2>&1 | tee "logs/$name.log"; }
run 01_retrain        retrain_final_seeded.py
run 02_eval_noisy     evaluate_external_noisy.py
run 03_eval_clean     evaluate_external.py
run 04_analyze_noisy  analyze_external.py --images noisy
run 05_analyze_clean  analyze_external.py --images clean
run 06_gradcam_noisy  gradcam_resnet.py --images noisy
run 07_gradcam_clean  gradcam_resnet.py --images clean
echo "=== $(date '+%F %T') done"
