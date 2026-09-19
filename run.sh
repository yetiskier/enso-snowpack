#!/usr/bin/env bash
# One-shot: download everything (resumable, cached under data/raw) and analyse.
# Needs internet access to noaa.gov and usda.gov. Re-run to resume after a hiccup.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
python3 -m enso_snowpack run "$@"
echo
echo "Report: results/summary.md   Figures: results/fig*.png   Tables: results/*.csv"
