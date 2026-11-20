#!/usr/bin/env bash
# Build thesis/ch7/overleaf_upload/ (folders as Overleaf expects) and ch7-overleaf-upload.zip next to it.
# Paths match the thesis: sections/..., figures/...   Run from anywhere: bash thesis/scripts/build_overleaf_upload.sh
set -euo pipefail
CH7="$(cd "$(dirname "$0")/../ch7" && pwd)"
OUT="$CH7/overleaf_upload"
rm -rf "$OUT" && mkdir -p "$OUT/sections/tables" "$OUT/figures/exp2/networks" "$OUT/figures/ch7-ws"
cp "$CH7"/exp2/figures/exp2/networks/*.pdf "$OUT/figures/exp2/networks/"
cp "$CH7/exp2/appendix_networks.tex" "$OUT/sections/appendix-exp2-networks.tex"
cp "$CH7/exp2/tables/exp2/exp2-signalticks-adult.tex" "$OUT/sections/tables/ch7-ex2-adult.tex"
cp "$CH7/exp2/tables/exp2/exp2-signalticks-churn.tex" "$OUT/sections/tables/ch7-ex2-churn.tex"
cp "$CH7/exp2/tables/exp2/exp2-signalticks-heart100.tex" "$OUT/sections/tables/ch7-ex2-heart.tex"
cp "$CH7/exp1/tables/exp1-networks.tex" "$OUT/sections/tables/ch7-ex1-networks.tex"
cp "$CH7/exp1/tables/exp1-benchmarks.tex" "$OUT/sections/tables/ch7-ex1-benchmarks.tex"
cp "$CH7/exp2/tables/exp2/exp2-benchmarks.tex" "$OUT/sections/tables/ch7-ex2-benchmarks.tex"
cp "$CH7/appendix/tables/ch7-heart-splits.tex" "$OUT/sections/tables/ch7-heart-splits.tex"
cp "$CH7/exports/ch7-datasets.tex" "$OUT/sections/tables/ch7-datasets.tex"   # sync_ch7_exports.py <- experiments/thesis_datasets
# The thesis table style (every generated table calls \thesistablestyle): at the thesis root, next to
# dissertation.tex and uathesis.cls, because the preamble loads it once: \input{thesis-tablestyle}
cp "$CH7/../tablestyle/thesis-tablestyle.tex" "$OUT/thesis-tablestyle.tex"
cp "$CH7/exports/ch7-run-config.tex" "$OUT/sections/tables/ch7-run-config.tex"   # sync_ch7_exports.py <- experiments/thesis_run_config
cp "$CH7"/screenshots/ch7-ws-0[0-8]-*.png "$OUT/figures/ch7-ws/"
# Chapter 7 figures (thesis/ch7/figlib.py): used = picked for the chapter, options = the rest. The floats'
# \includegraphics paths are figures/ch7/<group>/<id>; FIGURES.md lists them.
for g in used options; do
  mkdir -p "$OUT/figures/ch7/$g" "$OUT/sections/figures/$g"
  cp "$CH7"/figures/sections/figures/$g/*.tex "$OUT/sections/figures/$g/"
  for t in "$CH7"/figures/sections/figures/$g/*.tex; do
    id="$(basename "$t" .tex)"
    if grep -q "{figures/ch7/$g/$id.png}" "$t"; then   # a raster-only option names its .png
      cp "$CH7/figures/figures/ch7/$g/$id.png" "$OUT/figures/ch7/$g/"
    else
      cp "$CH7/figures/figures/ch7/$g/$id.pdf" "$OUT/figures/ch7/$g/"
    fi
  done
done
rm -f "$OUT/ch7-overleaf-upload.zip"
(cd "$OUT" && zip -qr ch7-overleaf-upload.zip thesis-tablestyle.tex sections figures)
echo "built $OUT and $OUT/ch7-overleaf-upload.zip"
