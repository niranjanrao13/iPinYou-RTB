#!/bin/bash
# Reimplementation of the make-ipinyou-data Makefile, using Python 3 scripts.
# Run this from the make-ipinyou-data project root (the folder that contains
# this script's sibling "py3" folder and "schema.txt", and where you want the
# per-campaign output folders like "1458" to be created).
#
# Usage:
#   ./py3/run_pipeline.sh
#
# Prerequisites (Ubuntu/WSL):
#   sudo apt update && sudo apt install -y bzip2 python3

set -euo pipefail

BASE="."
ORIGINALFOLDER="$BASE/original-data/ipinyou.contest.dataset"
TRAIN="$ORIGINALFOLDER/train"
TEST="$ORIGINALFOLDER/test"
PY="$BASE/py3"
SCHEMA="$PY/schema.txt"

if [ -d "$TRAIN" ] || [ -d "$TEST" ]; then
    echo "== init: $TRAIN or $TEST already exist, skipping re-decompression =="
    echo "   (delete them first with: rm -rf \"$TRAIN\" \"$TEST\"  -- if you want a clean re-run)"
else
    echo "== init: staging + decompressing raw files =="
    mkdir -p "$TRAIN"
    cp "$ORIGINALFOLDER"/training2nd/imp.*.bz2 "$TRAIN"/
    cp "$ORIGINALFOLDER"/training2nd/clk.*.bz2 "$TRAIN"/
    cp "$ORIGINALFOLDER"/training3rd/imp.*.bz2 "$TRAIN"/
    cp "$ORIGINALFOLDER"/training3rd/clk.*.bz2 "$TRAIN"/
    bzip2 -d "$TRAIN"/*

    mkdir -p "$TEST"
    cp "$ORIGINALFOLDER"/testing2nd/* "$TEST"/
    cp "$ORIGINALFOLDER"/testing3rd/* "$TEST"/
    bzip2 -d "$TEST"/*
fi

mkdir -p "$BASE/all"

echo "== clk: merging click logs =="
cat "$TRAIN"/clk*.txt > "$BASE/all/clk.all.txt"

echo "== train.log =="
cat "$TRAIN"/imp*.txt | python3 "$PY/mkdata.py" "$SCHEMA" "$BASE/all/clk.all.txt" > "$BASE/all/train.log.txt"
python3 "$PY/formalizeua.py" "$BASE/all/train.log.txt"

echo "== test.log =="
cat "$TEST"/*.txt | python3 "$PY/mktest.py" "$SCHEMA" > "$BASE/all/test.log.txt"
python3 "$PY/formalizeua.py" "$BASE/all/test.log.txt"

echo "== advertisers: splitting per-campaign =="
python3 "$PY/splitadvertisers.py" "$BASE" 25 "$BASE/all/train.log.txt" "$BASE/all/test.log.txt"

echo "== yzx: building feature-indexed vectors =="
advertisers="1458 2261 2997 3386 3476 2259 2821 3358 3427"
for advertiser in $advertisers; do
    if [ -d "$BASE/$advertiser" ]; then
        echo "$advertiser"
        python3 "$PY/mkyzx.py" "$BASE/$advertiser/train.log.txt" "$BASE/$advertiser/test.log.txt" \
            "$BASE/$advertiser/train.yzx.txt" "$BASE/$advertiser/test.yzx.txt" "$BASE/$advertiser/featindex.txt"
    fi
done

echo "== done. Campaign 1458 output is in ./1458/ =="
