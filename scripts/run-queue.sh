#!/bin/bash
# run-queue.sh QUEUEFILE OUTDIR: run "name args..." lines sequentially; skips names whose log is already complete.
Q=$1; O=$2; mkdir -p "$O"
R=.build/out/Products/Release
cp "$R/ane-train" "$O/ane-train.bin"
cp -R "$R"/*.bundle "$R/PackageFrameworks" "$O/" 2>/dev/null  # metallib bundles must sit beside the binary
while read -r name args; do
  [[ -z $name || $name == \#* ]] && continue
  grep -qs "throughput excluding" "$O/$name.log" && { echo "skip $name" >> "$O/queue.log"; continue; }
  echo "$name start $(date -u +%FT%TZ)" >> "$O/queue.log"
  "$O/ane-train.bin" $args --jsonl "$O/$name.jsonl" > "$O/$name.log" 2>&1 < /dev/null; rc=$?
  echo "$name end $(date -u +%FT%TZ) exit $rc" >> "$O/queue.log"
done < "$Q"
