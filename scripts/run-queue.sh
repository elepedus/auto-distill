#!/bin/bash
# run-queue.sh QUEUEFILE OUTDIR [BINARY=ane-train]: run "name args..." lines sequentially; skips names whose log is already complete.
Q=$1; O=$2; B=${3:-ane-train}; mkdir -p "$O"
R=.build/out/Products/Release
cp "$R/$B" "$O/$B.bin"
cp -R "$R"/*.bundle "$R/PackageFrameworks" "$O/" 2>/dev/null  # metallib bundles must sit beside the binary
while read -r name args; do
  [[ -z $name || $name == \#* ]] && continue
  grep -qs "throughput excluding" "$O/$name.log" && { echo "skip $name" >> "$O/queue.log"; continue; }
  echo "$name start $(date -u +%FT%TZ)" >> "$O/queue.log"
  "$O/$B.bin" $args --jsonl "$O/$name.jsonl" > "$O/$name.log" 2>&1 < /dev/null; rc=$?
  echo "$name end $(date -u +%FT%TZ) exit $rc" >> "$O/queue.log"
done < "$Q"
