#!/usr/bin/env bash
# Final scope E01-E04 (DESIGN_DECISIONS D18): the per-exercise cross-view job runs exercises in order E01 -> E09, so it
# is stopped as soon as the log shows the first run of the first out-of-scope exercise. Finished folds keep their
# done.json; a partly trained E05 fold has none and is never used. Only a cross-view job started WITHOUT --exercises is
# stopped (a resumed job limited to E01-E04 never reaches E05). Polls once a minute; stops when the job ends on its own.
# usage (from cvdc-physio/): bash scripts/pilot/phase5/stop_crossview_at.sh [E05] [runs/full/perex_crossview/train.log]
STOP=${1:-E05}
LOG=${2:-runs/full/perex_crossview/train.log}
OUT=$(dirname "$LOG")/watcher.log
echo "$(date '+%F %T') watching $LOG; stop the cross-view job at the first '=== $STOP' run" >> "$OUT"
while true; do
  if grep -q "^=== $STOP " "$LOG"; then
    powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" |
      Where-Object { \$_.CommandLine -match 'cvdc\.eval\.cross_view' -and \$_.CommandLine -notmatch '--exercises' } |
      ForEach-Object { Write-Output (\"stopping pid \" + \$_.ProcessId); Stop-Process -Id \$_.ProcessId -Force }" >> "$OUT" 2>&1
    echo "$(date '+%F %T') stopped: $STOP reached (D18, E01-E04 only)" >> "$OUT"
    break
  fi
  if tail -n 1 "$LOG" | grep -q "^exit "; then
    echo "$(date '+%F %T') job ended on its own: $(tail -n 1 "$LOG")" >> "$OUT"
    break
  fi
  sleep 60
done
