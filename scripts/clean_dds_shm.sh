#!/usr/bin/env bash
# Remove orphaned Fast DDS shared-memory segments.
#
# Adapted from TEMPO's scripts/clean_dds_shm.sh. With --ipc=host the container's
# Fast DDS segments ARE the host's /dev/shm entries, and a participant killed
# without a clean shutdown never removes its own. They pile up until a new
# participant blocks scanning them and every ros2 command hangs with no error
# message. Only orphans are removed; anything a live process holds open is left
# alone.
set -u

removed=0
kept=0

for f in /dev/shm/fastrtps_* /dev/shm/fastdds_* /dev/shm/sem.fastrtps_* \
         /dev/shm/sem.fastdds_*; do
    [ -e "$f" ] || continue
    # fuser is the reliable "is anyone holding this open" test; if it is absent
    # we conservatively KEEP the segment rather than risk killing a live run.
    if command -v fuser >/dev/null 2>&1; then
        if fuser -s "$f" 2>/dev/null; then
            kept=$((kept + 1))
        else
            rm -f "$f" 2>/dev/null && removed=$((removed + 1))
        fi
    else
        kept=$((kept + 1))
    fi
done

if [ "$removed" -gt 0 ]; then
    echo "[dds] removed $removed orphaned shared-memory segment(s), kept $kept in use"
elif [ "$kept" -gt 0 ]; then
    echo "[dds] $kept shared-memory segment(s) in use, nothing to clean"
fi
exit 0
