#!/usr/bin/env bash
set -uo pipefail

ids_file=${1:?usage: relay_modal_volumes.sh IDS_FILE}
source_profile=${SOURCE_PROFILE:-gpudashboard0}
target_profile=${TARGET_PROFILE:-baptinjean}
volume_name=${VOLUME_NAME:-perception-test-2026-data}
log_file=${RELAY_LOG:-/tmp/2706-detached-relay.log}
work_dir=$(mktemp -d /tmp/2706-detached-relay.XXXXXX)
pending_file=$(mktemp /tmp/2706-detached-pending.XXXXXX)
next_file=$(mktemp /tmp/2706-detached-next.XXXXXX)
cp "$ids_file" "$pending_file"

cleanup() {
    find "$work_dir" -maxdepth 1 -type f -delete 2>/dev/null || true
    rmdir "$work_dir" 2>/dev/null || true
    unlink "$pending_file" 2>/dev/null || true
    unlink "$next_file" 2>/dev/null || true
}
trap cleanup EXIT

for round in 1 2 3 4 5; do
    : > "$next_file"
    while IFS= read -r video_id; do
        [[ -n "$video_id" ]] || continue
        volume_path="/shared/walking-tours/$video_id/$video_id.mkv"
        local_path="$work_dir/$video_id.mkv"
        printf 'start %s round=%s video=%s\n' "$(date -Is)" "$round" "$video_id" >> "$log_file"
        if MODAL_PROFILE="$source_profile" modal volume get --force "$volume_name" "$volume_path" "$local_path" \
                > /tmp/2706-detached-get.log 2>&1 \
            && MODAL_PROFILE="$target_profile" modal volume put --force "$volume_name" "$local_path" "$volume_path" \
                > /tmp/2706-detached-put.log 2>&1; then
            printf 'complete %s round=%s video=%s\n' "$(date -Is)" "$round" "$video_id" >> "$log_file"
        else
            printf 'failed %s round=%s video=%s\n' "$(date -Is)" "$round" "$video_id" >> "$log_file"
            printf '%s\n' "$video_id" >> "$next_file"
        fi
        unlink "$local_path" 2>/dev/null || true
    done < "$pending_file"
    [[ -s "$next_file" ]] || {
        printf 'finished %s status=success\n' "$(date -Is)" >> "$log_file"
        exit 0
    }
    cp "$next_file" "$pending_file"
done

printf 'finished %s status=failed remaining=%s\n' "$(date -Is)" "$(wc -l < "$pending_file")" >> "$log_file"
exit 1
