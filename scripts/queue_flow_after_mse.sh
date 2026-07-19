#!/usr/bin/env bash
set -euo pipefail

echo "retired: this script used rasterized TTY glyphs, not native tile pixels" >&2
echo "use scripts/queue_pixel_action_after_flow.sh for the approved experiment" >&2
exit 1
