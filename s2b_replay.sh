#!/usr/bin/env bash
# ============================================================================
# Stage 2b — Replay-based post-validation Spre → Shigh (paper §III-C)
#
# Open-loop replays every Spre episode and keeps only the survivors
# (task completion + collision-free). The paper uses a MuJoCo digital
# twin; the released code ships three verifiers:
#
#   * ``trust_spre`` (default): no verification, every Spre episode passes.
#                                Use for smoke testing / when no simulator
#                                is available yet.
#   * ``energy``: cheap kinematic heuristic (out-of-range states + acc spikes).
#   * ``mujoco``: real MuJoCo replay. Plug in your own twin module via
#                 ``REPLAY_TWIN_MODULE`` (must expose ``replay(states, actions,
#                 task_id) -> {"success": bool, "collision_free": bool}``).
# ============================================================================
set -Eeuo pipefail

# Resolve paths from this script, so the command also works when invoked as
# ``bash /path/to/Dexora/s2b_replay.sh`` from the repository's parent dir.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${DEXORA_LEROBOT_ROOT:=data/Dexora_Real-World_Dataset/airbot_pick_and_place}"
: "${SPRE_DIR:=runs/spre}"
: "${SHIGH_FILE:=runs/shigh.json}"
: "${REPLAY_VERIFIER:=trust_spre}"   # trust_spre | energy | mujoco
: "${REPLAY_TWIN_MODULE:=}"          # only used when REPLAY_VERIFIER=mujoco
: "${AIRBOT_MUJOCO_TASK_MODULE:=}"   # Discoverse task module: SimNode + cfg
: "${AIRBOT_MUJOCO_XML:=}"            # fallback plain MuJoCo XML
: "${AIRBOT_MUJOCO_GUI:=0}"           # 0=headless, 1=open MuJoCo viewer

extra_args=()
visualize_args=()
if [[ -n "$REPLAY_TWIN_MODULE" ]]; then
    extra_args+=(--twin_module "$REPLAY_TWIN_MODULE")
fi

if [[ "$REPLAY_VERIFIER" == "mujoco" && -z "$REPLAY_TWIN_MODULE" ]]; then
    REPLAY_TWIN_MODULE=scripts.airbot_mujoco_twin
    extra_args+=(--twin_module "$REPLAY_TWIN_MODULE")
fi

if [[ "$REPLAY_VERIFIER" == "mujoco" ]]; then
    export AIRBOT_MUJOCO_TASK_MODULE AIRBOT_MUJOCO_XML AIRBOT_MUJOCO_GUI
fi

if [[ "$AIRBOT_MUJOCO_GUI" == "1" || "$AIRBOT_MUJOCO_GUI" == "true" || "$AIRBOT_MUJOCO_GUI" == "yes" ]]; then
    visualize_args+=(--visualize)
fi

# Use the regular Python interpreter for the MuJoCo viewer.
GUI_VALUE="$(printf '%s' "$AIRBOT_MUJOCO_GUI" | tr '[:upper:]' '[:lower:]')"
if [[ "$GUI_VALUE" =~ ^(1|true|yes|on)$ ]]; then
    # A GUI viewer cannot be created with EGL/OSMesa. This is especially easy
    # to hit on Ubuntu systems that export MUJOCO_GL=egl for training jobs.
    export MUJOCO_GL=glfw
    if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
        echo "WARNING: MuJoCo GUI requested, but DISPLAY/WAYLAND_DISPLAY is not set; no desktop window can be shown." >&2
    fi
fi
mkdir -p "$(dirname "$SHIGH_FILE")"
echo "==> Stage-2b replay verification (verifier=$REPLAY_VERIFIER)"
python scripts/replay_validate.py \
    --pre_screening_file="$SPRE_DIR/complete_analysis_results.json" \
    --lerobot_root="$DEXORA_LEROBOT_ROOT" \
    --output_file="$SHIGH_FILE" \
    --verifier="$REPLAY_VERIFIER" \
    "${visualize_args[@]}" \
    "${extra_args[@]}"

echo "==> Shigh written to $SHIGH_FILE"
