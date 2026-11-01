#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${GRPO_REPO_DIR:-/home/zs/assignment5-alignment}"
SESSION_NAME="${GRPO_SESSION:-grpo-wait}"
POLL_SECONDS="${GRPO_POLL_SECONDS:-30}"
STABLE_CHECKS="${GRPO_STABLE_CHECKS:-3}"
MAX_MEMORY_MIB="${GRPO_MAX_MEMORY_MIB:-1024}"
MAX_UTILIZATION="${GRPO_MAX_UTILIZATION:-5}"
LOG_FILE="${GRPO_LOG_FILE:-/tmp/grpo-wait.log}"
PYTHON_BIN="${GRPO_PYTHON:-$REPO_DIR/.venv/bin/python}"
TRAIN_SCRIPT="${GRPO_SCRIPT:-}"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

usage() {
    cat <<'USAGE'
Usage: bash /tmp/run_grpo_when_gpus_free.sh

Creates the tmux session grpo-wait and waits for GPUs 0 and 1.
Each GPU must have no compute processes, at most 1024 MiB in use,
and at most 5% utilization for 3 checks, 30 seconds apart.
The training output is appended to /tmp/grpo-wait.log.

Optional environment variables:
  GRPO_REPO_DIR          Project directory.
  GRPO_SESSION           New tmux session name.
  GRPO_POLL_SECONDS      Seconds between checks.
  GRPO_STABLE_CHECKS     Required consecutive idle checks.
  GRPO_MAX_MEMORY_MIB    Maximum used memory per GPU.
  GRPO_MAX_UTILIZATION   Maximum utilization per GPU, from 0 to 100.
  GRPO_LOG_FILE          Absolute path to the log file.
  GRPO_PYTHON            Absolute path to the Python executable.
  GRPO_SCRIPT            Training script, absolute or relative to the project.

Attach: tmux attach -t grpo-wait
Detach: press Ctrl-b, then d.
Cancel waiting or stop training: attach, then press Ctrl-c.
USAGE
}

validate_configuration() {
    [[ "$SESSION_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Invalid session name.'
    [[ "$REPO_DIR" == /* && -d "$REPO_DIR" ]] || fail "Project directory does not exist: $REPO_DIR"
    [[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || fail "Python executable not found: $PYTHON_BIN"
    [[ "$LOG_FILE" == /* ]] || fail 'GRPO_LOG_FILE must be an absolute path.'

    local setting
    for setting in POLL_SECONDS STABLE_CHECKS MAX_MEMORY_MIB MAX_UTILIZATION; do
        [[ "${!setting}" =~ ^(0|[1-9][0-9]*)$ ]] || fail "$setting must be a nonnegative integer."
    done
    (( POLL_SECONDS > 0 && STABLE_CHECKS > 0 )) || fail 'Poll interval and stable check count must be positive.'
    (( MAX_UTILIZATION <= 100 )) || fail 'Maximum utilization cannot exceed 100.'

    if [[ -z "$TRAIN_SCRIPT" ]]; then
        if [[ -f "$REPO_DIR/cs336_alignment/grpo_script_on_policy.py" ]]; then
            TRAIN_SCRIPT="$REPO_DIR/cs336_alignment/grpo_script_on_policy.py"
        else
            TRAIN_SCRIPT="$REPO_DIR/cs336_alignment/grpo_script_on_poliy.py"
        fi
    elif [[ "$TRAIN_SCRIPT" != /* ]]; then
        TRAIN_SCRIPT="$REPO_DIR/$TRAIN_SCRIPT"
    fi
    [[ -f "$TRAIN_SCRIPT" ]] || fail "Training script not found: $TRAIN_SCRIPT"
    command -v nvidia-smi >/dev/null || fail 'nvidia-smi is not installed or not on PATH.'
    command -v tee >/dev/null || fail 'tee is not installed or not on PATH.'
}

gpus_are_idle() {
    local gpu_status compute_processes gpu_index used_memory utilization
    local gpu_zero_seen=0 gpu_one_seen=0 idle=1

    if ! gpu_status="$(nvidia-smi --id=0,1 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>&1)"; then
        log "GPU query failed; continuing to wait: $gpu_status"
        return 1
    fi

    while IFS=',' read -r gpu_index used_memory utilization; do
        gpu_index="${gpu_index//[[:space:]]/}"
        used_memory="${used_memory//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [[ "$gpu_index" =~ ^[01]$ && "$used_memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || {
            log "Unrecognized GPU status; continuing to wait: $gpu_status"
            return 1
        }
        if [[ "$gpu_index" == 0 ]]; then
            gpu_zero_seen=1
        else
            gpu_one_seen=1
        fi
        log "GPU $gpu_index: ${used_memory} MiB used, ${utilization}% utilization."
        if (( used_memory > MAX_MEMORY_MIB || utilization > MAX_UTILIZATION )); then
            idle=0
        fi
    done <<< "$gpu_status"

    if (( !gpu_zero_seen || !gpu_one_seen )); then
        log 'Both GPUs 0 and 1 must be present; continuing to wait.'
        return 1
    fi

    if ! compute_processes="$(nvidia-smi --id=0,1 --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)"; then
        log "GPU process query failed; continuing to wait: $compute_processes"
        return 1
    fi
    if [[ -n "${compute_processes//[[:space:]]/}" ]]; then
        log 'Compute processes are still using GPU 0 or 1.'
        return 1
    fi
    (( idle == 1 ))
}

run_worker() {
    mkdir -p -- "$(dirname -- "$LOG_FILE")"
    local consecutive_idle=0 training_status
    log "Waiting for GPUs 0 and 1; need $STABLE_CHECKS consecutive idle checks."
    log "Training script: $TRAIN_SCRIPT"

    while true; do
        if gpus_are_idle; then
            consecutive_idle=$((consecutive_idle + 1))
            log "Idle check $consecutive_idle/$STABLE_CHECKS."
            if (( consecutive_idle >= STABLE_CHECKS )); then
                break
            fi
        else
            consecutive_idle=0
        fi
        sleep "$POLL_SECONDS"
    done

    cd -- "$REPO_DIR"
    log 'Both GPUs are idle. Starting training with CUDA_VISIBLE_DEVICES=0,1.'
    set +e
    CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" -u "$TRAIN_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
    training_status=${PIPESTATUS[0]}
    set -e
    log "Training exited with status $training_status."
    return "$training_status"
}

main() {
    case "${1:-}" in
        --help|-h) usage; return ;;
        --worker) ;;
        '') ;;
        *) fail "Unknown argument: $1. Use --help." ;;
    esac
    (( $# <= 1 )) || fail 'Too many arguments. Use --help.'
    validate_configuration

    if [[ "${1:-}" == --worker ]]; then
        run_worker
        return
    fi

    command -v tmux >/dev/null || fail 'tmux is not installed or not on PATH.'
    if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
        fail "Session $SESSION_NAME already exists. Attach with: tmux attach -t $SESSION_NAME"
    fi

    local script_path worker_command pane_id
    script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
    printf -v worker_command '%q ' env \
        "GRPO_REPO_DIR=$REPO_DIR" "GRPO_SESSION=$SESSION_NAME" \
        "GRPO_POLL_SECONDS=$POLL_SECONDS" "GRPO_STABLE_CHECKS=$STABLE_CHECKS" \
        "GRPO_MAX_MEMORY_MIB=$MAX_MEMORY_MIB" "GRPO_MAX_UTILIZATION=$MAX_UTILIZATION" \
        "GRPO_LOG_FILE=$LOG_FILE" "GRPO_PYTHON=$PYTHON_BIN" "GRPO_SCRIPT=$TRAIN_SCRIPT" \
        bash "$script_path" --worker

    pane_id="$(tmux new-session -d -P -F '#{pane_id}' -s "$SESSION_NAME" -n training -c "$REPO_DIR" 'bash --noprofile --norc')"
    tmux send-keys -t "$pane_id" -l -- "$worker_command"
    tmux send-keys -t "$pane_id" Enter
    printf 'Created tmux session: %s\nAttach: tmux attach -t %s\nLog: %s\n' "$SESSION_NAME" "$SESSION_NAME" "$LOG_FILE"
}

main "$@"
