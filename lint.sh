#!/bin/bash
set -euo pipefail
error_handler() {
    echo "Error on line ${BASH_LINENO[0]}: ${BASH_COMMAND}"
}
trap 'error_handler' ERR

source .venv/bin/activate

# Get file path (second argument if --check is first, otherwise first argument)
if [[ "${1:-}" == "--check" ]]; then
    FILE="${2:-}"
    if [[ -n "$FILE" ]]; then
        ruff check "$FILE"
        basedpyright "$FILE"
        ruff format --check "$FILE"
    else
        ruff check .
        basedpyright
        ruff format --check
    fi
else
    FILE="${1:-}"
    if [[ -n "$FILE" ]]; then
        ruff check --fix "$FILE"
        basedpyright "$FILE"
        ruff format "$FILE"
    else
        ruff check --fix .
        basedpyright
        ruff format
    fi
fi
