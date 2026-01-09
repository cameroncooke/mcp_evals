#!/bin/bash
set -euo pipefail
export EVAL_MCP_PROCESS=1
exec "$@"
