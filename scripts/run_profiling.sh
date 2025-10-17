#!/usr/bin/env bash
set -euo pipefail

context_lengths=(128 256 512 1024)

for context_length in "${context_lengths[@]}"; do
	echo "[Profiling] context_length=${context_length}"
	uv run nsys profile \
		--capture-range=cudaProfilerApi \
		--python-backtrace=cuda \
		--pytorch=autograd-nvtx \
		--force-overwrite true \
		-o "result_context_length${context_length}" \
		python cs336_systems/cmd/run_profiling.py --context-length ${context_length}
done
