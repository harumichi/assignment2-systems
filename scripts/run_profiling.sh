#!/usr/bin/env bash
set -euo pipefail

batch_sizes=(128 256 512 1024)

for bs in "${batch_sizes[@]}"; do
	echo "[Profiling] batch size=${bs}"
	uv run \
        nsys profile \
		--capture-range=cudaProfilerApi \
		--python-backtrace=cuda \
		--pytorch=autograd-nvtx \
		--force-overwrite true \
		-o "result_bs${bs}" \
		python cs336_systems/cmd/run_profiling.py --batch-size
done
