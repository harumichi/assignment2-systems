uv run \
nsys profile \
--capture-range=cudaProfilerApi \
--python-backtrace=cuda \
--pytorch=autograd-nvtx \
-o result \
python cs336_systems/cmd/run_profiling.py
