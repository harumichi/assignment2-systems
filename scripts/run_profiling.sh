
nsys profile \
--capture-range=cudaProfilerApi --python-backtrace=cuda -o result \
python cmd/run_profiling.py
