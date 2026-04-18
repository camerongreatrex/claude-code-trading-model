"""
v1/pipeline/
------------
Offline strategy pipeline — run via ``python -m v1.scripts.run_v1`` or
individually with ``python -m v1.pipeline.<module>``.

Module order (each reads from the previous module's output):
  data_pipeline       → data/v1/raw/
  feature_engineering → data/v1/features/
  macro_features      → data/shared/macro/
  signal_generation   → data/v1/signals/
  backtester          → data/v1/results/
  portfolio           → data/v1/results/
"""
