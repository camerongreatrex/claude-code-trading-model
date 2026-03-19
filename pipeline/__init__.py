"""
pipeline/
---------
Offline strategy pipeline — run via ``python run.py`` or individually
with ``python -m pipeline.<module>``.

Module order (each reads from the previous module's output):
  data_pipeline       → data/raw/
  feature_engineering → data/features/
  macro_features      → data/macro/
  signal_generation   → data/signals/
  backtester          → data/results/
  portfolio           → data/results/
"""
