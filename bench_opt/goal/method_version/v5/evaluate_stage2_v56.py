#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5.6 entry: identical Stage2 standalone evaluation with the b56 z-metric.

``b56_zmetric.fast_eval_z_metric`` is a cell-hash re-implementation of the
pytorch3d ``ball_query(K=1, radius=0.1)`` brute-force sweep used by
``evalution.eval_z_metric``.  It returns the same value bit-for-bit (verified
on the a9_reference epoch_002 run: identical per-query indices, delta 0.0) in
~0.04 s instead of ~7.8 s.

Usage: same arguments as ``bench_opt/evaluate_stage2_opt.py``.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import evaluate_stage2_opt as E  # noqa: E402

from b56_zmetric import fast_eval_z_metric  # noqa: E402

E.eval_z_metric = fast_eval_z_metric

if __name__ == "__main__":
    E.main()
