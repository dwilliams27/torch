#!/usr/bin/env python3
"""Show where Core ML places each op (ANE / GPU / CPU) and the estimated cost split.

  ~/hypnagogia-cache/venv-coreml/bin/python bench/coreml/plan.py ~/hypnagogia-cache/coreml/sdturbo_unet_512_ane.mlmodelc [NE|ALL|GPU]
"""
import collections
import sys

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

path = sys.argv[1]
cu = {"NE": ct.ComputeUnit.CPU_AND_NE, "ALL": ct.ComputeUnit.ALL, "GPU": ct.ComputeUnit.CPU_AND_GPU}[
    sys.argv[2] if len(sys.argv) > 2 else "NE"]
plan = MLComputePlan.load_from_path(path=path, compute_units=cu)
prog = plan.model_structure.program
by_dev = collections.Counter()
cost_dev = collections.Counter()
offenders = collections.Counter()
for fname, fn in prog.functions.items():
    for op in fn.block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        c = plan.get_estimated_cost_for_mlprogram_operation(op)
        if u is None:
            continue
        dev = type(u.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")
        by_dev[dev] += 1
        w = c.weight if c is not None else 0.0
        cost_dev[dev] += w
        if dev != "NeuralEngine":
            offenders[(op.operator_name, dev)] += 1
tot = sum(cost_dev.values()) or 1
print("ops by device:", dict(by_dev))
print("est. cost share:", {k: round(v / tot, 3) for k, v in cost_dev.items()})
print("non-ANE ops:", offenders.most_common(20))
