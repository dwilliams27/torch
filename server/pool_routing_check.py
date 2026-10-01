#!/usr/bin/env python3
"""CPU stub check of the pool scheduler (DreamServer._next_job): --pool-held carry keeps a repeated
framing off engines without takes_held, without starving it (no engines, no network, ~1 s).

    python server/pool_routing_check.py      # PASS/FAIL per case; exit 1 on any FAIL
"""
import collections, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.app import DreamServer
class W:
    def __init__(s, name, held): s.name, s.serving, s.waiting = name, True, True; s.engine = types.SimpleNamespace(takes_held=held)
class C:
    def __init__(s, cid): s.cid, s.pending, s.closed, s.want_started = cid, None, False, False
class F:
    def __init__(s, conn, fid, seed=1): s.conn, s.fid, s.seed, s.id, s.engine = conn, fid, seed, 0, None
def srv(workers, carry=True):
    d = DreamServer.__new__(DreamServer)
    d.args = types.SimpleNamespace(pool_held="carry" if carry else "any")
    d.workers, d.conns, d._pool_kf = workers, collections.OrderedDict(), collections.OrderedDict()
    d.ms_infer_ema, d.bench_ms = 0, 0
    return d
def job(d, w):
    r = d._next_job(w); return None if r is None else r[1]
ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS" if cond else "FAIL"), name)
# 1. coreml ranked first (faster), torch second, both idle; a held frame must go to torch
ane, gpu = W("coreml", None), W("torch", True)
d = srv([ane, gpu]); c = C(1); d.conns[1] = c
c.pending = F(c, 5); check("new framing: fastest idle (ANE) takes it", job(d, ane) is not None)
c.pending = F(c, 5); check("held frame: ANE refuses", job(d, ane) is None)
check("held frame: GPU takes it though the ANE is faster and idle", job(d, gpu) is not None)
# 2. a new framing while both idle: the slower GPU defers to the idle ANE
c.pending = F(c, 6); check("new framing: GPU defers to faster idle ANE", job(d, gpu) is None)
check("new framing: ANE takes it", job(d, ane) is not None)
# 3. no carrying engine serves: carry off, held frames still painted
a1, a2 = W("coreml", None), W("coreml2", None)
d = srv([a1, a2]); c = C(1); d.conns[1] = c
c.pending = F(c, 5); job(d, a1); c.pending = F(c, 5)
check("no takes_held engine: held frame painted", job(d, a1) is not None)
# 4. frames without a framing id are never refused
d = srv([ane, gpu]); c = C(1); d.conns[1] = c
c.pending = F(c, None); check("fid None: ANE takes it", job(d, ane) is not None)
# 5. 'any' keeps the old behaviour
d = srv([ane, gpu], carry=False); c = C(1); d.conns[1] = c
c.pending = F(c, 5); job(d, ane); c.pending = F(c, 5); check("any: ANE takes held frames", job(d, ane) is not None)
# 6. fairness: a skipped connection keeps its place; two clients
d = srv([ane, gpu]); c1, c2 = C(1), C(2); d.conns[1] = c1; d.conns[2] = c2
c1.pending = F(c1, 5); job(d, ane); c1.pending = F(c1, 5); c2.pending = F(c2, 9)
gpu.waiting = False   # GPU busy
f = job(d, ane); check("ANE skips held c1, takes c2", f is not None and f.conn is c2)
check("c1 still first in the round", next(iter(d.conns)) == 1)
# 7. the GPU engine's process dies: it stops serving, carry turns off, the ANE paints held frames
import threading
d = srv([ane, gpu]); c = C(1); d.conns[1] = c; gpu.waiting = True
d.lock, d._busy_window, d.gap_ms_ema = threading.Condition(), collections.deque(), 0
d.loop = types.SimpleNamespace(call_soon_threadsafe=lambda *a: (_ for _ in ()).throw(RuntimeError()))
for w in (ane, gpu): w.done, w.ms_ema, w.gap_ms_ema = 0, 0, 0
gpu.engine._proc = types.SimpleNamespace(is_alive=lambda: False)
c.pending = F(c, 5); job(d, ane); c.pending = F(c, 5)
d._frame_done(gpu, c.pending, None, RuntimeError("engine process died"), 0.0, 0.1, None)
check("dead GPU process: no longer serving", not gpu.serving)
check("dead GPU process: ANE takes the held frame", job(d, ane) is not None)
ane.engine._proc = types.SimpleNamespace(is_alive=lambda: False)
d._frame_done(ane, F(c, 6), None, RuntimeError("engine process died"), 0.0, 0.1, None)
check("last serving engine is kept (errors reach the client)", ane.serving)
sys.exit(0 if ok else 1)
