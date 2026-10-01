#!/usr/bin/env python3
"""Time Core ML diffusion stages per compute unit (run on the mini).

  ~/hypnagogia-cache/venv-coreml/bin/python bench/coreml/bench.py \
      --unet ~/hypnagogia-cache/coreml/sdxs_unet_512_ane.mlmodelc \
      --enc  ~/hypnagogia-cache/coreml/sdxs_tae_enc_512.mlmodelc \
      --dec  ~/hypnagogia-cache/coreml/sdxs_tae_dec_512.mlmodelc \
      --cu NE GPU ALL --iters 30 [--overlap]

All timed sections hold an exclusive flock on ~/hypnagogia-cache/bench.lock.
"""
import argparse
import contextlib
import fcntl
import json
import os
import statistics
import threading
import time

import numpy as np
import coremltools as ct

LOCK = os.path.expanduser("~/hypnagogia-cache/bench.lock")
CU = {
    "NE": ct.ComputeUnit.CPU_AND_NE,
    "GPU": ct.ComputeUnit.CPU_AND_GPU,
    "ALL": ct.ComputeUnit.ALL,
    "CPU": ct.ComputeUnit.CPU_ONLY,
}


@contextlib.contextmanager
def bench_lock():
    f = open(LOCK, "a+")
    t = time.time()
    fcntl.flock(f, fcntl.LOCK_EX)
    waited = time.time() - t
    if waited > 0.5:
        print(f"  (waited {waited:.1f}s for bench lock)", flush=True)
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def load(path, cu):
    t = time.time()
    m = ct.models.CompiledMLModel(path, compute_units=CU[cu])
    return m, time.time() - t


def timeit(fn, iters):
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"median": statistics.median(ts), "p10": ts[len(ts) // 10], "p90": ts[(len(ts) * 9) // 10]}


def unet_inputs(path):
    ref = path.replace(".mlmodelc", "_ref.npz")
    if os.path.exists(ref):
        r = np.load(ref)
        feed = {"sample": r["sample"], "timestep": r["timestep"], "encoder_hidden_states": r["ehs"]}
        if "depth" in r.files:   # a depth-grafted UNet (convert.py --depth-graft)
            feed["depth"] = r["depth"]
        return feed, r["out"]
    raise SystemExit(f"missing {ref}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet")
    ap.add_argument("--enc")
    ap.add_argument("--dec")
    ap.add_argument("--cu", nargs="+", default=["NE", "GPU", "ALL"])
    ap.add_argument("--vae-cu", nargs="+", default=None, help="compute units for enc/dec (default: same list)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--overlap", action="store_true", help="also test UNet(NE) || VAE(GPU) concurrency")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    results = []

    if a.unet:
        feeds, ref = unet_inputs(a.unet)
        for cu in a.cu:
            m, tload = load(a.unet, cu)
            t = time.time()
            out = m.predict(feeds)["noise_pred"]
            tfirst = time.time() - t
            err = float(np.abs(out - ref).max())
            cos = float((out * ref).sum() / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-9))
            for _ in range(3):
                m.predict(feeds)
            with bench_lock():
                st = timeit(lambda: m.predict(feeds), a.iters)
            r = {"stage": "unet", "model": os.path.basename(a.unet), "cu": cu, "load_s": round(tload, 2),
                 "first_s": round(tfirst, 2), "maxerr": round(err, 4), "cos": round(cos, 5), **{k: round(v, 2) for k, v in st.items()}}
            print(json.dumps(r), flush=True)
            results.append(r)
            del m

    lat_shape = None
    for stage, path in (("enc", a.enc), ("dec", a.dec)):
        if not path:
            continue
        res = int(path.rstrip("/").split("_")[-1].split(".")[0])
        if stage == "enc":
            feeds = {"image": np.random.rand(1, 3, res, res).astype(np.float32)}
        else:
            feeds = {"latent": np.random.randn(1, 4, res // 8, res // 8).astype(np.float32)}
        for cu in (a.vae_cu or a.cu):
            m, tload = load(path, cu)
            for _ in range(3):
                m.predict(feeds)
            with bench_lock():
                st = timeit(lambda: m.predict(feeds), a.iters)
            r = {"stage": stage, "model": os.path.basename(path), "cu": cu, "load_s": round(tload, 2),
                 **{k: round(v, 2) for k, v in st.items()}}
            print(json.dumps(r), flush=True)
            results.append(r)
            del m

    if a.overlap and a.unet and a.enc and a.dec:
        res = int(a.enc.rstrip("/").split("_")[-1].split(".")[0])
        feeds_u, _ = unet_inputs(a.unet)
        fe = {"image": np.random.rand(1, 3, res, res).astype(np.float32)}
        fd = {"latent": np.random.randn(1, 4, res // 8, res // 8).astype(np.float32)}
        mu, _ = load(a.unet, "NE")
        me, _ = load(a.enc, "GPU")
        md, _ = load(a.dec, "GPU")
        for _ in range(3):
            mu.predict(feeds_u); me.predict(fe); md.predict(fd)
        n = a.iters
        with bench_lock():
            t = time.perf_counter()
            for _ in range(n):
                mu.predict(feeds_u)
            t_u = (time.perf_counter() - t) / n * 1000
            t = time.perf_counter()
            for _ in range(n):
                me.predict(fe); md.predict(fd)
            t_v = (time.perf_counter() - t) / n * 1000
            t = time.perf_counter()
            for _ in range(n):
                me.predict(fe); mu.predict(feeds_u); md.predict(fd)
            t_seq = (time.perf_counter() - t) / n * 1000

            def run_u():
                for _ in range(n):
                    mu.predict(feeds_u)

            def run_v():
                for _ in range(n):
                    me.predict(fe); md.predict(fd)

            th = [threading.Thread(target=run_u), threading.Thread(target=run_v)]
            t = time.perf_counter()
            for x in th:
                x.start()
            for x in th:
                x.join()
            t_par = (time.perf_counter() - t) / n * 1000
        r = {"stage": "overlap", "unet_ne_ms": round(t_u, 2), "vae_gpu_ms": round(t_v, 2),
             "sequential_ms": round(t_seq, 2), "parallel_per_frame_ms": round(t_par, 2),
             "pipelined_fps": round(1000 / t_par, 2), "sequential_fps": round(1000 / t_seq, 2)}
        print(json.dumps(r), flush=True)
        results.append(r)

    if a.json:
        with open(a.json, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
