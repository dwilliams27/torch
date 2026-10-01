#!/usr/bin/env python3
"""Build the M112 page (static HTML + inline SVG, no scripts): how fast the dream runs on the
mini, and what was measured and did not make it faster.

    python3 docs/shots/m112/brief.py --out DIR/hypnagogia-speed.html

Every number is read when this runs: data/engines.json (bench/bench_engines.py rows),
data/batch.json (bench/batch_probe.py), data/pool-*.json and data/flicker-pool-*.json (the
Neural Engine pool: bench/coreml/pool_consistency.py, bench/ws_client.py, tools/shoot.mjs),
data/gap-warp.json and data/flicker-*.json (this mission's experiments), and M115/M116 data it
reuses (docs/shots/m115/data/size-*.json for the capture size, docs/shots/m116/data/ for the
in-game dream rate and the fresh look). A missing file prints "not yet measured".
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SHOTS, "m111"))
from brief import ZONES, bars  # noqa: E402  (the M111 brief's chart helper)

OLD, NEW, THIRD = ("#5d6b80", "#8fb8ff", "#c9a2ff")


def load(path):
    p = os.path.join(SHOTS, path)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def flick(prefix, tags):
    """8-zone means over tools/shoot.mjs --flicker runs"""
    runs = [r for r in (load(f"{prefix}-{t}.json") for t in tags) if r]
    if not runs:
        return None
    m = lambda f: mean([mean([f(r["flicker"][z]) for z in ZONES]) for r in runs])
    return {"fps_walk": m(lambda z: z["dreamFps"]["walking"]), "fps_still": m(lambda z: z["dreamFps"]["still"]),
            "change_walk": m(lambda z: z["warped"]["walking"]["mean"]), "change_still": m(lambda z: z["warped"]["still"]["mean"]),
            "detail_walk": m(lambda z: z["detail"]["walking"]),
            "detail_still": m(lambda z: z["detail"]["still"]), "runs": len(runs)}


def chart(title, svg, note=None):
    return f'<p class="ct">{html.escape(title)}</p>{svg}' + (f'<p class="nt">{html.escape(note)}</p>' if note else "")


def pct(a, b):
    return f"{100 * (a / b - 1):+.0f}%"


def pool_section():
    """The Neural Engine pool: returns (html, headline or None, gaps)."""
    gaps = []
    pc = load("m112/data/pool-consistency.json")
    ws = {k: load(f"m112/data/pool-ws-{k}.json") for k in ("torch-if2", "torch-if3", "pool-if2", "pool-if3")}
    perf = {k: load(f"m112/data/pool-perf-{k}.json") for k in ("torch", "torch-if3", "pool", "pool-if3")}
    fl = {k: flick(f"m112/data/flicker-pool-{k}", ("1",)) for k in ("torch", "pool", "torch-if3", "pool-if3")}
    parts, head, shimmer = [], None, None
    if pc:
        ps = [r["psnr_db"] for r in pc["stills"]]
        md = [r["mean_abs_diff"] for r in pc["stills"]]
        parts.append(f"""<p>Both engines run the same depth-grafted model, the Neural Engine's copy built for the game's
{pc['size'][0]}x{pc['size'][1]} frames. Given the same frame, depth, prompt and noise, they paint nearly the same picture:
{min(ps):.0f} to {max(ps):.0f} dB PSNR between them on {len(ps)} test views, a mean difference of {min(md):.0f} to
{max(md):.0f} levels out of 255 (the GPU with its frame-to-frame tricks off, so each engine paints from scratch).</p>
<img src="hypnagogia-speed-pool.jpg" alt="Three test views: the input, the GPU's painting, the Neural Engine's painting, and their difference times four" style="width:100%;height:auto">
<p class="nt">From the top: the input, the GPU's painting, the Neural Engine's (NE in the charts below), and their difference x4 (bench/coreml/pool_consistency.py).</p>""")
    else:
        gaps.append("whether the two engines paint alike")
    LAB = {"torch-if2": "GPU alone, 2 sent", "torch-if3": "GPU alone, 3 sent", "pool-if2": "GPU + NE, 2 sent", "pool-if3": "GPU + NE, 3 sent"}
    if all(ws.values()):
        pe = ws["pool-if3"]["per_engine"]
        each = " and ".join(f"{v['ms_infer']:.0f} ms on {html.escape(k)}" for k, v in sorted(pe.items()))
        shown = ("torch-if2", "pool-if2", "pool-if3")
        parts.append(chart("Frames a second through the server at 512x320, with 2 or 3 frames sent ahead (higher is faster)",
                           bars("Frames a second through the server at 512x320", [(LAB[k], [ws[k]["fps_total"]]) for k in shown],
                                [("frames a second", NEW)], "frames a second", label_w=132),
                           f"bench/ws_client.py, 30 s a run, synthetic frames, {ws['pool-if3']['date'][:10]}; in the pool a frame takes {each}."))
        parts.append(f"""<p class="nt">The GPU alone with 3 sent ({ws['torch-if3']['fps_total']:.1f} a second) is left out. With one engine the server
keeps one frame painting and one waiting; a third replaces the waiting one, and this client re-sends every drop at once,
so that run mostly counts retries ({ws['torch-if3']['dropped']:,} drops in 30 s).</p>""")
        head = (ws["torch-if2"]["fps_total"], max(ws["pool-if2"]["fps_total"], ws["pool-if3"]["fps_total"]))
    else:
        gaps.append("server throughput with the pool")

    def dream(pf):
        m = re.search(r"dream=([\d.]+)/s", (pf or {}).get("perf", {}).get("lastPerf") or "")
        return float(m.group(1)) if m else None
    if all(perf.values()):
        parts.append(chart("Dreams a second in the game, riding the nave tour (higher is faster)",
                           bars("In the game: dreams a second", [(LAB[k if k.endswith("if3") else k + "-if2"], [dream(perf[k])]) for k in perf],
                                [("dreams a second", NEW)], "dreams a second", label_w=132),
                           f"tools/shoot.mjs --perf 20, WebKit 1280x720, 20 s a run, {perf['pool']['date'][:10]}."))
        pt, pp = perf["torch"]["perf"], perf["pool-if3"]["perf"]
        pace = f"{pt['fps']:.0f} a second in the harness either way" if round(pt["fps"]) == round(pp["fps"]) \
            else f"{pt['fps']:.0f} and {pp['fps']:.0f} a second in the harness"
        slow = (f"{pt['p95']:g} ms in both runs" if pt["p95"] == pp["p95"]
                else f"{pt['p95']:g} ms with the GPU alone and {pp['p95']:g} ms with the pool")
        parts.append(f"""<p>The game's own frames kept their pace: {pace}, and the slowest 5% took {slow} (99th percentile
{pt['p99']:g} and {pp['p99']:g} ms). With the default 2 sent, the pool manages only {dream(perf['pool']):.1f} dreams a second;
the {dream(perf['pool-if3']):.1f} needs 3.</p>""")
    else:
        gaps.append("the game's dream rate and frame time with the pool")
    if fl["torch"] and fl["pool"]:
        n_sent = [n for n, (t, q) in ((2, ("torch", "pool")), (3, ("torch-if3", "pool-if3"))) if fl[t] and fl[q]]
        pairs = {2: ("torch", "pool"), 3: ("torch-if3", "pool-if3")}
        for what, key in (("Standing still", "change_still"), ("Walking", "change_walk")):
            parts.append(chart(f"Frame-to-frame change {what.lower()}, 0-255 (lower is calmer)",
                               bars(f"{what}: frame-to-frame change", [(f"{n} sent", [round(fl[pairs[n][0]][key], 2), round(fl[pairs[n][1]][key], 2)])
                                                                     for n in n_sent], [("GPU alone", OLD), ("GPU + NE", NEW)], "0-255"),
                               f"tools/shoot.mjs --flicker, WebKit, {len(ZONES)} rooms, one run each"
                               + ("; walking after reprojecting the previous frame through the depth." if key == "change_walk" else ".")))
        t, q = (("torch-if3", "pool-if3") if 3 in n_sent else ("torch", "pool"))
        shimmer = (fl[q]["change_still"] / fl[t]["change_still"], fl[q]["change_walk"] / fl[t]["change_walk"])
        parts.append(f"""<p>The price is shimmer: consecutive frames come from two engines that paint a little differently, and only
the GPU's engine carries anything over from the frame before (it reuses its deep layers and looks back at its last
painting). With {n_sent[-1]} sent, frame-to-frame change is {pct(fl[q]['change_still'], fl[t]['change_still'])} standing still and
{pct(fl[q]['change_walk'], fl[t]['change_walk'])} walking; walking detail is {pct(fl[q]['detail_walk'], fl[t]['detail_walk'])}; in these
flicker runs the walking dream rate is {pct(fl[q]['fps_walk'], fl[t]['fps_walk'])} ({fl[t]['fps_walk']:.1f} to {fl[q]['fps_walk']:.1f} a second).</p>""")
    else:
        gaps.append("flicker and detail with the pool")
    cr = {k: flick(f"m112/data/flicker-carry-{k}", ("1", "2")) for k in ("torch", "any", "carry")}
    if all(cr.values()):
        t, q, c = cr["torch"], cr["any"], cr["carry"]
        parts.append(chart("Frame-to-frame change standing still, when an unchanged view goes only to the GPU, 0-255 (lower is calmer)",
                           bars("Standing still: GPU alone, GPU + NE, and GPU + NE sending repeated views only to the GPU",
                                [("GPU alone", [round(t["change_still"], 2)]), ("GPU + NE", [round(q["change_still"], 2)]),
                                 ("GPU + NE, repeats to GPU", [round(c["change_still"], 2)])], [("frame-to-frame change", NEW)], "0-255",
                                label_w=170),
                           f"tools/shoot.mjs --flicker, WebKit, {len(ZONES)} rooms, {'two runs each, averaged' if min(v['runs'] for v in cr.values()) == 2 else 'one run each'}; client ?plainwalk=1&inflight=3, "
                           "server --engine-arg held_only=0 (framing ids used only for routing); --pool-held any (then the default) for the second bar, carry for the third."))
        parts.append(f"""<p>The client can tell the server when a frame repeats the view before it (then only the fresh look did). With
that, the server can send every frame of an unchanged view only to the GPU and let the Neural Engine paint only new views.
Standing still, the change drops from {q['change_still']:.2f} to {c['change_still']:.2f}, level with the GPU alone ({t['change_still']:.2f}),
and the speed gain goes too: {c['fps_still']:.1f} dreams a second against {t['fps_still']:.1f} (GPU + NE {q['fps_still']:.1f}). Walking, it still
paints {100 * (c['fps_walk'] / t['fps_walk'] - 1):.0f}% more, {c['fps_walk']:.1f} against {t['fps_walk']:.1f} (GPU + NE {q['fps_walk']:.1f}), with change
{c['change_walk']:.2f} (GPU + NE {q['change_walk']:.2f}, GPU alone {t['change_walk']:.2f}), but with {100 * (1 - c['detail_walk'] / t['detail_walk']):.0f}% less
detail than the GPU alone ({c['detail_walk']:.2f} against {t['detail_walk']:.2f}; GPU + NE {q['detail_walk']:.2f}), which is not yet explained.</p>""")
    else:
        gaps.append("the pool keeping held views on the GPU")
    dflt = flick("m112/data/flicker-carry-default", ("1",))
    t = cr["torch"]
    if dflt and all(cr.values()):
        c = cr["carry"]
        parts.append(f"""<p>Every look now tags each frame with its view, and a server running both engines paints a held view on
the GPU alone by default. One run in the default look ({len(ZONES)} rooms, only ?inflight=3,
{load("m112/data/flicker-carry-default-1.json")["date"][:10]}) matches the runs above: standing still, change {dflt['change_still']:.2f}
({c['change_still']:.2f} above); walking, detail {dflt['detail_walk']:.2f} ({c['detail_walk']:.2f}) and {dflt['fps_walk']:.1f} dreams a second ({c['fps_walk']:.1f}).</p>""")
    else:
        gaps.append("the pool's default routing in the default look")
    more = f", and about {100 * (dflt['fps_walk'] / t['fps_walk'] - 1):.0f}% more dreams a second walking than the GPU alone" if dflt and t else ""
    parts.append(f"""<p>To try it: pull, stop play-on-mini, and on the mini, in the project folder, run <code>bench/coreml/build_models.sh graft</code>,
then <code>./run.sh --host 0.0.0.0 --engine torch_turbo,coreml_turbo --width 512 --height 320</code>, and open
<code>http://mini:8765/?inflight=3</code>. Expect a calm picture standing still{more}. Adding <code>--pool-held any</code> to the server
command lets either engine take any frame, held views included: more dreams standing still, with the shimmer above.</p>""")
    fixed = all(cr.values()) and cr["carry"]["change_still"] <= 1.1 * cr["torch"]["change_still"]
    return "\n".join(parts) or "<p>Not yet measured.</p>", head, gaps, shimmer, fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    eng = load("m112/data/engines.json")
    runs = {}
    for r in (eng["rows"] if eng else []):
        runs.setdefault((r["tag"], str(r["size"])), []).append(r)
    tags = [("stock", "stock"), ("defaults", "with tricks"), ("served", "as served")]
    sizes = [("512x320", "512x320"), ("384", "384x384")]
    engine_html, s2 = "<p>Engine timings: not yet measured.</p>", None
    need = [(t, s) for t, _ in tags for s, _ in sizes]
    fw = flick("m116/data/flicker-waking", ("1", "2"))
    if eng and all(k in runs for k in need):
        ms = lambda t, s: mean([r["e2e_ms"] for r in runs[(t, s)]])
        spread = max((max(r["e2e_ms"] for r in rs) - min(r["e2e_ms"] for r in rs)) / mean([r["e2e_ms"] for r in rs])
                     for rs in runs.values())
        n_runs = min(len(rs) for rs in runs.values())
        iters = re.search(r"--iters (\d+)", eng["by"]["defaults"])
        engine_chart = chart("One frame through the engine, milliseconds (lower is faster)",
                             bars("One frame through the engine, milliseconds", [(lab, [round(ms(t, s), 1) for t, _ in tags]) for s, lab in sizes],
                                  [(lab, c) for (_, lab), c in zip(tags, (OLD, NEW, THIRD))], "milliseconds"),
                             f"bench/bench_engines.py, {iters.group(1) if iters else '?'} frames a run, {n_runs} runs each "
                             f"(within {100 * spread:.0f}%), {eng['date']}")
        sv = runs[("served", "512x320")][0]
        st = sv["stages"]
        stage_chart = chart("Where one served 512x320 frame's time goes, milliseconds",
                            bars("Where one served 512x320 frame's time goes", [(k, [round(st[k], 1)]) for k in ("upload", "encode", "unet", "decode", "readback") if k in st],
                                 [("ms", NEW)], "milliseconds"),
                            "One run, timed stage by stage, which adds waits: the bars sum past the whole frame.")
        s0, s2, s1 = ms("stock", "512x320"), ms("served", "512x320"), ms("defaults", "512x320")
        graft = (f"costs nothing we can measure ({s2:.1f} ms with it, {s1:.1f} without, on flat depth)"
                 if abs(s2 - s1) <= max(spread, 0.02) * s2 else f"costs {s2 - s1:+.1f} ms (on flat depth)")
        reuse = (f" On the bench's synthetic walk DeepCache reuses the deep layers for {100 * st['dc_reuse']:.0f}% of frames; "
                 f"in the game, {100 * mean([load(f'm116/data/flicker-waking-{i}.json').get('dcReuseAfter') for i in (1, 2)]):.0f}%."
                 if fw and st.get("dc_reuse") is not None else "")
        engine_html = f"""<p>On the bench, outside the game, the engine as the server runs it (with the depth graft that keeps
pillars in front of walls) paints a 512x320 frame in {s2:.0f} ms, {1000 / s2:.1f} frames a second. SD-Turbo without our
tricks takes {s0:.0f} ms (that baseline already encodes and decodes with TAESD, the usual small autoencoder), so the
tricks from earlier missions make it {s0 / s2:.1f} times as fast: skipping repeated attention work, an even slimmer
decoder, and DeepCache, which reuses the model's deep layers between frames. The graft {graft}.{reuse}</p>
{engine_chart}
{stage_chart}
<p>The UNet, the network that does the painting, takes {100 * st.get('unet', 0) / st.get('total', s2):.0f}% of a frame, so big
gains mean making it do less, or running a second copy on the Neural Engine.</p>"""

    game = ""
    if fw:
        same = round(fw["fps_still"]) == round(fw["fps_walk"])
        rate = (f"about {fw['fps_still']:.0f} dreams a second, standing or walking" if same
                else f"about {fw['fps_still']:.0f} dreams a second standing and {fw['fps_walk']:.0f} walking")
        game = f"""<p>In the game you get {rate} ({fw['fps_still']:.1f} and {fw['fps_walk']:.1f}; the WebKit harness,
{len(ZONES)} rooms, {fw['runs']} runs){f", below the bench's {1000 / s2:.1f}" if s2 else ""}, probably because the browser shares the GPU
and each frame also crosses the socket and goes through JPEG.</p>"""

    pool_html, head, gaps, shimmer, fixed = pool_section()

    # what was measured and didn't help: (title, text, "costly" | "no help")
    tried = []
    sz = {n: [load(f"m115/data/size-{n}-{i}.json") for i in (1, 2)] for n in ("384", "320")}
    caps = {}
    for n in ("384", "320"):
        p = os.path.join(SHOTS, "m115", "data", f"size-shots-{n}.log")
        caps[n] = sorted(set(re.findall(r"capture (\d+x\d+) @ (\d+)°", open(p).read()))) if os.path.exists(p) else []
    if all(all(v) for v in sz.values()) and all(len(c) == 1 for c in caps.values()):
        g = lambda n, k1, k2: mean([mean([r["flicker"][z][k1][k2] for z in ZONES]) for r in sz[n]])
        (b_size, b_fov), (s_size, s_fov) = caps["384"][0], caps["320"][0]
        tried.append((f"Smaller frames ({s_size} instead of {b_size})",
                      f"at rest {pct(g('320', 'dreamFps', 'still'), g('384', 'dreamFps', 'still'))} dreams a second but "
                      f"{pct(g('320', 'detail', 'still'), g('384', 'detail', 'still'))} detail; walking "
                      f"{pct(g('320', 'dreamFps', 'walking'), g('384', 'dreamFps', 'walking'))} and "
                      f"{pct(g('320', 'detail', 'walking'), g('384', 'detail', 'walking'))}; the view also narrows from {b_fov}° to {s_fov}°. "
                      "The frames come out visibly misty (M115)", "costly"))
    else:
        gaps.append("smaller frames")
    bt = load("m112/data/batch.json")
    if bt:
        u, w = bt["per_frame_ms"], bt["unet_ms"]
        tried.append(("Painting two or four frames per call",
                      f"one full UNet pass, with DeepCache off, takes {u['1']:.0f} ms; batched, {u['2']:.0f} ms a frame in twos and {u['4']:.0f} in fours. "
                      f"Twos give the UNet {100 * (u['1'] / u['2'] - 1):.0f}% more frames, but a pair takes {w['2']:.0f} ms, so each frame "
                      "would wait for the next capture and the dream would lag behind you. The gain with DeepCache on, and the lag "
                      "itself, are not yet measured", "costly"))
    else:
        gaps.append("batching")
    gw, fwp, ff = load("m112/data/gap-warp.json"), flick("m112/data/flicker-warp", ("1", "2")), flick("m116/data/flicker-fresh", ("1", "2"))
    pr = load("m116/data/probes.json")
    if gw and fwp and fw and ff and pr and "p0-waking" in pr and "p4-fresh" in pr:
        gg = next(iter(gw.values()))
        tried.append(("DeepCache's layers moved to each new view, to get the fresh look without its speed cost",
                      f"it ran at waking's speed ({fwp['fps_walk']:.1f} dreams a second walking, against {fw['fps_walk']:.1f} for waking and "
                      f"{ff['fps_walk']:.1f} for fresh) but looked no nearer the stopped view than waking ({gg['all']['lpips']:.2f} against "
                      f"{pr['p0-waking']['all']['lpips']:.2f}; fresh {pr['p4-fresh']['all']['lpips']:.2f}; LPIPS on M116's three-room walks, "
                      "lower is nearer). Views 0.3 m apart move those layers only a pixel or two, so what went stale is probably "
                      "their content, not their position", "no help"))
    else:
        gaps.append("the reprojected DeepCache")
    fs, fa = flick("m112/data/flicker-noise-screen", ("1", "2")), flick("m112/data/flicker-noise-anchor", ("1", "2"))
    if fs and fa:
        tried.append(("Each dream's starting noise fixed to the surfaces instead of the screen, to calm walking flicker",
                      f"it made walking flicker {100 * (fa['change_walk'] / fs['change_walk'] - 1):.0f}% worse. At walking pace the view "
                      "barely changes from one frame to the next, so noise fixed to the screen probably already keeps consecutive "
                      "frames alike", "no help"))
    else:
        gaps.append("surface-anchored noise")
    tried_html = ("<ul>" + "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(v)}.</li>" for k, v, _ in tried) + "</ul>"
                  "<p>Compiling the model and a different memory layout were also slower on this GPU (bench/RESULTS.md, 2026-09-28).</p>")
    n_costly, n_none = sum(c == "costly" for *_, c in tried), sum(c == "no help" for *_, c in tried)
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    others = (f"Of the other {words.get(len(tried), len(tried))} ideas measured this week, {words.get(n_costly, n_costly)} "
              f"were faster but cost too much (misty frames, lag) and {words.get(n_none, n_none)} didn't help.")
    if head:
        calm = (" The picture shimmers more, though, so it stays off unless you switch it on"
                + (" (a first fix already makes standing still as calm as the GPU alone, and no faster)." if fixed else ".")
                if shimmer and max(shimmer) > 1.1 else "")
        intro = (f"The Neural Engine now paints alongside the GPU: {head[1]:.1f} frames a second through the server against "
                 f"{head[0]:.1f} for the GPU alone, each at its best number of frames sent ahead.{calm} {others}")
    else:
        intro = f"The Neural Engine, a separate AI processor on the mini's chip, is the next step. {others}"

    sp = load("m112/data/student-pilot.json")
    STUDENT, STUDENT_CHART = "Training one is a long distillation run; not yet tried.", ""
    if sp:
        cc = load("m115/data/cut-census.json")
        save = f"{100 * cc['cuts'][sp['cut']]['saving']:.0f}%" if cc and sp["cut"] in cc["cuts"] else "some"
        cur = sp["curve"]
        flat = next((c[0] + 1 for i, c in enumerate(cur) if all(abs(v - c[1]) < 0.1 for _, v in cur[i + 1:])), None)
        STUDENT = (f"A first pilot cut out the kinds of blocks a published slimmed-down Stable Diffusion drops (BK-SDM), which saves "
                   f"{save} of the UNet's time on a full pass (DeepCache off; the saving on a served frame is not yet measured). "
                   f"It trained the model's second half for {sp['steps']} steps, {sp['steps'] * sp['step_s_median'] / 60:.0f} minutes, "
                   f"to copy the full model's paintings, drawing at random from {sp['train_captures']} captures of "
                   f"{sp['train_captures'] // 2} views (each captured once walking, once at rest). On {sp['held_out_captures']} captures of views it "
                   f"never saw, likeness to the full model went from {sp['psnr_untrained']:.1f} to {sp['psnr_trained']:.1f} dB (PSNR)"
                   + (f" and stopped rising after {flat} steps" if flat else "")
                   + ", well short of the 30 dB where, as a rule of thumb, two paintings look nearly the same. Why it stalls is "
                   "not yet known; the published recipe trains for days, on more frames, with extra losses.")
        STUDENT_CHART = chart("A cut model learning to copy the full one: likeness on unseen views (PSNR, dB)",
                              bars("A cut model learning to copy the full one, PSNR in dB",
                                   [("untrained", [sp["psnr_untrained"]])] + [("1 step" if st_ == 0 else f"{st_ + 1} steps", [v]) for st_, v in cur],
                                   [("dB", NEW)],
                                   f"bench/student_pilot.py, cut {sp['cut']}, {sp['trainable_m']:.0f}M parameters trained, {sp['date']}"))
    else:
        gaps.append("a trained smaller model")
    dates = sorted({d for d in (eng and eng.get("date"), bt and bt.get("date"), sp and sp.get("date")) if d})
    machine = (eng or {}).get("machine") or "the mini"
    gaps.append("the game on a laptop or a phone")
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hypnagogia: how fast the dream runs</title>
<style>body{{background:#0e1117;color:#e6e9ef;font:17px/1.55 system-ui,-apple-system,sans-serif;margin:0 auto;padding:24px 18px 48px;max-width:980px}}
a{{color:#9ec9ff}}h1{{font-size:clamp(1.9rem,5vw,2.8rem);line-height:1.15;margin:.6rem 0}}h2{{margin-top:2.2rem;font-size:1.25rem}}
svg{{width:100%;height:auto;margin:.6rem 0 0}}.nt{{margin:0 0 1rem;font-size:.8rem;color:#98a2b3}}svg text{{fill:#c9d1dc;font:12px system-ui,sans-serif}}svg .k{{fill:#98a2b3}}svg .v{{fill:#e6e9ef}}
li{{margin:.35rem 0}}footer{{margin-top:2.5rem;color:#98a2b3;font-size:.85rem}}.ct{{margin:1.2rem 0 0;font-size:.95rem;color:#c9d1dc}}</style>
<a href="./index.html">← Showcase</a>
<h1>How fast the dream runs</h1>
<p>You asked for crazy optimisations. {intro}</p>
{engine_html}
{game}
<h2>The Neural Engine as a second painter</h2>
{pool_html}
<h2>Measured this week, and set aside</h2>
{tried_html}
<h2>What could still make it faster</h2>
<p>A smaller model trained to copy SD-Turbo; switching parts off without training breaks the picture (M115).
{STUDENT}</p>
{STUDENT_CHART}
<h2>Not yet measured</h2>
<p>{html.escape("; ".join(gaps)).capitalize()}.</p>
<footer>Rebuilt by <code>python3 docs/shots/m112/brief.py</code> (in the hypnagogia project) from
<code>docs/shots/m112/data/</code> and M115/M116 data. Measured {", ".join(dates) or "on dates not recorded"} on {html.escape(machine)}
(the mini); torch_turbo on the GPU (MPS), coreml_turbo on the Neural Engine.</footer>
</html>
"""
    with open(a.out, "w") as f_:
        f_.write(page)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
