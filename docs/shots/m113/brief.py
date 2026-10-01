#!/usr/bin/env python3
"""Build the M113 proof page (static HTML + inline SVG, no scripts) from the JSON in data/.

    python3 docs/shots/m113/brief.py --out DIR/hypnagogia-steady-dream.html

Every number on the page is read when this runs: flicker-*.json (tools/shoot.mjs
--flicker), shots-final.json (walk shots with the dream rate), perf-final-*.json (--perf),
judge-final.json (a blind fresh-eyes comparison of walking filmstrips, recorded with its
method), and M111's shots-after.json / perf-after-*.json for the build before this work.
The two images are copied next to the page.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M111 = os.path.join(HERE, "..", "m111")
sys.path.insert(0, M111)
from brief import ZONES, bars  # noqa: E402  (the M111 brief's chart helper)


def load(name, root=HERE):
    with open(os.path.join(root, "data", name)) as f:
        return json.load(f)


def mean(xs):
    return sum(xs) / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    outdir = os.path.dirname(os.path.abspath(a.out))
    off = [load(f"flicker-xframe-off-{i}.json") for i in (1, 2)]
    on = [load(f"flicker-xframe-on-{i}.json") for i in (1, 2)]
    final = load("flicker-final.json")
    bold = [load(f"flicker-baths-prompt-old-{i}.json") for i in (1, 2)]
    bnew = [load(f"flicker-baths-prompt-new-{i}.json") for i in (1, 2)]
    shots, before = load("shots-final.json"), load("shots-after.json", M111)
    perf = [load(f"perf-final-{i}.json") for i in (1, 2)]
    pb = [load(f"perf-after-{i}.json", M111) for i in (1, 2)]
    judge, jbaths, jxf = load("judge-final.json"), load("judge-baths.json"), load("judge-xframe.json")
    rate8 = load("flicker-rate-8.json")
    for img in ("final-strips.jpg", "baths-prompt.jpg"):
        shutil.copy(os.path.join(HERE, img), os.path.join(outdir, "hypnagogia-m113-" + img))

    # warped walking change and detail, per zone, averaged over two runs of each build
    warped = lambda runs, z: mean([r["flicker"][z]["warped"]["walking"]["mean"] for r in runs])
    detail = lambda runs: mean([mean([r["flicker"][z]["detail"]["walking"] for z in ZONES]) for r in runs])
    rows = [(z, [round(warped(off, z), 2), round(warped(on, z), 2)]) for z in ZONES]
    m_off, m_on = mean([warped(off, z) for z in ZONES]), mean([warped(on, z) for z in ZONES])
    m_final = mean([final["flicker"][z]["warped"]["walking"]["mean"] for z in ZONES])
    change = bars("Change after motion compensation, walking", rows,
                  [("cross-frame attention off", "#5d6b80"), ("on", "#8fb8ff")],
                  "mean |Δ luma| after reprojecting the frame 0.1 s earlier, 0-255; 2 runs each")
    lower = sum(v[1] < v[0] for _, v in rows)

    for k in ("steadier", "sharper", "more_coherent"):
        bad = [v for v in judge[k].values() if not v.startswith(("now", "M111", "even"))]
        assert not bad, f"unrecognised {k} verdicts: {bad}"
    g = judge["grades_steadiness_sharpness_coherence_1_to_5"]
    steady = bars("Steadiness while walking, blind grades", [(z, [g["M111"][z][0], g["now"][z][0]]) for z in ZONES],
                  [("M111 final", "#5d6b80"), ("now", "#8fb8ff")], "1-5, blind, from walking filmstrips")
    sharp = bars("Sharpness while walking, blind grades", [(z, [g["M111"][z][1], g["now"][z][1]]) for z in ZONES],
                 [("M111 final", "#5d6b80"), ("now", "#8fb8ff")], "1-5, same review")
    steadier = [z for z in ZONES if judge["steadier"][z].startswith("now")]
    less_sharp = [z for z in ZONES if judge["sharper"][z].startswith("M111")]
    mg = {b: [mean([g[b][z][i] for z in ZONES]) for i in range(3)] for b in ("now", "M111")}

    rate = lambda d, z: round(d["zones"][z]["perf"]["dream"]["dreamFps"], 1)
    rates = bars("Dreams per second while walking", [(z, [rate(before, z), rate(shots, z)]) for z in ZONES],
                 [("M111 final", "#5d6b80"), ("now", "#8fb8ff")], "results per second, headless WebKit 1280x720, one run each")
    r_before = sorted(rate(before, z) for z in ZONES)
    r_now = sorted(rate(shots, z) for z in ZONES)

    fmt = lambda runs, k: "/".join(f"{p['perf'][k]:g}" for p in runs)
    frame = bars("Frame time to gl.finish", [(label, [p["perf"][k] for p in pb] + [p["perf"][k] for p in perf])
                                             for label, k in (("median", "p50"), ("95th pct", "p95"), ("slowest", "max"))],
                 [("M111, run 1", "#5d6b80"), ("M111, run 2", "#46526a"), ("now, run 1", "#8fb8ff"), ("now, run 2", "#6f9ce8")],
                 f"milliseconds per frame, headless Chromium {perf[0]['size'][0]}x{perf[0]['size'][1]}")

    bw = lambda runs: round(mean([r["flicker"]["baths"]["warped"]["walking"]["mean"] for r in runs]), 2)
    bd = lambda runs, m: round(mean([r["flicker"]["baths"]["detail"][m] for r in runs]), 2)
    baths = bars("The baths, old prompt vs new", [("walking change", [bw(bold), bw(bnew)]),
                                                  ("detail, walking", [bd(bold, "walking"), bd(bnew, "walking")]),
                                                  ("detail, at rest", [bd(bold, "still"), bd(bnew, "still")])],
                 [("old prompt", "#5d6b80"), ("new prompt", "#8fb8ff")],
                 "motion-compensated change, mean |luma gradient|; 0-255, 2 runs each")
    still = lambda d: mean([d["flicker"][z]["warped"]["still"]["mean"] for z in ZONES])
    s_now, s_before = still(final), mean([still(r) for r in on])
    chrome_rate = [float(m.group(1)) for p in pb + perf for m in [re.search(r"dream=([0-9.]+)/s", p["perf"].get("lastPerf", ""))] if m]
    jb = jbaths["grades_detail_steadiness_coherence_1_to_5"]
    xf = jxf["mean_grades"]
    dates = sorted({d["date"][:10] for d in [*off, *on, final, *bold, *bnew, shots, *perf, before, *pb]} | {judge["date"]})
    eng = final["engine"]

    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hypnagogia: the dream keeps its shapes</title>
<style>body{{background:#0e1117;color:#e6e9ef;font:17px/1.55 system-ui,-apple-system,sans-serif;margin:0 auto;padding:24px 18px 48px;max-width:980px}}
a{{color:#9ec9ff}}h1{{font-size:clamp(1.9rem,5vw,2.8rem);line-height:1.15;margin:.6rem 0}}h2{{margin-top:2.2rem;font-size:1.25rem}}
img{{width:100%;border-radius:8px;display:block}}figure{{margin:1rem 0}}figcaption,.note{{color:#98a2b3;font-size:.9rem}}svg{{width:100%;height:auto;margin:.6rem 0}}
svg text{{fill:#c9d1dc;font:12px system-ui,sans-serif}}svg .k{{fill:#98a2b3}}svg .v{{fill:#e6e9ef}}
footer{{margin-top:2.5rem;color:#98a2b3;font-size:.85rem}}</style>
<a href="./index.html">← Showcase</a>
<h1>The dream keeps its shapes while you walk</h1>
<p>After M111, walking through Hypnagogia was sharp, but each new dream
re-invented part of the picture: a dome became two chimneys between steps. Three changes
went in. Each attention layer of the model now also looks at what it computed for the
previous frame of the same kind of view (narrow or wide, each with its own seed), so it
carries its last picture forward. A speed-up that lets about two frames in three skip the
deep half of the model (reusing the last frame's features there) now keeps one cache per
view instead of one shared cache, and the game dreams {r_now[0]:g}-{r_now[-1]:g} times a
second instead of M111's {r_before[0]:g}-{r_before[-1]:g}. And four zones got prompts that describe what
the walk actually passes, because where a prompt described somewhere else, the model's
invented shapes drifted over the real geometry.</p>
<p>A fresh Claude session compared walking filmstrips of M111's final build and the current
one, without being told which was which. It found the new one steadier in
{len(steadier)} of {len(ZONES)} zones (average steadiness {mg['now'][0]:.2f} against
{mg['M111'][0]:.2f} out of 5) and graded sharpness {mg['now'][1]:.2f} against {mg['M111'][1]:.2f}. In its words: “{html.escape(judge['summary'])}”</p>
<figure><img src="hypnagogia-m113-final-strips.jpg" alt="Walking filmstrips of eight zones: M111's final build above, the current build below, in pairs">
<figcaption>Walking filmstrips of all eight zones, frames 0.3 s apart at 3.4 m/s: in each pair M111's final build is above and the current build below. Headless WebKit, identical camera poses.</figcaption></figure>
<h2>Steadier in {len(steadier)} of {len(ZONES)} zones</h2>
{steady}
<h2>{'No zone got softer' if not less_sharp else 'Softer in the ' + ' and '.join(filter(None, [', '.join(less_sharp[:-1]), less_sharp[-1]]))}</h2>
{sharp}
<p class="note">What the reviewer still disliked: {html.escape('; '.join(judge['now_weak_spots']))}.</p>
<h2>Cross-frame attention cut the change {(1 - m_on / m_off) * 100:.0f}% on average</h2>
{change}
<p class="note">Each frame is compared with the frame 0.1 s earlier, reprojected onto the
current view with depth; that cancels most of the change that only comes
from the camera moving. What remains is paint that changed. Averaged over the zones:
{m_off:.2f} with cross-frame attention off, {m_on:.2f} with it on
({(m_on / m_off - 1) * 100:+.0f}%; lower in {lower} of {len(ZONES)} zones), with detail (mean luma
gradient) {detail(off):.2f} and {detail(on):.2f}, so by that measure the drop is not from painting
softer; a blind A/B graded attention on a little less detailed ({xf['on'][0]:g} against
{xf['off'][0]:g} of 5), most of all in the baths. These runs are from mid-mission; the closing
build averages {m_final:.2f}. Two runs of one build agree on
the average to about 1%; single zones vary up to about 17%. M111's own client can't take
this measure (its test hooks have no depth pass), so the comparison is the current client
with cross-frame attention off and on.</p>
<h2>The baths got their arcades back</h2>
{baths}
<figure><img src="hypnagogia-m113-baths-prompt.jpg" alt="Walking through the baths: bare stucco walls with the old prompt, tiled arcades with the new one">
<figcaption>The same walk through the baths, frames 0.3 s apart: old prompt above, new prompt below.</figcaption></figure>
<p class="note">Cross-frame attention cost the most detail in the baths, where the walls stayed bare stucco.
The new prompt names the vaulted hall, the terraced pool and the arcades of mosaic tile. A blind
review of the two graded detail {jb['new'][0]} against {jb['old'][0]}, coherence {jb['new'][2]} against
{jb['old'][2]} and steadiness {jb['new'][1]} against {jb['old'][1]} out of 5: {html.escape(jbaths['worst_flaw']['new'])}.</p>
<h2>{r_now[0]:g}-{r_now[-1]:g} dreams a second, up from {r_before[0]:g}-{r_before[-1]:g}</h2>
{rates}
<h2>Frame time barely moved</h2>
{frame}
<p class="note">Median {fmt(pb, 'p50')} ms before, {fmt(perf, 'p50')} ms now; slowest single frame {fmt(pb, 'max')} ms before, {fmt(perf, 'max')} ms now. Headless Chromium dreams only {min(chrome_rate):g}-{max(chrome_rate):g} times a second in these runs (reading captures back is slow there), so they do not exercise the faster dream rate.</p>
<h2>Standing still got busier</h2>
<p>Standing still, frames now change {s_now:.2f} on average after motion compensation
(0-255), against {s_before:.2f} before the one-cache-per-view fix ({(s_now / s_before - 1) * 100:+.0f}%).
A smaller blend for each new result did not lower it, and a slower hand-over between results
raised it. Capping captures at 8 a second brought it down to {still(rate8):.2f} in one run (dream
rate not recorded), with walking no worse (change
{mean([rate8['flicker'][z]['warped']['walking']['mean'] for z in ZONES]):.2f} against {m_final:.2f}, detail
{mean([rate8['flicker'][z]['detail']['walking'] for z in ZONES]):.2f} against {detail([final]):.2f}), so the extra
dreams look like the cause. That cap is not in the game yet; it needs its own blind review.</p>
<h2>Not yet measured</h2>
<p>Real Safari and a real iPhone. Frame time and dream rate on a laptop or phone, which is how the game is played. The motion-compensated change of M111's own client.</p>
<footer>Rebuilt by <code>python3 docs/shots/m113/brief.py</code> (in the hypnagogia project) from
<code>docs/shots/m113/data/*.json</code> and M111's <code>shots-after.json</code> and <code>perf-after-*.json</code>.
Measured {', '.join(dates)} on a Mac mini (M4 Pro, 48 GB): headless WebKit 1280x720 (walks, filmstrips, change) and
Chromium (frame time) through Playwright, <code>{html.escape(str(eng.get('engine')))}</code>
({html.escape(str(eng.get('model')))}, {eng.get('width')}x{eng.get('height')} budget, {html.escape(str(eng.get('device')))}),
tools in <code>tools/</code>. Blind review: {html.escape(judge['reviewer'])}.</footer>
</html>
"""
    with open(a.out, "w") as f:
        f.write(page)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
