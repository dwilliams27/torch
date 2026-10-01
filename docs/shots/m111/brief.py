#!/usr/bin/env python3
"""Build the M111 proof page (static HTML + inline SVG, no scripts) from the JSON in data/.

    python3 docs/shots/m111/brief.py --out DIR/hypnagogia-waking-world.html

Every number on the page is read from data/*.json when this runs: shots-*.json
(tools/shoot.mjs), flicker-*.json (--flicker), perf-*.json (--perf), smoke.json
(tools/smoke.mjs), judge.json (a blind fresh-eyes comparison, recorded with its prompt).
The sheets are copied next to the page.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ZONES = ["vestibule", "nave", "stacks", "geode", "baths", "garden", "atrium", "desert"]


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def bars(title, rows, series, unit, width=560, label_w=92):
    """Grouped horizontal bars. rows: [(label, [v per series])]; series: [(name, colour)]."""
    top, bar, gap = 30, 11, 9
    vmax = max(v for _, vs in rows for v in vs if v is not None) or 1
    span = width - label_w - 70
    h = top + len(rows) * (len(series) * bar + gap) + 26
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{html.escape(title)}">']
    x = label_w
    for name, colour in series:
        out.append(f'<rect x="{x}" y="6" width="10" height="10" fill="{colour}"/>'
                   f'<text x="{x + 14}" y="15" class="k">{html.escape(name)}</text>')
        x += 14 + 7.2 * len(name) + 18
    y = top
    for label, vs in rows:
        out.append(f'<text x="0" y="{y + bar * len(series) / 2 + 4}">{html.escape(label)}</text>')
        for (name, colour), v in zip(series, vs):
            if v is None:
                out.append(f'<text x="{label_w}" y="{y + bar - 2}" class="k">not measured</text>')
            else:
                w = max(1.0, span * v / vmax)
                out.append(f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar - 1}" rx="2" fill="{colour}"/>'
                           f'<text x="{label_w + w + 5:.1f}" y="{y + bar - 2}" class="v">{v:g}</text>')
            y += bar
        y += gap
    out.append(f'<text x="{label_w}" y="{h - 6}" class="k">{html.escape(unit)}</text></svg>')
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    outdir = os.path.dirname(os.path.abspath(a.out))
    before, after = load("shots-before.json"), load("shots-after.json")
    fn, fo, fb = load("flicker-after.json"), load("flicker-after-fovea-off.json"), load("flicker-before.json")
    pb = [load(f"perf-before-{i}.json") for i in (1, 2)]
    pa = [load(f"perf-after-{i}.json") for i in (1, 2)]
    smoke = load("smoke.json")
    phone = load("shots-phone.json")
    judge = load("judge.json")
    won = {m: [z for z in ZONES if judge[m][z] == "now"] for m in ("walking", "rest")}
    lost = {m: [z for z in ZONES if judge[m][z] == "before"] for m in ("walking", "rest")}
    rate = sorted(z["perf"]["dream"]["dreamFps"] for z in after["zones"].values())
    pcap = phone["zones"]["nave"].get("capture", {})
    for img in ("walk-before-after.jpg", "nave-before-after.jpg", "zones-after.jpg", "phone-after.jpg"):
        shutil.copy(os.path.join(HERE, img), os.path.join(outdir, "hypnagogia-m111-" + img))

    runs = [("now", "#8fb8ff", fn), ("now, foveation off", "#5d80b3", fo), ("before", "#5d6b80", fb)]
    change = {m: [[f["flicker"][z][m]["mean"] for _, _, f in runs] for z in ZONES] for m in ("still", "walking")}
    still = bars("Change between frames, standing still", list(zip(ZONES, change["still"])),
                 [(n, c) for n, c, _ in runs], "mean |Δ luma| between frames 0.1 s apart, 0-255")
    walk = bars("Change between frames, walking", list(zip(ZONES, change["walking"])),
                [(n, c) for n, c, _ in runs], "same, while walking the tour (tools/shoot.mjs --flicker)")
    calmer = sum(v[0] <= v[2] for v in change["still"])
    busier = sum(v[0] > v[2] for v in change["walking"])
    perf_rows = [(label, [p["perf"][k] for p in pb] + [p["perf"][k] for p in pa])
                 for label, k in (("median", "p50"), ("95th pct", "p95"), ("99th pct", "p99"), ("slowest", "max"))]
    perf = bars("Frame time to gl.finish", perf_rows,
                [("before, run 1", "#5d6b80"), ("before, run 2", "#46526a"), ("now, run 1", "#8fb8ff"), ("now, run 2", "#6f9ce8")],
                f"milliseconds per frame at {pb[0]['size'][0]}x{pb[0]['size'][1]}, {min(p['perf']['frames'] for p in pb + pa)}+ frames each")
    delivered = ", ".join(f"{p['perf']['fps']:g}" for p in pb) + " before; " + ", ".join(f"{p['perf']['fps']:g}" for p in pa) + " now"
    fmt = lambda runs, k: "/".join(f"{p['perf'][k]:g}" for p in runs)
    worst = [w for p in pa for w in p["perf"].get("worst", [])[:1]]
    when = (" In the new runs it came " + " and ".join(f"{w['at']:g} s" for w in worst)
            + " after timing began; not yet diagnosed.") if len(worst) == len(pa) else " Not yet diagnosed."
    slowest = (f"The slowest single frame of each run rose from {fmt(pb, 'max')} ms to {fmt(pa, 'max')} ms."
               + when)
    passed = sum(r["ok"] for r in smoke["results"])
    checks = "".join(f"<li>{'✓' if r['ok'] else '✗'} {html.escape(r['name'])}</li>" for r in smoke["results"])
    eng = after["engine"]
    cap = after["zones"]["nave"].get("capture", {})
    shots = {sh["phase"]: sh for sh in after["zones"]["nave"]["shots"]}
    t_walk2, t_rest = shots["walk2"]["t"], shots["rest"]["t"] - shots["walk3"]["t"]
    dates = sorted({d["date"][:10] for d in [before, after, fn, fo, fb, *pb, *pa, smoke, phone]})

    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hypnagogia: the dream in motion</title>
<style>body{{background:#0e1117;color:#e6e9ef;font:17px/1.55 system-ui,-apple-system,sans-serif;margin:0 auto;padding:24px 18px 48px;max-width:980px}}
a{{color:#9ec9ff}}h1{{font-size:clamp(1.9rem,5vw,2.8rem);line-height:1.15;margin:.6rem 0}}h2{{margin-top:2.2rem;font-size:1.25rem}}
img{{width:100%;border-radius:8px;display:block}}figcaption,.note{{color:#98a2b3;font-size:.9rem}}svg{{width:100%;height:auto;margin:.6rem 0}}
svg text{{fill:#c9d1dc;font:12px system-ui,sans-serif}}svg .k{{fill:#98a2b3}}svg .v{{fill:#e6e9ef}}ul{{columns:2;padding-left:1.1rem}}li{{font-size:.9rem}}
footer{{margin-top:2.5rem;color:#98a2b3;font-size:.85rem}}</style>
<a href="./index.html">← Showcase</a>
<h1>Walking is now mostly as crisp as standing still</h1>
<p>Hypnagogia paints its world with SD-Turbo as you move through it. Before this work, the
paint went soft and milky whenever you walked, because each surface only had the atlas, a
blurry average of many views. Now the newest few results (the live views) are projected
straight onto the world at full resolution, and the model refines its own last picture.
Each capture is aimed at where you will be when its result arrives. About half the
captures are foveated: they zoom in on the middle of the view with 1.7 times the pixels
per degree, and where results overlap the sharper one shows. Captures roughly follow the
screen's shape: {cap.get('size', ['?', '?'])[0]}x{cap.get('size', ['?', '?'])[1]} on a 16:9 display.</p>
<p>A fresh Claude session, told nothing about which build was which, compared every frame
below at full size. Walking, it found the new build sharper in {len(won['walking'])} of {len(ZONES)} zones;
{' and '.join(f"the {z} ({judge['walking_margin'][z]} margin)" for z in lost['walking'])} went the other way. Standing
still, it preferred the new build in {len(won['rest'])} of {len(ZONES)}. Asked whether the new build is as crisp walking
as standing, it answered "{judge['now_walking_vs_rest']}".</p>
<p>What it still disliked in the new build: {'; '.join(judge['now_weak_spots'])}.</p>
<h2>The same walk, before and after</h2>
<figure><img src="hypnagogia-m111-walk-before-after.jpg" alt="Eight zones at the same moment of the same walk: soft and milky before, painted scenes after">
<figcaption>Each row is one zone, {t_walk2:.0f} seconds into the same scripted walk at {after['speed']:g} m/s. Left: the client before this work. Right: the client now. Headless WebKit, dreaming {rate[0]:.1f}-{rate[-1]:.1f} times a second.</figcaption></figure>
<figure><img src="hypnagogia-m111-nave-before-after.jpg" alt="The drowned nave at three moments, before and after">
<figcaption>The drowned nave: early in the walk, late in the walk, and {t_rest:.0f} seconds after stopping.</figcaption></figure>
<h2>Standing still, frames change no more than before in {calmer} of {len(ZONES)} zones</h2>
{still}
<p class="note">A new result from an unchanged view blends into the last ones instead of replacing them. The middle bar is the current build with foveated captures turned off.</p>
<h2>Walking, frames change more than before in {busier} of {len(ZONES)} zones</h2>
{walk}
<p class="note">Two things raise this number: sharp detail moving across the screen counts as change, and each new dream re-invents part of the picture. Foveated captures sharpen the middle of the view and re-invent more of it (compare the middle bar). Noise anchored to the world, the usual fix for re-invention, looked less steady in walking filmstrips (judged by eye, not measured), so it was dropped.</p>
<h2>Typical frame time barely moved</h2>
{perf}
<p class="note">Median frame time went from {fmt(pb, 'p50')} ms to {fmt(pa, 'p50')} ms. {slowest} Frames delivered per second in the same runs: {delivered}. The headless browser shares the mini's GPU and CPU with the model at background priority, so those rates are low. Per-frame cost is the fair comparison.</p>
<h2>{passed} of {len(smoke['results'])} checks pass</h2>
<ul>{checks}</ul>
<p class="note">The iPhone runs are WebKit with touch emulation. They do not apply iOS's rule that sound may only start inside a tap.</p>
<h2>Every zone, at rest</h2>
<figure><img src="hypnagogia-m111-zones-after.jpg" alt="Eight zones after standing still: gilded vestibule, drowned nave, library stair, geode, neon baths, garden, atrium, desert">
<figcaption>{t_rest:.0f} seconds after stopping at the end of the walk.</figcaption></figure>
<h2>On a phone</h2>
<figure><img src="hypnagogia-m111-phone-after.jpg" alt="Four zones on an emulated iPhone held upright">
<figcaption>WebKit emulating an iPhone 15 held upright, not a real phone. The capture turns tall ({pcap.get('size', ['?', '?'])[0]}x{pcap.get('size', ['?', '?'])[1]}), and the phone gets a smaller atlas and three live views.</figcaption></figure>
<h2>Not yet measured</h2>
<p>Real Safari and a real iPhone (touch feel, sound on the first tap). Frame time and dream rate on a laptop or phone client, which is how the game is actually played. How often the dream re-invents objects while you walk (the reviewer saw it happen; no measure separates it from sharp detail moving).</p>
<footer>Rebuilt by <code>python3 docs/shots/m111/brief.py</code> (in the hypnagogia project) from
<code>docs/shots/m111/data/*.json</code>. Measured {', '.join(dates)} on a Mac mini (M4 Pro, 48 GB):
headless WebKit (walks, stability, phone) and Chromium (frame time) through Playwright, <code>{html.escape(str(eng.get('engine')))}</code>
({html.escape(str(eng.get('model')))}, {eng.get('width')}x{eng.get('height')} budget, {html.escape(str(eng.get('device')))}), tools in
<code>tools/</code>.</footer>
</html>
"""
    with open(a.out, "w") as f:
        f.write(page)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
