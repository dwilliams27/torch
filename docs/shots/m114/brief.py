#!/usr/bin/env python3
"""Build the M114 proof page (static HTML + inline SVG, no scripts) from the JSON in data/.

    python3 docs/shots/m114/brief.py --out DIR/hypnagogia-lucid-dream.html

Every measured number on the page is read when this runs (set-up constants such as the
0.35 s time constant are written in): lens-probe.json (the centre lens measured on the
29 September build), depth-graft-probe.json and depth-graft-controls.json
(bench/depth_graft.py), flicker-*.json and perf-*.json (tools/shoot.mjs: the 29 September
build, client d4cf3a1c with the stock engine, against this one with the depth graft), and
judge.json with judge-key.json (a blind fresh-eyes review). The images are copied next to
the page.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "m111"))
from brief import ZONES, bars  # noqa: E402  (the M111 brief's chart helper)

OLD, NEW = ("#5d6b80", "#8fb8ff")


def load(name):
    with open(os.path.join(HERE, "data", name)) as f:
        return json.load(f)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    outdir = os.path.dirname(os.path.abspath(a.out))
    lens = load("lens-probe.json")
    probe, ctrl = load("depth-graft-probe.json"), load("depth-graft-controls.json")
    fb = [load(f"flicker-base-{i}.json") for i in (1, 2)]
    fn = [load(f"flicker-final-{i}.json") for i in (1, 2)]
    f3 = load("flicker-final-3.json")
    fw, ft, fg = load("flicker-warp15.json"), load("flicker-tau0.json"), load("flicker-stock.json")
    pb = [load(f"perf-base-{i}.json") for i in (1, 2)]
    pn = [load(f"perf-final-{i}.json") for i in (1, 2)]
    judge, key = load("judge.json"), load("judge-key.json")
    for img in ("lens.jpg", "pillar.jpg", "graft.jpg", "graft-weight.jpg"):
        shutil.copy(os.path.join(HERE, img), os.path.join(outdir, "hypnagogia-m114-" + img))

    # 1. the lens: how differently the centre was painted in the 29 September build
    lz = list(lens["zones"])
    lens_chart = bars("Colour and composition difference, narrow views on vs off", [(z, [lens["zones"][z]["frame"], lens["zones"][z]["centre"]]) for z in lz],
                      [("whole frame", OLD), ("central 30%", NEW)], "mean |Δ| after a 6 px blur, 0-255, the 29 September build")

    # 2. the depth graft on nine views: silhouette contrast, stock vs graft
    # (the first probe's atrium capture had no depth; its rerun with depth is in the controls file)
    R, C = probe["results"], ctrl["results"]
    src = lambda v: C if v == "atrium" else R   # noqa: E731
    views = [v for v in ("pillar", "vestibule", "nave", "stacks", "geode", "baths", "garden", "atrium", "desert")
             if src(v).get(f"{v}|stock|s0.6", {}).get("ratio") is not None]
    g_rows = [(v, [round(src(v)[f"{v}|stock|s0.6"]["ratio"], 2), round(src(v)[f"{v}|graft0.8|s0.6"]["ratio"], 2)]) for v in views]
    graft_chart = bars("Luma step across depth silhouettes ÷ within surfaces", g_rows, [("stock SD-Turbo", OLD), ("depth graft 0.8", NEW)],
                       "one frame per view, strength 0.6; higher = more separated")
    up = sum(r[1][1] > r[1][0] for r in g_rows)
    unscored = [v for v in ("geode", "atrium") if v not in views]
    flip_true, flip_flip = C["pillar|graft0.8-flip|s0.6"]["ratio"], C["pillar|graft0.8-flip|s0.6"]["flip_match"]

    # 3. in the game: standing still, walking, detail, silhouettes (two runs of each build)
    Z = ZONES
    f = lambda runs, z, k1, k2: mean([r["flicker"][z][k1][k2]["mean"] if k1 == "warped" else r["flicker"][z][k1][k2] for r in runs])
    still = bars("Change standing still, after motion compensation", [(z, [round(f(fb, z, "warped", "still"), 2), round(f(fn, z, "warped", "still"), 2)]) for z in Z],
                 [("29 Sep build", OLD), ("now", NEW)], "mean |Δ luma| against the frame 0.1 s earlier, reprojected; 0-255, 2 runs each")
    walk = bars("Change walking, after motion compensation", [(z, [round(f(fb, z, "warped", "walking"), 2), round(f(fn, z, "warped", "walking"), 2)]) for z in Z],
                [("29 Sep build", OLD), ("now", NEW)], "same measure, walking at 1.7 m/s; 2 runs each")
    sil = lambda runs, z: mean([r["flicker"][z]["edges"]["silStill"] / r["flicker"][z]["edges"]["texStill"] for r in runs
                                if r["flicker"][z].get("edges", {}).get("texStill")])
    sil_chart = bars("In the game, standing still: silhouette step ÷ surface step", [(z, [round(sil(fb, z), 2), round(sil(fn, z), 2)]) for z in Z],
                     [("29 Sep build", OLD), ("now", NEW)], "160x90 grid, 40 frames, 2 runs each")
    s_b, s_n = mean([f(fb, z, "warped", "still") for z in Z]), mean([f(fn, z, "warped", "still") for z in Z])
    w_b, w_n = mean([f(fb, z, "warped", "walking") for z in Z]), mean([f(fn, z, "warped", "walking") for z in Z])
    d_b = {m: mean([f(fb, z, "detail", m) for z in Z]) for m in ("still", "walking")}
    d_n = {m: mean([f(fn, z, "detail", m) for z in Z]) for m in ("still", "walking")}
    calmer = sum(f(fn, z, "warped", "still") < f(fb, z, "warped", "still") for z in Z)
    sil_up = sum(sil(fn, z) > sil(fb, z) for z in Z)
    # ablations (one run each, same client): which part did what
    one = lambda r, k1, k2: mean([r["flicker"][z][k1][k2]["mean"] if k1 == "warped" else r["flicker"][z][k1][k2] for z in Z])
    one_sil = lambda r: mean([r["flicker"][z]["edges"]["silStill"] / r["flicker"][z]["edges"]["texStill"] for z in Z if r["flicker"][z].get("edges", {}).get("texStill")])
    abl = [("now", f3), ("warp on", fw), ("no average", ft), ("stock engine", fg)]
    abl_chart = bars("What each part does (one run each)", [(lab, [round(one(r, "warped", "still"), 2), round(one(r, "warped", "walking"), 2),
                                                                   round(one(r, "detail", "still"), 2), round(one_sil(r), 2)]) for lab, r in abl],
                     [("rest change", "#8fb8ff"), ("walk change", "#5d6b80"), ("rest detail", "#c9a86a"), ("silhouettes", "#8fd19e")],
                     "8-zone means, one run each")
    # run-to-run agreement of the final build's 8-zone means (rest and walking change, detail)
    agree = 100 * max(abs(a_ - b_) / ((a_ + b_) / 2) for a_, b_ in [
        (one(fn[0], k1, k2), one(fn[1], k1, k2)) for k1, k2 in (("warped", "still"), ("warped", "walking"), ("detail", "still"), ("detail", "walking"))])
    rate = lambda runs, ph: mean([r["flicker"][z]["dreamFps"][ph] for r in runs for z in Z if r["flicker"][z].get("dreamFps")])

    # 4. frame time
    frame = bars("Frame time to gl.finish", [(lab, [p["perf"][k] for p in pb] + [p["perf"][k] for p in pn]) for lab, k in (("median", "p50"), ("95th pct", "p95"), ("slowest", "max"))],
                 [("29 Sep, run 1", OLD), ("29 Sep, run 2", "#46526a"), ("now, run 1", NEW), ("now, run 2", "#6f9ce8")],
                 f"milliseconds, headless Chromium {pn[0]['size'][0]}x{pn[0]['size'][1]}, 20 s on the nave tour")
    fmt = lambda runs, k: "/".join(f"{p['perf'][k]:g}" for p in runs)

    # 5. blind review
    J = judge["zones"]
    who = lambda z, ab: key[z][ab]                      # "A"/"B" -> "base"/"new"
    pick = lambda z, k: "even" if J[z][k] == "even" else who(z, J[z][k])
    lens_old = sum(J[z][f"lens_{ab}"] for z in Z for ab in "AB" if who(z, ab) == "base")
    lens_new = sum(J[z][f"lens_{ab}"] for z in Z for ab in "AB" if who(z, ab) == "new")
    n_geo_new = sum(pick(z, "geometry") == "new" for z in Z)
    n_geo_old = sum(pick(z, "geometry") == "base" for z in Z)
    n_pref_new = sum(pick(z, "preferred") == "new" for z in Z)
    n_pref_old = sum(pick(z, "preferred") == "base" for z in Z)
    n_beauty_new = sum(pick(z, "beauty") == "new" for z in Z)
    n_beauty_old = sum(pick(z, "beauty") == "base" for z in Z)
    lens_where = [J[z][f"lens_{ab}_where"] for z in Z for ab in "AB" if who(z, ab) == "base" and J[z][f"lens_{ab}"]]
    arts_new = [f"{z}: {J[z][f'artifacts_{ab}']}" for z in Z for ab in "AB" if who(z, ab) == "new" and J[z][f"artifacts_{ab}"]]

    dates = sorted({d["date"][:10] for d in [*fb, *fn, f3, fw, ft, fg, *pb, *pn]} | {judge["date"], lens["date"]})
    eng_b, eng_n = fb[0]["engine"], fn[0]["engine"]
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hypnagogia: a pillar stands in front of the wall</title>
<style>body{{background:#0e1117;color:#e6e9ef;font:17px/1.55 system-ui,-apple-system,sans-serif;margin:0 auto;padding:24px 18px 48px;max-width:980px}}
a{{color:#9ec9ff}}h1{{font-size:clamp(1.9rem,5vw,2.8rem);line-height:1.15;margin:.6rem 0}}h2{{margin-top:2.2rem;font-size:1.25rem}}
img{{width:100%;border-radius:8px;display:block}}figure{{margin:1rem 0}}figcaption,.note{{color:#98a2b3;font-size:.9rem}}svg{{width:100%;height:auto;margin:.6rem 0}}
svg text{{fill:#c9d1dc;font:12px system-ui,sans-serif}}svg .k{{fill:#98a2b3}}svg .v{{fill:#e6e9ef}}
footer{{margin-top:2.5rem;color:#98a2b3;font-size:.85rem}}</style>
<a href="./index.html">← Showcase</a>
<h1>A pillar stands in front of the wall</h1>
<p>After playing the 29 September build, you wrote that a pillar "completely blends into
the wall behind it unless I move", that a lens-like rectangle sat over the middle of your
view, that the dream was still a bit nauseating, and that you'd like options to try. Every
comparison below is against that build.</p>
<figure><img src="hypnagogia-m114-pillar.jpg" alt="Left: the stock model paints a flat wall where two columns stand; right: the columns stand in front of the wall. Below: a garden painted over its real layout, then following it">
<figcaption>Standing still in the nave, looking at two columns 6.5 m away in front of a wall 14 m away (top), and in the garden (bottom). Left: the 29 September build. Right: now. Same pose, headless WebKit, after 8 s of dreaming.</figcaption></figure>

<h2>The model now paints the geometry</h2>
<p>The model used to keep the light and dark layout of the view it was sent and paint over
any shape whose tone matched its background: a stone pillar in front of a stone wall
became part of the wall, even with feedback, cross-frame attention and DeepCache all off,
and even when the game lit the pillar brighter or ringed it with shadow. So the model now
also gets the view's depth. There is no depth version of SD-Turbo, so one was grafted:
SD-Turbo plus 0.8 times the difference between Stability's depth-conditioned SD 2 and the
SD 2 it was trained from, with the depth model's extra input channel. Nothing was trained,
and a dream takes no longer to paint.</p>
{graft_chart}
<p class="note">Higher in {up} of {len(g_rows)} views that the measure can score ({' and '.join(unscored)}
{'have' if len(unscored) > 1 else 'has'} too few depth edges in frame). To check that the model follows the depth
rather than just changing style, the same pillar view was given its depth mirrored left to
right: the picture then matches the mirrored depth far better than the true one
({flip_flip:.2f} against {flip_true:.2f} on the chart's measure, graft weight 0.8). At a graft weight of 1.0
the walls came out blotchier (below); 0.8 kept more of their texture.</p>
<figure><img src="hypnagogia-m114-graft.jpg" alt="Seven views: what the model is given, the stock result, and the result with the depth graft">
<figcaption>Left to right: what the model is given, stock SD-Turbo, with the depth graft (strength 0.6, one frame each, no frame-to-frame state). Rows, top to bottom: pillar, vestibule, geode, baths, garden, desert, atrium.</figcaption></figure>
<figure><img src="hypnagogia-m114-graft-weight.jpg" alt="The geode and the desert: the input, then graft weights 0.8 and 1.0; 1.0 is blotchier">
<figcaption>Graft weight 0.8 (middle) against 1.0 (right), same frames: 1.0 smears the geode's crystal and mottles the desert's walls.</figcaption></figure>
<h2>In the game, standing still</h2>
{sil_chart}
<p class="note">Higher in {sil_up} of {len(Z)} zones. This measure is coarse (a 160x90 grid), so read it as
support for the pictures.</p>

<h2>No lens</h2>
<p>The lens was real. Half the captures at rest zoomed in on the middle of the view for
more detail. They had their own seed and history, so the model painted things there that
the wider view did not have: a shaft of light, a fountain, a figure. The zoomed region's
border faded out over a band about where you saw the edge of the lens.</p>
{lens_chart}
<figure><img src="hypnagogia-m114-lens.jpg" alt="Four zones in the 29 September build: the frame with its zoomed centre views, without them, and a map of where they differ">
<figcaption>The 29 September build, dream paused: with its zoomed centre views (left), without them (middle), and where the colour and composition differ (right).</figcaption></figure>
<p>Those zoomed captures are off now, so every capture paints the whole view. Zooming inside
one image instead (1.5 times the pixels per degree in the middle, fewer at the edges) has no
border, but it measured less detailed even in the middle of the view, so it is off too.
Detail (mean luma gradient) in the middle half of the frame is now {mean([f(fn, z, 'detail', 'stillCentre') for z in Z]):.2f} at rest against
{mean([f(fb, z, 'detail', 'stillCentre') for z in Z]):.2f} before, and {mean([f(fn, z, 'detail', 'walkingCentre') for z in Z]):.2f} against {mean([f(fb, z, 'detail', 'walkingCentre') for z in Z]):.2f} walking.</p>

<h2>Standing still, {abs(s_n / s_b - 1) * 100:.0f}% less change</h2>
{still}
<p class="note">Averaged over the zones: {s_b:.2f} before, {s_n:.2f} now ({(s_n / s_b - 1) * 100:+.0f}%), lower in {calmer} of {len(Z)}.
Two changes share the credit. Standing still, each view now follows its newest result
gradually (a running average with a 0.35 s time constant) instead of stacking the last few
and jumping when the oldest drops out; and the depth graft pins the shapes, so successive
results agree more. With either one taken away, change at rest is about 0.4 (chart below).
Detail at rest went from {d_b['still']:.2f} to {d_n['still']:.2f}.</p>
<h2>Walking, {abs(w_n / w_b - 1) * 100:.0f}% less change and {abs(d_n['walking'] / d_b['walking'] - 1) * 100:.0f}% less detail</h2>
{walk}
<p class="note">Walking: {w_b:.2f} before, {w_n:.2f} now ({(w_n / w_b - 1) * 100:+.0f}%); detail {d_b['walking']:.2f} and {d_n['walking']:.2f} ({(d_n['walking'] / d_b['walking'] - 1) * 100:+.0f}%).
Dreams per second standing still (WebKit harness): {rate(fb, 'still'):.1f} before, {rate(fn, 'still'):.1f} now.</p>
<h2>Which part does what</h2>
{abl_chart}
<p class="note">One run each of the shipped client with one part changed. Without the running
average, change at rest is {one(ft, 'warped', 'still'):.2f} against {one(f3, 'warped', 'still'):.2f}; with the stock engine the
silhouette ratio is {one_sil(fg):.2f} against {one_sil(f3):.2f}; with the warp, rest detail is {one(fw, 'detail', 'still'):.2f}
against {one(f3, 'detail', 'still'):.2f}. Two runs of one build agree on these 8-zone means to within {agree:.0f}%;
single zones vary more. The two runs in the charts above predate the client's last change
(it now copies a result only once a framing is held); a third run on the shipped client
agrees with them.</p>

<h2>Looks to try</h2>
<p>Press 1 to 5, or use the buttons at the top of the settings panel (the gear on a phone):
<b>waking</b> (the default), <b>drifting</b> (slow, soft changes), <b>lucid</b> (the
architecture shows through), <b>fever</b> (vivid and restless) and <b>lens</b> (the
29 September centre lens, to compare). The game remembers the last one.</p>

<h2>A blind review</h2>
<p>A fresh Claude session saw each zone's real geometry (the undreamt blueprint) beside a
standing and a walking frame from both builds, labelled A and B in a random order per zone,
and was not told what had changed. It found the current build followed the geometry better
in {n_geo_new} of {len(Z)} zones (the 29 September build in {n_geo_old}), preferred it overall in
{n_pref_new} (the old one in {n_pref_old}), and thought it the more beautiful in {n_beauty_new} (the old one in
{n_beauty_old}). It flagged a centre region painted differently in {lens_old} of the 29 September
build's {len(Z)} zones and {lens_new} of the current build's. In its words: “{html.escape(judge['summary'])}”</p>
<p class="note">Flaws it named in the current build, zone by zone (first of each):</p>
<ul class="note">{''.join(f'<li>{html.escape(a.split(";")[0])}</li>' for a in arts_new)}</ul>

<h2>Frame time</h2>
{frame}
<p class="note">Milliseconds (run 1/run 2). Median: {fmt(pb, 'p50')} before, {fmt(pn, 'p50')} now. 95th percentile:
{fmt(pb, 'p95')} before, {fmt(pn, 'p95')} now. Slowest frame: {fmt(pb, 'max')} before, {fmt(pn, 'max')} now.</p>

<h2>Not yet measured</h2>
<p>Nothing here was measured in real Safari, on a real iPhone or on a laptop. Whether the
running average feels calmer to a person is unmeasured; the numbers above are proxies. The
Neural Engine build has no depth input yet.</p>
<footer>Rebuilt by <code>python3 docs/shots/m114/brief.py</code> (in the hypnagogia project) from
<code>docs/shots/m114/data/*.json</code>. Measured {', '.join(dates)} on a Mac mini (M4 Pro, 48 GB): headless WebKit
1280x720 (change, silhouettes, review frames) and Chromium (frame time) through Playwright;
<code>{html.escape(str(eng_n.get('engine')))}</code> ({html.escape(str(eng_n.get('model')))}, {html.escape(str(eng_n.get('device')))}), with
<code>depth_graft</code> 0.8 now and 0 for the 29 September build (client <code>d4cf3a1c</code>). Tools in <code>tools/</code>
and <code>bench/depth_graft.py</code>. Blind review: {html.escape(judge['reviewer'])}.</footer>
</html>
"""
    with open(a.out, "w") as f_:
        f_.write(page)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
