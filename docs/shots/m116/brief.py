#!/usr/bin/env python3
"""Build the M116 page (static HTML + inline SVG, no scripts): does walking look as
finished as standing still, and what closed the gap.

    python3 docs/shots/m116/brief.py --out DIR/hypnagogia-one-pass.html

Every number is read when this runs, from data/: bench/walk_gap.py on harvests made by
tools/harvest_pairs.mjs (gap-harvest1.json, floor.json, probes.json), bench/one_pass.py
(lora.json), tools/shoot.mjs --flicker (flicker-*.json) and two blind reviews (judge-fresh.json,
judge.json, each with its key). A section whose data is missing says so. Images are copied
next to the page.
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

OLD, NEW, THIRD = ("#5d6b80", "#8fb8ff", "#c9a2ff")


def load(name):
    p = os.path.join(HERE, "data", name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def chart(title, svg):
    return f'<p class="ct">{html.escape(title)}</p>{svg}'


def flick(tag):
    """two tools/shoot.mjs --flicker runs of one build -> 8-zone means, run-averaged"""
    runs = [r for r in (load(f"flicker-{tag}-{i}.json") for i in (1, 2)) if r]
    if not runs:
        return None
    zone = lambda r, z, f: f(r["flicker"][z])
    m = lambda f: mean([mean([zone(r, z, f) for z in ZONES]) for r in runs])
    return {"runs": len(runs), "fps_walk": m(lambda z: z["dreamFps"]["walking"]), "fps_still": m(lambda z: z["dreamFps"]["still"]),
            "change_walk": m(lambda z: z["warped"]["walking"]["mean"]), "change_still": m(lambda z: z["still"]["mean"]),
            "detail_walk": m(lambda z: z["detail"]["walking"]), "detail_still": m(lambda z: z["detail"]["still"]),
            "dc_reuse": mean([r.get("dcReuseAfter") for r in runs]),
            "engine": runs[0].get("engine"), "client": runs[0].get("client"), "date": runs[0].get("date")}


def PROBE_TEXT(pr, fd, fh):
    reuse = f"{100 * fd['dc_reuse']:.0f}% of frames" if fd and fd.get("dc_reuse") else "most frames"
    more = f"{100 * (fh['change_walk'] / fd['change_walk'] - 1):.0f}% more" if fd and fh else "more"
    g = lambda k: pr[k]["all"]["lpips"]
    part = (f"about {100 * (g('p0-waking') - g('p4-fresh')) / (g('p0-waking') - g('p3-neither')):.0f}% as much"
            if pr and all(k in pr for k in ("p0-waking", "p3-neither", "p4-fresh")) else "less")
    return f"""<p>DeepCache is one of the engine's speed tricks: on {reuse} the model skips its deep
layers and reuses what the last full pass computed. Standing still that is the same view, so it
costs nothing. Walking, it is the previous view, and the new frame inherits its layout.
Cross-frame attention, the other trick that looks back at the last frame, barely moves the gap on
its own. Turning both off for new views closes the most, but in the game it made walking flicker
{more} (below), because cross-frame attention is what keeps one frame consistent with the next.
Turning off only DeepCache's reuse, on new views only, closes {part} and keeps the flicker where
it was. That is the fresh look. The cost is speed: new views need full passes, so the game dreams
fewer frames a second while you walk.</p>"""


def LORA_TEXT(lo, h, fl, g):
    closer = 100 * (1 - h["lpips"]["lora"] / h["lpips"]["stock"])
    room = 100 * (1 - fl["lpips"] / h["lpips"]["stock"]) if fl else None
    det = 100 * (h["detail"]["lora"] / h["detail"]["stock"] - 1)
    rest = h["rest_lpips"]["lora"] / h["rest_lpips"]["stock"]
    fit = 100 * (1 - lo["loss_last100"] / lo["loss_first100"])
    return f"""<p>The mission's first plan was a LoRA: a small add-on to the model, trained on these
walks, that pulls one pass toward the rest view. A plain pass (one full pass of the model, both
speed tricks off) already lands much nearer the rest view than the frame the game showed while
walking. Trained on {lo['train']} harvested views and tested on {lo['held_out_n']} from stretches of the tour it never
saw, the LoRA brought one pass {closer:.0f}% closer than a plain pass{f" (the floor caps any gain at {room:.0f}%)" if room is not None else ""},
with {abs(det):.0f}% {"more" if det >= 0 else "less"} detail. Its training loss fell {fit:.0f}%, but on new views it gained
only that {closer:.0f}%. Standing still, it moved the picture {rest:.0f} times as far from where it had settled as a
plain pass does. So it stays off. Much of what separates a walking frame from the rest view is what
the {mean([x["passes"] for x in g["pairs"]]):.0f} or so passes a standing view gets go on to invent (the light rays in the second row
below), which no single pass can know in advance.</p>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    outdir = os.path.dirname(os.path.abspath(a.out))
    for img in ("gap.jpg", "ab.jpg", "lora.jpg"):
        if os.path.exists(os.path.join(HERE, img)):
            shutil.copy(os.path.join(HERE, img), os.path.join(outdir, "hypnagogia-m116-" + img))
    img = lambda name, alt, cap: (f'<figure><img src="hypnagogia-m116-{name}" alt="{html.escape(alt)}"><figcaption>{cap}</figcaption></figure>'
                                  if os.path.exists(os.path.join(HERE, name)) else "")

    # 1. the gap, over a whole harvest
    g = load("gap-harvest1.json")
    g = g and next(iter(g.values()))
    fl = load("floor.json")
    gap_html = "<p>Not yet measured.</p>"
    if g:
        zs = [z for z in ZONES if z in g["zones"]]
        gap_chart = chart("How different the view is 5 s after you stop (LPIPS; 0 = the same)",
                          bars("gap", [(z, [round(g["zones"][z]["lpips"], 3)]) for z in zs], [("walking view to rest view", NEW)],
                               f"{g['all']['n']} views along the whole tour; LPIPS (AlexNet)"))
        gap_html = f"""<p>We walked the whole tour and, at {g['all']['n']} places along it, kept the frame the model
painted for you while walking. Then we stood at each of those camera positions for five seconds, as you
would if you stopped, and kept what the view had become. On average the two are {g['all']['lpips']:.2f} apart
on a perceptual scale where 0 is identical, and the rest view carries
{100 * (g['all']['detail_target'] / g['all']['detail_walk'] - 1):.0f}% more detail.</p>
{gap_chart}"""
        if fl:
            gap_html += f"""<p>The rest view is not one fixed picture, though. Holding {fl['n']} of the same
views again, starting from a different state of the world, gives rest views {fl['lpips']:.2f} apart.
That is the floor: a walking frame can't be expected to land closer to "the" rest view than two
rest views land to each other.</p>"""

    # 2. where it comes from
    pr = load("probes.json")
    probe_html = "<p>Not yet measured.</p>"
    if pr:
        names = [("p0-waking", "as shipped (waking)"), ("p1-noxframe", "no cross-frame attention"), ("p2-nodeepcache", "no DeepCache at all"),
                 ("p3-neither", "new views: neither trick"), ("p4-fresh", "new views: no DeepCache (fresh)")]
        have = [(k, lab) for k, lab in names if k in pr]
        rows = "".join(f"<tr><td>{html.escape(lab)}</td><td>{pr[k]['all']['lpips']:.2f}</td><td>{pr[k]['walk']['dreamsPerSecond']:.1f}</td>"
                       f"<td>{pr[k]['all']['n']}</td></tr>" for k, lab in have)
        probe_html = f"""<table><tr><th>engine</th><th>walking-to-rest gap</th><th>dreams per second, walking</th><th>views</th></tr>{rows}</table>
<p class="note">The tour's first stretch (vestibule, nave, stacks), walked at 3.4 m/s; each build walked and
held on its own. Gap: perceptual distance (LPIPS) between the walking frame and the same view 5 s after
stopping; the floor, measured on other views, is {f"{fl['lpips']:.2f}" if fl else "not measured"}. The two whole-engine switches
(no cross-frame attention, no DeepCache) ran on an earlier client whose default look behaves the same; with
no DeepCache at all, holds got {f"{pr['p2-nodeepcache']['passes_median']:.0f}" if "p2-nodeepcache" in pr else "fewer"} passes in 5 s against
{f"{pr['p0-waking']['passes_median']:.0f}" if "p0-waking" in pr else "more"} for the shipped engine, so its rest views had settled less.</p>"""
    # 3. the LoRA
    lo = load("lora.json")
    lora_html = "<p>Not yet measured.</p>"
    if lo:
        h = lo["held_out"]
        lora_chart = chart("Held-out views: perceptual distance to the rest view (LPIPS; lower = closer)",
                           bars("lora", [("shown walking", [round(h["lpips"]["walk"], 3)]), ("plain pass", [round(h["lpips"]["stock"], 3)]),
                                         ("with LoRA", [round(h["lpips"]["lora"], 3)])] +
                                ([("floor", [round(fl["lpips"], 3)])] if fl else []),
                                [("LPIPS", NEW)], f"{lo['held_out_n']} held-out views; {lo['steps']} steps on {lo['train']} views, rank {lo['rank']}; "
                                                  f"floor: {fl['n'] if fl else 0} other views held twice"))
        lora_html = LORA_TEXT(lo, h, fl, g) + lora_chart + img("lora.jpg", "Held-out views: the walking capture, the frame served while walking, one pass with the LoRA, and the view at rest",
                                                             "Left to right: what the model was given while walking, what the game showed, one pass with the LoRA, the view 5 s after stopping.")

    # 4. in the game
    fd, fh, fc = flick("waking"), flick("neither"), flick("fresh")
    judge, key = load("judge-fresh.json"), load("judge-fresh-key.json")   # waking against fresh
    game_html = "<p>Not yet measured.</p>"
    count = lambda J, K, q, who: sum(1 for z in J if J[z][q] in ("A", "B") and K[z][J[z][q]] == who)
    evens = lambda J, q: sum(1 for z in J if J[z][q] == "even")
    won = lambda J, K, who: ", ".join(z for z in J if J[z]["preferred"] in ("A", "B") and K[z][J[z]["preferred"]] == who) or "no zone"
    if fd and fh:
        cols = [("waking (as shipped)", fd)] + ([("fresh", fc)] if fc else []) + [("neither trick on new views", fh)]
        rows = [("dreams per second, walking", "fps_walk", 1), ("dreams per second, standing", "fps_still", 1),
                ("flicker while walking (lower is calmer)", "change_walk", 2), ("flicker standing", "change_still", 2),
                ("detail walking", "detail_walk", 2), ("detail standing", "detail_still", 2)]
        cell = lambda f, k, n: f"{f[k]:.{n}f}" + ("" if f is fd else f" ({100 * (f[k] / fd[k] - 1):+.0f}%)")
        tr = "".join(f"<tr><td>{lab}</td>" + "".join(f"<td>{cell(f, k, n)}</td>" for _, f in cols) + "</tr>" for lab, k, n in rows)
        game_html = f"""<table><tr><th></th>{"".join(f"<th>{html.escape(c)}</th>" for c, _ in cols)}</tr>{tr}</table>
<p class="note">tools/shoot.mjs --flicker, WebKit 1280x720, 8 zones, two runs each, one client and one server
build; walking at 1.7 m/s (the tour's pace). Flicker: mean |Δ luma| between consecutive frames after
reprojecting the earlier one, 0-255. Detail: mean luma gradient.</p>"""
        if judge and key:
            J = judge["zones"]
            new = next(v for z in key for v in key[z].values() if v != "waking")
            game_html += f"""<p>A new Claude session, not told which build was which or what had changed, compared
walking filmstrips of waking and fresh in all {len(J)} zones, with the undreamt geometry beside each frame.
It would rather walk through fresh in {won(J, key, new)}, and through waking in {won(J, key, 'waking')}. Walking frames
looked more finished with fresh in {count(J, key, 'finished', new)} zones and with waking in {count(J, key, 'finished', 'waking')}; geometry split
{count(J, key, 'geometry', new)} to {count(J, key, 'geometry', 'waking')}. Where it saw differences, waking washed out the vestibule's outer frames,
turned the baths' walls into a magenta wash and smeared the garden, while fresh mottled a globe in the
stacks and left a ghostly slab in the atrium, where waking kept the arched doorway. But it also judged the
standing views different in {count(J, key, 'rest', new) + count(J, key, 'rest', 'waking')} zones, where the two looks differ very little: one
filmstrip per zone is a noisy judge, so read this as "no worse", not as a win.</p>"""
        game_html += img("ab.jpg", "Walking frames from the same walk: the waking look and the fresh look",
                         "The same moments of the same walk: waking, as shipped (left), and fresh (right). Rows: nave, baths, garden, desert. Only the garden differs much.")

    CHANGED = "<p>Not yet measured.</p>"
    if fd and fc and pr and "p0-waking" in pr and "p4-fresh" in pr:
        pct = lambda a, b: 100 * (a / b - 1)
        steady = lambda x: f"unchanged ({x:+.1f}%)" if abs(x) < 2 else f"{x:+.0f}%"
        g0, g4 = pr["p0-waking"]["all"]["lpips"], pr["p4-fresh"]["all"]["lpips"]
        CHANGED = f"""<p>A sixth look, <b>fresh</b> (key 6, or the panel): while you walk, each new view gets a full pass
of the model instead of borrowing the last view's deep layers. Walking frames carry {pct(fc['detail_walk'], fd['detail_walk']):.0f}% more detail
and land {-pct(g4, g0):.0f}% nearer what you'll see when you stop ({g0:.2f} to {g4:.2f} in the first three rooms), with
walking flicker {steady(pct(fc['change_walk'], fd['change_walk']))}. The cost is dreams per second while walking: {-pct(fc['fps_walk'], fd['fps_walk']):.0f}% fewer at
the tour's pace, {-pct(pr['p4-fresh']['walk']['dreamsPerSecond'], pr['p0-waking']['walk']['dreamsPerSecond']):.0f}% fewer at full walking speed, where every view is new;
standing still the rate is {steady(pct(fc['fps_still'], fd['fps_still']))}, and the picture changes {pct(fc['change_still'], fd['change_still']):.0f}% more there (the
first frame of each sideways glance is a new view, so it gets a full pass too). Only the fresh look changes
what the engine does. Waking stays the
default until you decide. The LoRA stays in the bench, off.</p>"""
    body = f"""<p>You asked for walking to look as finished as standing still. This is what stands between
the two, measured in the game, and what closed part of it.</p>
{gap_html}
{img("gap.jpg", "Four views: what the game showed while walking, and the same view five seconds after stopping", "Left: the frame shown while walking. Right: the same camera, five seconds after stopping. Rows: nave, geode, baths, desert.")}
<h2>Where the gap comes from</h2>
{PROBE_TEXT(pr, fd, fh)}
{probe_html}
<h2>Can training close the gap?</h2>
{lora_html}
<h2>In the game: the fresh look</h2>
{game_html}
<h2>What changed</h2>
{CHANGED}
<h2>Not yet measured</h2>
<p>How the fresh look feels to a person while walking; real Safari, a real iPhone or a laptop; a
LoRA trained on fresh-look walks; whether a faster full pass (a smaller model) could give the
fresh look's frames without its speed cost.</p>"""

    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hypnagogia: what changes when you stop walking</title>
<style>body{{background:#0e1117;color:#e6e9ef;font:17px/1.55 system-ui,-apple-system,sans-serif;margin:0 auto;padding:24px 18px 48px;max-width:980px}}
a{{color:#9ec9ff}}h1{{font-size:clamp(1.9rem,5vw,2.8rem);line-height:1.15;margin:.6rem 0}}h2{{margin-top:2.2rem;font-size:1.25rem}}
img{{width:100%;border-radius:8px;display:block}}figure{{margin:1rem 0}}figcaption,.note{{color:#98a2b3;font-size:.9rem}}svg{{width:100%;height:auto;margin:.6rem 0}}
svg text{{fill:#c9d1dc;font:12px system-ui,sans-serif}}svg .k{{fill:#98a2b3}}svg .v{{fill:#e6e9ef}}li{{margin:.25rem 0}}
footer{{margin-top:2.5rem;color:#98a2b3;font-size:.85rem}}.ct{{margin:1.2rem 0 0;font-size:.95rem;color:#c9d1dc}}
table{{border-collapse:collapse;width:100%;font-size:.92rem;margin:.8rem 0}}td,th{{border-bottom:1px solid #2a3140;padding:.35rem .4rem;text-align:left}}
th{{color:#98a2b3;font-weight:500}}</style>
<a href="./index.html">← Showcase</a>
<h1>What changes when you stop walking</h1>
{body}
<footer>Rebuilt by <code>python3 docs/shots/m116/brief.py</code> (in the hypnagogia project) from
<code>docs/shots/m116/data/*.json</code>. Measured 2026-09-30 on a Mac mini (M4 Pro, 48 GB) with
<code>tools/harvest_pairs.mjs</code>, <code>bench/walk_gap.py</code>, <code>bench/one_pass.py</code> and
<code>tools/shoot.mjs</code> (headless WebKit); torch_turbo on MPS with the depth graft.</footer>
</html>
"""
    with open(a.out, "w") as f_:
        f_.write(page)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
