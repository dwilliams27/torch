#!/usr/bin/env python3
"""Build the M115 page (static HTML + inline SVG, no scripts): an idea session, and what
happened when its ideas were tried.

    python3 docs/shots/m115/brief.py --out DIR/hypnagogia-idea-lab.html

Measured numbers are read when this runs, from data/ (bench/cut_census.py, bench/depth_graft.py,
bench/fixed_point.py, eyes.mjs, tools/shoot.mjs) and from the M114 data this session reused;
the idea counts and the ranked list come from docs/ideas/2026-09-30.md. The images are
copied next to the page.
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
IDEAS = os.path.join(HERE, "..", "..", "ideas", "2026-09-30.md")
sys.path.insert(0, os.path.join(HERE, "..", "m111"))
from brief import ZONES, bars  # noqa: E402  (the M111 brief's chart helper)

OLD, NEW = ("#5d6b80", "#8fb8ff")
# the top twelve in plain words (the ranked list's names are its authors' shorthand)
GLOSS = {1: "keep paint from bleeding across the edges of things", 2: "measure the lens and the flicker the way an eye would",
         3: "switch off each suspect for the nausea, one at a time", 4: "cap how fast the picture may change",
         5: "the room is dreamt again while your eyes are shut", 6: "split the model between the GPU and the Neural Engine",
         7: "let the model remember what it painted at each place", 8: "light the painting from the level's real shapes",
         9: "teach one pass to look like thirty", 10: "a smaller model trained to copy SD-Turbo",
         11: "a depth model checks whether the painting kept the pillars", 12: "a wide view that keeps verticals straight"}


def chart(title, svg):
    """A chart with its title on screen (the shared bars() helper only puts it in aria-label)."""
    return f'<p class="ct">{html.escape(title)}</p>{svg}'


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
    for img in ("eyes.jpg", "graft-strength.jpg", "cut-census.jpg", "fixed-point-l2.jpg", "fixed-point-lpips.jpg", "size.jpg"):
        shutil.copy(os.path.join(HERE, img), os.path.join(outdir, "hypnagogia-m115-" + img))

    # the idea session, from the ranked list itself
    md = open(IDEAS).read()
    raw_n = int(re.search(r"(\d+)\s+numbered ideas", md).group(1))
    merged_n = int(re.search(r"leaves (\d+) distinct ideas", md).group(1))
    rows = re.findall(r"^\| (\d+) \| ([^|]+) \| [^|]+ \| (\d) \| ([^|]+) \| ([yn]) \| ([^|]+) \|", md, re.M)
    top = "".join(f"<li><b>{html.escape(name.strip())}</b>: {GLOSS.get(int(n), '')} (payoff {pay}/5, {html.escape(cost.strip())}"
                  f"{', crazy' if crazy == 'y' else ''}; {html.escape(status.split(':')[0].strip())})</li>"
                  for n, name, pay, cost, crazy, status in rows[:12])

    # student cut census
    census = load("cut-census.json")
    cuts = census["cuts"]
    show = ["mida", "mid", "d3.1", "d2.1", "d1.1", "d0.1", "u3.2a", "u3.*a", "bk"]
    plain = {"mida": "mid attention", "mid": "mid block", "d3.1": "down 4, pair 2", "d2.1": "down 3, pair 2", "d1.1": "down 2, pair 2", "d0.1": "down 1, pair 2",
             "u3.2a": "top up attn", "u3.*a": "top up attns", "bk": "BK-style"}
    cut_title = "Model time saved by each cut (%), and how close the picture stays to the uncut model (PSNR, dB)"
    cut_chart = bars(cut_title,
                     [(plain[c], [round(100 * cuts[c]["saving"], 1), cuts[c]["psnr_vs_none"]]) for c in show],
                     [("time saved, %", NEW), ("PSNR, dB", OLD)],
                     f"{census['size'][0]}x{census['size'][1]}, strength {census['strength']}, depth graft on, one frame per view, no training")
    bk = cuts["bk"]
    cut_chart = chart(cut_title, cut_chart)
    kept = [c for c in cuts if c != "none" and cuts[c]["psnr_vs_none"] >= 30]
    kept_best = max(100 * cuts[c]["saving"] for c in kept)

    # graft strength
    gs = load("graft-strength.json")
    views = ["pillar", "vestibule", "nave", "stacks", "baths", "garden", "desert"]
    R = gs["results"]
    st_rows = [(f"strength {s:g}", [round(mean([R[f"{v}|stock|s{s:g}"]["ratio"] for v in views]), 2),
                                    round(mean([R[f"{v}|graft0.8|s{s:g}"]["ratio"] for v in views]), 2)]) for s in gs["strengths"]]
    st_title = "How strongly silhouettes stand out (silhouette step ÷ surface step, mean of 7 views)"
    st_chart = chart(st_title, bars(st_title, st_rows, [("stock SD-Turbo", OLD), ("depth graft 0.8", NEW)],
                                    "one frame per view; higher = the shapes stand apart"))
    gap = mean([g / s_ for _, (s_, g) in st_rows])

    # fixed-point LoRA
    fp = {k: load(f"fixed-point-{k}.json") for k in ("l2", "l1grad", "lpips")}
    fp_title = "Detail in one pass (mean brightness step between neighbouring pixels)"
    fp_chart = bars(fp_title,
                    [("stock", [round(fp["lpips"]["detail"]["stock_one_pass"], 2)]),
                     ("plain LoRA", [round(fp["l2"]["detail"]["lora_one_pass"], 2)]),
                     ("edge LoRA", [round(fp["l1grad"]["detail"]["lora_one_pass"], 2)]),
                     ("LPIPS LoRA", [round(fp["lpips"]["detail"]["lora_one_pass"], 2)]),
                     ("30 passes", [round(fp["lpips"]["detail"]["target"], 2)])],
                    [("detail", NEW)], f"{fp['lpips']['held_out']} held-out framings, 0-255 scale")
    fp_chart = chart(fp_title, fp_chart)
    # (only the LPIPS run's distances are used: the L2 and L1 runs were scored before the
    # evaluation compared both sides after the same decode/encode round trip; detail is
    # measured on the decoded images and compares across all three)
    l2p = fp["lpips"]["held_out_l2"]
    closer = 100 * (1 - l2p["lora_one_pass"] / l2p["stock_one_pass"])
    st_ = {x["frame"]: x["l2"] for x in fp["lpips"]["per_frame"]["stock"]}
    n_closer = sum(x["l2"] < st_[x["frame"]] for x in fp["lpips"]["per_frame"]["lora"])

    # eyes
    ey = load("eyes.json")
    ez = list(ey["zones"])
    eyes_title = "How much the room changed, from before closing to 4 s after opening"
    eyes_chart = chart(eyes_title, bars(eyes_title, [(z, [round(ey["zones"][z]["newDream"], 1), round(ey["zones"][z]["settle24"], 1)]) for z in ez],
                                        [("before closing to +4 s", NEW), ("+2 s to +4 s, eyes open", OLD)],
                                        f"average brightness change per pixel, 0-255; one run, eyes closed {ey['hold']} s"))

    # capture size (graft dividend for speed), if measured
    size_html = ""
    if os.path.exists(os.path.join(HERE, "data", "size-384-1.json")):
        sz = {n: [load(f"size-{n}-{i}.json") for i in (1, 2)] for n in ("384", "320")}
        cap = {n: sorted(set(re.findall(r"capture (\d+x\d+)", open(os.path.join(HERE, "data", f"size-shots-{n}.log")).read()))) for n in ("384", "320")}
        assert cap == {"384": ["512x320"], "320": ["448x256"]}, cap   # the sizes the client chose (tools/shoot.mjs logs)
        f = lambda runs, k1, k2: mean([mean([r["flicker"][z][k1][k2]["mean"] if k1 == "warped" else r["flicker"][z][k1][k2] for z in ZONES]) for r in runs])
        fps = lambda runs: mean([mean([r["flicker"][z]["dreamFps"]["still"] for z in ZONES]) for r in runs])
        # the silhouette ratio's 8-zone mean swings between runs of one build (M115's snap
        # experiment): the median zone, per run
        def sil_runs(runs):
            out = []
            for r in runs:
                v = sorted(r["flicker"][z]["edges"]["silStill"] / r["flicker"][z]["edges"]["texStill"] for z in ZONES)
                out.append((v[3] + v[4]) / 2)
            return out
        sil = lambda runs: mean(sil_runs(runs))
        sil_apart = max(sil_runs(sz["320"])) < min(sil_runs(sz["384"]))
        pairs = [("dreams/s", fps), ("rest detail", lambda r: f(r, "detail", "still")),
                 ("walk detail", lambda r: f(r, "detail", "walking")), ("walk flicker", lambda r: f(r, "warped", "walking")),
                 ("silhouettes", sil)]
        size_title = "448x256 captures against 512x320 (512x320 = 100)"
        size_chart = chart(size_title, bars(size_title, [(k, [100, round(100 * g(sz["320"]) / g(sz["384"]))]) for k, g in pairs],
                                            [("512x320", OLD), ("448x256", NEW)],
                                            "WebKit harness, 8 zones, 2 runs each; walk flicker: lower is better; silhouettes: median zone"))
        rest = f(sz["320"], "detail", "still") / f(sz["384"], "detail", "still") * 100 - 100
        flick = f(sz["320"], "warped", "walking") / f(sz["384"], "warped", "walking") * 100 - 100
        silc = sil(sz["320"]) / sil(sz["384"]) * 100 - 100
        size_html = f"""<h2>Smaller captures: {fps(sz['320']) / fps(sz['384']) * 100 - 100:+.0f}% dreams per second, and blurrier</h2>
<p>With depth holding the architecture, a smaller capture might keep the shapes and dream
faster. 448x256 has 30% fewer pixels than 512x320. It does dream faster, but at rest the
painting loses {-rest:.0f}% of its detail and turns misty, walking flicker rises {flick:.0f}%, and
silhouettes stand out {-silc:.0f}% less{" (both runs below both 512x320 runs)" if sil_apart else " (within the swing between runs)"}.
The game keeps 512x320. The smaller size is still there for a slower machine
(<code>--width 320 --height 320</code>, a pixel budget the client shapes into 448x256).</p>
{size_chart}
<figure><img src="hypnagogia-m115-size.jpg" alt="Three rooms at rest: 512x320 captures on the left, 448x256 on the right">
<figcaption>At rest after the same walk: 512x320 captures (left) and 448x256 (right). Rows: nave, stacks, garden.</figcaption></figure>"""

    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hypnagogia: an idea session, tried</title>
<style>body{{background:#0e1117;color:#e6e9ef;font:17px/1.55 system-ui,-apple-system,sans-serif;margin:0 auto;padding:24px 18px 48px;max-width:980px}}
a{{color:#9ec9ff}}h1{{font-size:clamp(1.9rem,5vw,2.8rem);line-height:1.15;margin:.6rem 0}}h2{{margin-top:2.2rem;font-size:1.25rem}}
img{{width:100%;border-radius:8px;display:block}}figure{{margin:1rem 0}}figcaption,.note{{color:#98a2b3;font-size:.9rem}}svg{{width:100%;height:auto;margin:.6rem 0}}
svg text{{fill:#c9d1dc;font:12px system-ui,sans-serif}}svg .k{{fill:#98a2b3}}svg .v{{fill:#e6e9ef}}li{{margin:.25rem 0}}
footer{{margin-top:2.5rem;color:#98a2b3;font-size:.85rem}}.ct{{margin:1.2rem 0 0;font-size:.95rem;color:#c9d1dc}}</style>
<a href="./index.html">← Showcase</a>
<h1>An idea session, and what happened when we tried the ideas</h1>
<p>You asked for crazy technical work and for idea sessions that go off the beaten path. Five
fresh Claude sessions each wrote ideas for Hypnagogia from one angle: raw speed,
rendering, perception and comfort, how the model is conditioned, and play. That gave
{raw_n} ideas, {merged_n} after merging duplicates, ranked by a guessed payoff (1 to 5) against
cost, with a few weeks-long ideas moved up by hand. The list lives in the project at
<code>docs/ideas/2026-09-30.md</code>. Its top twelve, with where each stands:</p>
<ol>{top}</ol>
<p>Then the ideas met the machine. One skipped the ranking because it went into the game
that day: the depth graft that makes pillars stand out (<a href="hypnagogia-lucid-dream.html">its
own page</a>). Below, four of the twelve (1, 5, 9 and 10) and four other tests each got an
experiment and a verdict; the rest wait their turn.</p>

<h2>Close your eyes, now in the game</h2>
<p>Hold E, or tap "close your eyes" in the settings panel on a phone. The view sinks into a
warm eyelid glow while the model dreams the room again with a variant prompt (overgrown,
winter, glass, dawn, and more); open, and the same architecture wears a new dream. Each zone
keeps its variant until you blink there again.</p>
<figure><img src="hypnagogia-m115-eyes.jpg" alt="Four zones before closing the eyes, one and four seconds after opening, and after a second blink">
<figcaption>Left to right: before, 1 s after opening (overgrown prompt), 4 s after, and after a second blink (winter prompt). Rows: nave, baths, garden, desert.</figcaption></figure>
<p>The chart puts a number on the new dream: how far each pixel's brightness moved between
the view before closing and the view 4 s after opening (blue), against the room's own
drift with the eyes open, from 2 s to 4 s after opening (grey).</p>
{eyes_chart}

<h2>More strength, same shapes</h2>
<p>Strength is how far the model may stray from the capture. With the depth graft,
silhouettes stand out about {gap:.1f} times as strongly as with stock SD-Turbo at every strength
tried, up to 0.85. Higher strength paints richer detail, so the "fever" look runs at 0.74.</p>
{st_chart}
<figure><img src="hypnagogia-m115-graft-strength.jpg" alt="Five views: the input, then the depth graft at strengths 0.66, 0.75 and 0.85">
<figcaption>Left to right: what the model is given, then the depth graft at strength 0.66, 0.75 and 0.85.</figcaption></figure>

<h2>Cutting SD-Turbo down without retraining</h2>
<p>A smaller model is the long road to more dreams per second on a laptop or a phone. The
first step: switch parts of the model off, without retraining, and see what each costs.
Cuts that leave the picture nearly intact (PSNR 30 dB or more) save {kept_best:.0f}% at most,
and each cut was timed once, so savings that small are rough. A cut modelled on BK-SDM (a
published slimmed-down Stable Diffusion) saves {100 * bk['saving']:.0f}% of a full pass of the
model but, untrained, shatters the picture (PSNR {bk['psnr_vs_none']:.1f} dB); in the game, where
DeepCache already skips the deep levels on about two frames in three, it would save less. A
faster model needs a smaller one trained to copy SD-Turbo (a distilled student): weeks of
work.</p>
{cut_chart}
<figure><img src="hypnagogia-m115-cut-census.jpg" alt="Three views with no cut, the mid block cut, a middle pair cut, the top attention cut and the BK-style cut">
<figcaption>Left to right: uncut, mid block cut, the third down block's second pair cut, the top up block's last attention cut, the BK-style cut. Rows: three views.</figcaption></figure>

<h2>Can one pass look like thirty? A pilot</h2>
<p>When you stand still, a view converges over about thirty passes into a finished
painting; when you walk, each view gets one pass. A small LoRA was trained so that one pass
lands closer to where thirty would, on {fp['lpips']['framings'] - fp['lpips']['held_out']} views harvested from the game, and
tested on {fp['lpips']['held_out']} views it wasn't trained on (each next to training views along the walk,
so an easy test). At {fp['lpips']['step_s_median']:.2f} s a training step on the mini, 5,000 steps take about 40 minutes;
the pilot ran {fp['lpips']['steps']}. Trained on a plain average error, the single pass blurred (its detail fell from
{fp['l2']['detail']['stock_one_pass']:.1f} to {fp['l2']['detail']['lora_one_pass']:.1f}), and an edge term didn't help. Trained on a perceptual error (LPIPS), it stayed sharp
and moved slightly toward the finished look: {closer:.0f}% closer on average, closer in {n_closer} of
{fp['lpips']['held_out']} views. The teal lights and glows of the thirty-pass paintings start to appear
after one. M116 takes this further on the game's own walking views.</p>
{fp_chart}
<figure><img src="hypnagogia-m115-fixed-point-lpips.jpg" alt="Six views: the raw capture, one pass of the stock model, one pass with the perceptual-loss LoRA, and the thirty-pass target">
<figcaption>Left to right: what the model is given, one stock pass, one pass with the LoRA (perceptual loss), the thirty-pass target.</figcaption></figure>
<figure><img src="hypnagogia-m115-fixed-point-l2.jpg" alt="The same six views with the plain-error LoRA, which blurs">
<figcaption>The same with the plain average error: blurred.</figcaption></figure>
{size_html}
<h2>Tried and dropped</h2>
<ul>
<li><b>Zooming the middle of each capture</b> (the foveal warp): no seam, but less detail even in the middle; off.</li>
<li><b>Rebuilding the atlas's mipmaps less often</b>: no measurable change in frame time or dream rate; reverted.</li>
<li><b>Paint snapped to the geometry at silhouettes</b>: no difference beyond run-to-run noise, and none visible; reverted.</li>
</ul>

<h2>Not yet measured</h2>
<p>The LoRA inside the game; an in-game blind review of the close-your-eyes dreams;
everything on real Safari, a real iPhone or a laptop.</p>
<footer>Rebuilt by <code>python3 docs/shots/m115/brief.py</code> (in the hypnagogia project) from
<code>docs/shots/m115/data/*.json</code> and <code>docs/ideas/2026-09-30.md</code>. Measured 2026-09-30 on a Mac mini
(M4 Pro, 48 GB) with <code>bench/cut_census.py</code>, <code>bench/depth_graft.py</code>, <code>bench/fixed_point.py</code>,
<code>docs/shots/m115/eyes.mjs</code> and <code>tools/shoot.mjs</code> (headless WebKit); torch_turbo on MPS with the depth graft.</footer>
</html>
"""
    with open(a.out, "w") as f_:
        f_.write(page)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
