# Hypnagogia: agent notes

A browser 3D game whose visuals a local diffusion model paints onto the world. Read
`docs/DESIGN.md` first. It holds the architecture, the frozen WebSocket protocol and
Engine interface, and the client contracts. `README.md` is the public face.

- Taste first: the UI is light-touch and ambient. No strobe, no clutter. Look at a
  screenshot before claiming a visual change works.
- The render loop never waits on diffusion. Check `?perf=1` after client changes.
- Numbers come from `bench/` scripts, with machine, engine, size and date.
- `legacy/diffused-rays/` is frozen (the "before" picture).
- This directory is published verbatim to the public repo `dwilliams27/torch`, except
  `CHARTER.md`, `docs/STATUS.md` and `docs/journal/`. Never put secrets, private
  hostnames or personal details anywhere else in it.

Quick checks: `./run.sh --engine mock --port PORT` (binds 127.0.0.1; inside sink the
charter assigns each lane its port), then open `http://127.0.0.1:PORT/?autopilot=1`. World invariants:
`node client/src/world/tools/stats.mjs 1 --all`.
