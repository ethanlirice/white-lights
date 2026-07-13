# Deploying the real backend (free)

The hosted GitHub Pages link is a **simulated demo** — GitHub Pages can't run the
pose model. To put the *real* judge online you need a host that runs a Python +
PyTorch container with WebSocket support. The whole app (pages **and**
`/ws/live`) serves from one FastAPI process, so it deploys as a **single
container** — the included [`Dockerfile`](../Dockerfile) has the model baked in.

## Recommended free host: Hugging Face Spaces

Why: the free CPU tier is **2 vCPU / 16 GB RAM** — enough for torch (most free
tiers cap at ~512 MB and can't load it) — with native Docker + WebSocket support,
automatic HTTPS, and a public URL. Trade-off: a free Space **sleeps when idle**
(~30–60 s cold start on the next hit).

### Steps

1. **Create the Space** at <https://huggingface.co/new-space> → **SDK: Docker**,
   Hardware: **CPU basic (free)**.
2. **Add the Space config.** A Docker Space's `README.md` needs this frontmatter
   at the very top (keep it on the *Space's* README so the GitHub README stays
   clean):
   ```
   ---
   title: White Lights
   emoji: 
   colorFrom: gray
   colorTo: green
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
3. **Push the code to the Space** (it's a git repo). From this project:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/white-lights
   git push space main
   ```
   (Or clone the Space and copy the project files in.)
4. The Space builds the `Dockerfile` and starts on port **7860**. Open the Space
   URL → **`/live`** → real webcam judging over HTTPS.

Everything is served by the one app: `/`, `/live`, `/history`, `/stats`,
`/landing`, and the `/ws/live` WebSocket (upgraded to `wss://` automatically). The
frontend connects to `location.host + '/ws/live'`, so no config is needed.

## Test the container locally first

```bash
docker build -t white-lights .
docker run --rm -p 7860:7860 white-lights
# → http://127.0.0.1:7860/live
```

## Notes & alternatives

- **getUserMedia needs HTTPS** — HF Spaces (and any real host) provides it; only
  `localhost` is exempt.
- **Latency:** real-time webcam → remote CPU → back is laggier than local; fine
  for a demo, not competition-grade.
- **Other hosts:** Render / Railway / Fly.io work the same way (same Dockerfile)
  but their free tiers are RAM-limited or now require a card; a small VPS
  ($5/mo) behind Caddy gives full control.
- **Slim it down (future):** exporting YOLO to ONNX + onnxruntime drops torch
  (~2 GB → ~200 MB, faster CPU inference) and would fit almost any free tier — see
  [ROADMAP.md](ROADMAP.md).
