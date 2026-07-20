# Deployment — Vercel (frontend) + Render (backend API)

The live demo is a static frontend on **Vercel** that calls a FastAPI + ONNX backend on
**Render**. The model (IndicBERT fine-tuned, exported to fp32 ONNX) runs on CPU under
onnxruntime — no torch — so it fits Render's free 512 MB tier and demonstrates the paper's
on-device efficiency thesis (~10–20 ms/message on CPU).

```
Browser ─▶ Vercel (app/frontend, static)  ──POST /classify──▶  Render (app/backend, FastAPI+ONNX)
```

## Step 1 — Host the model (once)

The fp32 ONNX model (~128 MB) is too big for GitHub, so host it on HuggingFace and let
Render download it at startup.

```powershell
$env:HF_TOKEN = "hf_...write-scope..."
python scripts/push_model_to_hf.py --repo <your-hf-username>/indosmish-indicbert-onnx
```

## Step 2 — Deploy the backend on Render

1. Push this repo to GitHub.
2. Render → **New → Web Service** → connect the repo.
3. Settings:
   - **Root Directory:** `app/backend`
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
4. Environment variables:
   - `MODEL_REPO` = `<your-hf-username>/indosmish-indicbert-onnx`
   - `ALLOW_ORIGIN` = `*` (tighten to your Vercel URL after step 3)
5. Deploy. Test: `https://<your-service>.onrender.com/health` → `{"status":"ok"}`.

(Or use the included `app/backend/render.yaml` Blueprint.)

Note: Render free spins down after ~15 min idle; the first request after that takes
~30–50 s to wake. The frontend shows a "waking up" hint on timeout.

## Step 3 — Deploy the frontend on Vercel

1. Edit `app/frontend/config.js`:
   `window.INDOSMISH_API = "https://<your-service>.onrender.com";`
2. Vercel → **New Project** → import the repo.
   - **Root Directory:** `app/frontend`
   - Framework preset: **Other** (it's static — no build step).
3. Deploy. Open the Vercel URL and classify a message.
4. Back on Render, set `ALLOW_ORIGIN` to your Vercel URL and redeploy (locks down CORS).

## Local testing

```powershell
# backend
cd app/backend; $env:MODEL_DIR="model/onnx"
uvicorn main:app --port 8000
# frontend (separate terminal)
cd app/frontend; python -m http.server 5500
# open http://localhost:5500
```
