---
title: IndoSmish API
emoji: 📱
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Tri-class code-mixed Malayalam-English smishing detection API
---

# IndoSmish API

FastAPI + ONNX inference backend for tri-class **ham / spam / smishing** detection on
code-mixed Malayalam–English SMS. Serves a fine-tuned IndicBERT encoder via onnxruntime.

Implements the deployment layer of *“From SMS Spam Filters to Quantized LLMs”* (IEEE
Access 2026). The polished web UI lives separately on Vercel and calls this API.

## Endpoints

- `GET /health` → `{"status":"ok","model_loaded":true}`
- `POST /classify` with `{"text": "..."}` →
  `{"label", "probabilities", "cues", "latency_ms", "precision"}`

## Try it

```bash
curl -X POST https://<this-space>.hf.space/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"ningalude account block aayi, ee link click cheyyu"}'
```
